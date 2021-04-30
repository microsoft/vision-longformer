import time
import unittest
from typing import Union
import torch
import numpy as np
import random
from functools import lru_cache
from einops import rearrange, repeat
import torch.nn.functional as F
from torch import nn, einsum


@lru_cache()
def get_2dmask(seq_len, nx, ny, w, d):
    return torch.BoolTensor([
        [
            abs(i // ny - j // ny) > w or abs(i % ny - j % ny) > w or (i // ny - j // ny)%d or (i % ny - j % ny)%d for j in range(seq_len)
        ]
        for i in range(seq_len)
    ], device='cpu')


def naive2d_matmul_qk(q, k, nx, ny, w, d, padding=0.0):
    bsz, num_heads, seq_len, head_dim = q.size()
    attn_weights = q @ k.transpose(-2, -1)
    # get mask
    mask = get_2dmask(seq_len, nx, ny, w, d).to(q.device)
    mask = mask[None, None, :, :]
    attn_weights.masked_fill_(mask, padding)
    return attn_weights


def _get_invalid_locations_mask_fixed_dilation(seq_len: int, nx: int, ny: int, w: int, d: int):
    c1d = 2 * w + 1
    c = 2 * w * (w + 1)
    return torch.BoolTensor([
        [
            i // ny + d * (j // c1d - w) < 0 or i % ny + d * (j % c1d - w) < 0 or i % ny + d * (j % c1d - w) >= ny
            for j in range(c)
        ]
        for i in range(seq_len)
    ], device='cpu')


@lru_cache()
def _get_invalid_locations_mask(seq_len: int, nx: int, ny: int, w: int, d: Union[torch.Tensor,int], autoregressive: bool, device: str):
    if isinstance(d, int):
        mask = _get_invalid_locations_mask_fixed_dilation(seq_len, nx, ny, w, d)
        mask = mask[None, None, :, :]
        num_invalid = mask.sum()
    else:
        head_masks = []
        head_invalids = []
        d_list = d.cpu().numpy().tolist()
        for d in d_list:
            one_head_mask = _get_invalid_locations_mask_fixed_dilation(seq_len, nx, ny, w, d)
            head_masks.append(one_head_mask)
            head_invalids.append(one_head_mask.sum())
        mask = torch.stack(head_masks, dim=0)
        num_invalid = torch.stack(head_invalids, dim=0)
        mask = mask[None, :, :, :]

    ending_mask = None if autoregressive else mask.flip(dims=(2, 3)).to(device)
    end_num_invalid = None if autoregressive else num_invalid.to(device)

    return mask.to(device), ending_mask, num_invalid.to(device), end_num_invalid


def mask_invalid_locations(input_tensor: torch.Tensor, nx: int, ny: int, w: int, d: Union[torch.Tensor, int], autoregressive: bool) -> torch.Tensor:
    seq_len = input_tensor.size(2)
    beginning_mask, ending_mask, num_invalid, end_num_invalid = \
        _get_invalid_locations_mask(seq_len, nx, ny, w, d, autoregressive,
                                       input_tensor.device)
    c = 2 * w * (w + 1)
    beginning_input = input_tensor[:, :, :, :c]
    beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_input.masked_fill_(beginning_mask, -float('inf'))
    if not autoregressive:
        ending_input = input_tensor[:, :, :, -c:]
        ending_mask = ending_mask.expand(ending_input.size())
        ending_input.masked_fill_(ending_mask, -float('inf'))
        num_invalid = num_invalid + end_num_invalid

    return num_invalid


@lru_cache()
def _get_invalid_locations_mask_offical(nx: int, ny: int, w: int, d: int, autoregressive: bool, device: str):
    img_seq = torch.arange(nx * ny)
    k_img_indices = rearrange(img_seq.float(), '(h w) -> () () h w', h=nx)
    k_img_indices = F.pad(k_img_indices, (w * d,) * 4,
                          value=nx * ny)  # padding set to be max, so it is never attended to
    k_img_indices = F.unfold(k_img_indices, 2 * w + 1, dilation=d)
    k_img_indices = rearrange(k_img_indices, 'b j i -> b i j')

    if autoregressive:
        q_img_indices = rearrange(img_seq, 'i -> () i ()')
        mask = q_img_indices >= k_img_indices
    else:
        mask = k_img_indices >= nx * ny

    num_invalid = mask.sum()

    return mask.to(device), num_invalid.to(device)


def mask_invalid_locations_offical(input_tensor: torch.Tensor, nx: int, ny: int, w: int, d: int, autoregressive: bool) -> torch.Tensor:
    mask, num_invalid = _get_invalid_locations_mask_offical(
        nx, ny, w, d, autoregressive, input_tensor.device
    )
    input_tensor.masked_fill_(mask, -float('inf'))
    return num_invalid


def same_storage(x, y):
    '''Tests if two tensors share the same underlying storage (for memory optimizations)'''
    return x.storage().data_ptr() == y.storage().data_ptr()


class TestSlidingChunksMM(unittest.TestCase):
    def test_tvm_equal_naiven2(self):
        np.random.seed(300)
        random.seed(300)
        torch.manual_seed(300)
        torch.cuda.manual_seed(300)
        torch.cuda.manual_seed_all(300)

        torch.set_printoptions(sci_mode=False)
        nx = 30
        ny = 26
        N = nx * ny  # * 16
        M = 64  # hidden size
        W = 8  # one sided. Actual window size = (2w+1)**2
        nlocal = (2 * W + 1) ** 2
        B = 2
        D = 1  # no dilation
        padding = W * D
        H = 12  # number of heads
        autoregressive = False  # not autoregressive
        device = 'cuda'
        dtype = torch.float32

        failed_tests = 0
        time1 = time2 = 0
        for i in range(100):
            if i < 5:
                time1 = time2 = 0  # don't include the first few iterations because of high variance

            query = torch.randn(B * H * N * M, requires_grad=True, device=device, dtype=dtype).view(B, H, N, M)
            query.retain_grad()
            key = torch.randn(B * H * N * M, requires_grad=True, device=device, dtype=dtype).flip(dims=(0,)).view(B, H, N, M)
            key.retain_grad()
            value = torch.randn(B * H * N * M, requires_grad=True, device=device, dtype=dtype).view(B, H, N, M)
            value.retain_grad()

            # TVM MM
            torch.cuda.synchronize()
            start = time.time()
            (q_img, k_img, v_img) = map(lambda t: t.view(B * H, N, M), (query, key, value))
            k_img, v_img = map(lambda t: rearrange(t, 'b (h w) c -> b c h w', h=nx), (k_img, v_img))
            # start use torch.nn.F
            k_img, v_img = map(lambda t: F.unfold(t, 2*W+1, padding=padding, dilation=D), (k_img, v_img))
            k_img, v_img = map(lambda t: rearrange(t, 'b (d j) i -> b i j d', j=nlocal), (k_img, v_img))
            # end use torch.nn.F
            # start use tensor.unfold
            # (k_img, v_img) = map(
            #     lambda t: F.pad(t, (padding,)*4), (k_img, v_img)
            # )
            # (k_img, v_img) = map(
            #     lambda t: t.unfold(2, 2*W+1, 1).unfold(3, 2*W+1, 1), (k_img, v_img) # bh * c * nx * ny * 2w1 * 2w1
            # )
            # k_img, v_img = map(
            #     lambda t: rearrange(t, 'b c h w x y -> b (h w) (x y) c'),
            #     (k_img, v_img))
            # end use tensor.unfold
            dots_image = einsum('b i d, b i j d -> b i j', q_img, k_img)
            mask_invalid_locations_offical(dots_image, nx, ny, W, D, autoregressive)
            attention_probs1 = torch.nn.functional.softmax(dots_image, dim=-1)
            context1 = einsum('b i j, b i j d -> b i d', attention_probs1, v_img).view(B, H, N, M)
            context1.sum().backward()
            torch.cuda.synchronize()
            end = time.time()
            time1 += end - start
            query_grad1 = 1.0*query.grad
            query.grad.zero_()
            key_grad1 = 1.0*key.grad
            key.grad.zero_()
            value_grad1 = 1.0*value.grad
            value.grad.zero_()
            torch.cuda.empty_cache()

            assert D == 1
            assert not autoregressive
            torch.cuda.synchronize()
            start = time.time()
            attention2 = naive2d_matmul_qk(query, key, nx, ny, W, D, float('-inf'))
            attention_probs2 = torch.nn.functional.softmax(attention2, dim=-1) # (bsz, num_heads, seq_len, seq_len)
            context2 = attention_probs2 @ value  # (bsz, num_heads, seq_len, head_dim)
            context2.sum().backward()
            torch.cuda.synchronize()
            end = time.time()
            time2 += end - start
            query_grad2 = 1.0*query.grad
            query.grad.zero_()
            key_grad2 = 1.0*key.grad
            key.grad.zero_()
            value_grad2 = 1.0*value.grad
            value.grad.zero_()
            torch.cuda.empty_cache()

            try:
                # assert torch.allclose(attention1, attention2.float(), atol=1e-4, rtol=1e-5)
                assert torch.allclose(context1, context2.float(), atol=1e-4, rtol=1e-5), "context1"
                assert torch.allclose(query_grad1, query_grad2.float(), atol=1e-4, rtol=1e-3), "query_grad1"
                assert torch.allclose(key_grad1, key_grad2.float(), atol=1e-4, rtol=1e-3), "key_grad1"
                assert torch.allclose(value_grad1, value_grad2.float(), atol=1e-4, rtol=1e-3), "value_grad1"
            except AssertionError:
                failed_tests += 1

        print('Time unfold total: {0:.5f} s'.format(time1))
        print('Time pytorch naive implementation: {0:.5f} s'.format(time2))
        print('Unfold vs. Naive speedup: {0:.5f}x'.format(time1/time2))
        print(f'Failed tests: {failed_tests}/{i+1}')
        assert failed_tests == 0


if __name__ == '__main__':
    unittest.main()

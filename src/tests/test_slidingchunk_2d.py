import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import time
import unittest
import torch
import numpy as np
import random
from functools import lru_cache
from einops import rearrange
import torch.nn.functional as F
from models.layers.slidingchunk_2d import slidingchunk_2d, mask_invalid_locations, slidingchunk_2dautograd


@lru_cache()
def get_2dmask(nx: int, ny: int, w: int, device: str):
    return torch.BoolTensor([
        [
            abs((i // ny) // w - (j // ny) // w) > 1 or abs(
                (i % ny) // w - (j % ny) // w) > 1 for j in
            range(nx * ny)
        ]
        for i in range(nx * ny)
    ], device='cpu').to(device)


@lru_cache()
def get_2dmask_exact(nx, ny, w, device: str):
    return torch.BoolTensor([
        [
            abs(i // ny - j // ny) > w or abs(i % ny - j % ny) > w for j in
            range(nx * ny)
        ]
        for i in range(nx * ny)
    ], device='cpu').to(device)


def naive2d_matmul_qk(q, k, nx, ny, w, d, padding=0.0, exact=False):
    attn_weights = q @ k.transpose(-2, -1)
    # get mask
    if exact:
        mask = get_2dmask_exact(nx, ny, w, attn_weights.device)
    else:
        mask = get_2dmask(nx, ny, w, attn_weights.device)
    mask = mask[None, None, :, :]
    attn_weights.masked_fill_(mask, padding)
    return attn_weights


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
        nx = 40
        ny = 40
        N = nx * ny  # * 16
        M = 64  # hidden size
        W = 8  # one sided. Actual window size = (3*W)**2
        B = 2
        D = 1  # no dilation
        padding = W * D
        H = 12  # number of heads
        autoregressive = False  # not autoregressive
        device = 'cuda'
        dtype = torch.float32
        exact_sliding = 0

        failed_tests = 0
        time1 = time2 = 0
        for i in range(100):
            if i < 5:
                time1 = time2 = 0  # don't include the first few iterations because of high variance

            query = torch.randn(B * H * N * M, requires_grad=True,
                                device=device, dtype=dtype).view(B, H, N, M)
            query.retain_grad()
            key = torch.randn(B * H * N * M, requires_grad=True, device=device,
                              dtype=dtype).flip(dims=(0,)).view(B, H, N, M)
            key.retain_grad()
            value = torch.randn(B * H * N * M, requires_grad=True,
                                device=device, dtype=dtype).view(B, H, N, M)
            value.retain_grad()

            # TVM MM
            torch.cuda.synchronize()
            start = time.time()
            (q_img, k_img, v_img) = map(
                lambda t: rearrange(t, 'b h (x y) c -> (b h) c x y', x=nx),
                (query, key, value))
            # pad 0's to make sure that nx % W == 0, ny % W == 0
            (padx, pady) = map(lambda t: (W - t % W) % W, (nx, ny))
            (mx, my) = map(lambda t: (t[0]+t[1]) // W, ((nx, padx), (ny, pady)))
            if padx > 0 or pady > 0:
                (q_img, k_img, v_img) = map(
                    lambda t: F.pad(t, (0, pady, 0, padx)), (q_img, k_img, v_img)
                )
            (q_img, k_img, v_img) = map(
                lambda t: rearrange(t, 'b c (m x) (n y) -> b c m n (x y)', x=W, y=W), (q_img, k_img, v_img)
            )
            attention1 = slidingchunk_2d(q_img, k_img, False)
            # attention1 = slidingchunk_2dautograd(q_img, k_img, False)
            mask_invalid_locations(attention1, mx, my, padx, pady, W, exact=exact_sliding)
            attention_probs1 = torch.nn.functional.softmax(attention1, dim=-1)
            context1 = slidingchunk_2d(attention_probs1, v_img, True)
            # context1 = slidingchunk_2dautograd(attention_probs1, v_img, True)
            context1 = rearrange(context1, 'b c m n (x y) -> b (m x) (n y) c',
                                 x=W)
            context1 = context1[:, :nx, :ny].reshape(B, H, N, M)
            context1.sum().backward()
            torch.cuda.synchronize()
            end = time.time()
            time1 += end - start
            query_grad1 = 1.0 * query.grad
            query.grad.zero_()
            key_grad1 = 1.0 * key.grad
            key.grad.zero_()
            value_grad1 = 1.0 * value.grad
            value.grad.zero_()
            torch.cuda.empty_cache()

            # query = query.float()  # uncomment to profile the fp16 performance
            # query.retain_grad()
            # key = key.float()
            # key.retain_grad()
            # value = value.float()
            # value.retain_grad()
            assert D == 1
            assert not autoregressive
            torch.cuda.synchronize()
            start = time.time()
            attention2 = naive2d_matmul_qk(query, key, nx, ny, W, D,
                                           float('-inf'), exact=exact_sliding)
            attention_probs2 = torch.nn.functional.softmax(attention2, dim=-1)  # (bsz, num_heads, seq_len, seq_len)
            context2 = attention_probs2 @ value  # (bsz, num_heads, seq_len, head_dim)
            context2.sum().backward()
            torch.cuda.synchronize()
            end = time.time()
            time2 += end - start
            query_grad2 = 1.0 * query.grad
            query.grad.zero_()
            key_grad2 = 1.0 * key.grad
            key.grad.zero_()
            value_grad2 = 1.0 * value.grad
            value.grad.zero_()
            torch.cuda.empty_cache()

            # import pdb; pdb.set_trace()

            try:
                assert torch.allclose(context1.float(), context2.float(), atol=1e-4,
                                      rtol=1e-5), "context1"
                assert torch.allclose(query_grad1.float(), query_grad2.float(),
                                      atol=1e-4, rtol=1e-3), "query_grad1"
                assert torch.allclose(key_grad1.float(), key_grad2.float(), atol=1e-4,
                                      rtol=1e-3), "key_grad1"
                assert torch.allclose(value_grad1.float(), value_grad2.float(),
                                      atol=1e-4, rtol=1e-3), "value_grad1"
                # # uncomment to profile the fp16 performance
                # assert torch.allclose(context1.float(), context2.float(), atol=2e-2,
                #                       rtol=1e-1), "context1"
                # assert torch.allclose(query_grad1.float(), query_grad2.float(),
                #                       atol=5e-2, rtol=2e-1), "query_grad1"
                # assert torch.allclose(key_grad1.float(), key_grad2.float(), atol=5e-2,
                #                       rtol=2e-1), "key_grad1"
                # assert torch.allclose(value_grad1.float(), value_grad2.float(),
                #                       atol=2e-2, rtol=1e-1), "value_grad1"
            except AssertionError:
                failed_tests += 1

        print('Time SlidingChunk total: {0:.5f} s'.format(time1))
        print('Time pytorch naive implementation: {0:.5f} s'.format(time2))
        print('SlidingChunk vs. Naive speedup: {0:.5f}x'.format(time1 / time2))
        print(f'Failed tests: {failed_tests}/{i + 1}')
        assert failed_tests == 0


if __name__ == '__main__':
    unittest.main()

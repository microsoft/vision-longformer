import time
import unittest
import torch
import numpy as np
import random
from functools import lru_cache
from models.diagonaled_mm_2d import diagonaled_mm2d, mask_invalid_locations


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
        nx = 14
        ny = 14
        Nloc = nx * ny
        Nglo = 1
        N = Nloc + Nglo
        M = 64  # hidden size
        W = 13  # one sided. Actual window size = (2w+1)**2
        B = 2
        D = 1  # no dilation
        H = 6  # number of heads
        C = M * H
        autoregressive = False  # not autoregressive
        scale = M ** -0.5
        device = 'cuda'
        dtype = torch.float32

        failed_tests = 0
        time1 = time2 = 0
        for i in range(50):
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
            q = query[:, :, Nglo:].float().contiguous() * scale
            k = key.float()
            v = value.float()
            attn11 = diagonaled_mm2d(q, k[:, :, Nglo:].contiguous(), nx, ny,
                                     W, D, False, 0, autoregressive)
            mask_invalid_locations(attn11, nx, ny, W, D, autoregressive)
            attn10 = torch.bmm(q.view(B * H, Nloc, M), k[:, :, :Nglo].reshape(B * H, Nglo, M).transpose(-2, -1)).view(B, H, Nloc, Nglo)
            attn1 = torch.cat((attn10, attn11), dim=-1)
            attn1 = (attn1 - torch.max(attn1, dim=-1, keepdim=True)[0]).softmax(dim=-1)
            x1 = diagonaled_mm2d(attn1[:,:,:,Nglo:].float().contiguous(), v[:,:,Nglo:].contiguous(), nx, ny, W, D, True, 0, autoregressive)
            x1 = x1 + torch.bmm(attn1[:, :, :, :Nglo].view(B * H, Nloc, Nglo), v[:, :, :Nglo].reshape(B * H, Nglo, M)).view(B, H, Nloc, M)
            x1 = x1.transpose(1, 2).reshape(B, Nloc, C)

            q_global = query[:, :, :Nglo].float().contiguous() * scale
            k_global = k
            v_global = v
            attn0 = torch.bmm(q_global.view(B * H, Nglo, M), k_global.reshape(B * H, N, M).transpose(-2, -1))
            attn0 = (attn0 - torch.max(attn0, dim=-1, keepdim=True)[0]).softmax(dim=-1)
            x0 = torch.bmm(attn0, v_global.reshape(B * H, N, M)).view(B, H, Nglo, M).transpose(1, 2).reshape(B, Nglo, C)

            context1 = torch.cat((x0, x1), dim=1)
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
            attn = (query @ key.transpose(-2, -1)) * scale
            attn = (attn - torch.max(attn, dim=-1, keepdim=True)[0]).softmax(dim=-1)
            context2 = (attn @ value).transpose(1, 2).reshape(B, N, C)
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

        print('Time tvm: {0:.5f} s'.format(time1))
        print('Time pytorch naive implementation: {0:.5f} s'.format(time2))
        print('TVM vs. Naive speedup: {0:.5f}x'.format(time1/time2))
        print(f'Failed tests: {failed_tests}/{i+1}')
        assert failed_tests == 0


if __name__ == '__main__':
    unittest.main()

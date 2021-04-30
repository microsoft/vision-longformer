# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
# Written by Pengchuan Zhang, penzhan@microsoft.com
import argparse
import time
from functools import lru_cache
import torch
from torch import nn, einsum
import numpy as np
import random
from einops import rearrange
import torch.nn.functional as F


# tvm customized cuda kernel
def vil_mm2d(query, key, value, nx, ny, W, B, H, M, N, D=1):
    from models.diagonaled_mm_2d import diagonaled_mm2d, mask_invalid_locations
    attention = diagonaled_mm2d(query, key, nx, ny, W, D, False, 0,
                                 False)
    mask_invalid_locations(attention, nx, ny, W, D, False)
    attention_probs = torch.nn.functional.softmax(attention, dim=-1)
    context = diagonaled_mm2d(attention_probs, value, nx, ny, W, D, True, 0,
                               False)
    return context


# sliding chunk
def slidingchunk(query, key, value, nx, ny, W, B, H, M, N, exact_sliding, is_autograd):
    from models.layers.slidingchunk_2d import slidingchunk_2d, \
        mask_invalid_locations, slidingchunk_2dautograd
    (q_img, k_img, v_img) = map(
        lambda t: rearrange(t, 'b h (x y) c -> (b h) c x y', x=nx),
        (query, key, value))
    # pad 0's to make sure that nx % W == 0, ny % W == 0
    (padx, pady) = map(lambda t: (W - t % W) % W, (nx, ny))
    (mx, my) = map(lambda t: (t[0] + t[1]) // W, ((nx, padx), (ny, pady)))
    if padx > 0 or pady > 0:
        (q_img, k_img, v_img) = map(
            lambda t: F.pad(t, (0, pady, 0, padx)), (q_img, k_img, v_img)
        )
    # (q_img, k_img, v_img) = map(
    #     lambda t: t.unfold(2, W, W).unfold(3, W, W).reshape(
    #         B * H, M, mx, my, -1), (q_img, k_img, v_img)
    # )
    (q_img, k_img, v_img) = map(
        lambda t: rearrange(t, 'b c (m x) (n y) -> b c m n (x y)', x=W, y=W),
        (q_img, k_img, v_img)
    )
    if is_autograd:
        attention = slidingchunk_2dautograd(q_img, k_img, False)
    else:
        attention = slidingchunk_2d(q_img, k_img, False)
    mask_invalid_locations(attention, mx, my, padx, pady, W,
                              exact=exact_sliding)
    attention_probs = torch.nn.functional.softmax(attention, dim=-1)
    if is_autograd:
        context = slidingchunk_2dautograd(attention_probs, v_img, True)
    else:
        context = slidingchunk_2d(attention_probs, v_img, True)
    context = rearrange(context, 'b c m n (x y) -> b (m x) (n y) c', x=W)
    context = context[:, :nx, :ny].reshape(B, H, N, M)
    return context


# full and mask
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


def naive2d_matmul_qk(q, k, nx, ny, w, padding=0.0, exact=False):
    attn_weights = q @ k.transpose(-2, -1)
    # get mask
    if exact:
        mask = get_2dmask_exact(nx, ny, w, attn_weights.device)
    else:
        mask = get_2dmask(nx, ny, w, attn_weights.device)
    mask = mask[None, None, :, :]
    attn_weights.masked_fill_(mask, padding)
    return attn_weights


def full_and_mask(query, key, value, nx, ny, W, B, H, M, N, exact_sliding):
    attention = naive2d_matmul_qk(
        query, key, nx, ny, W, float('-inf'), exact=exact_sliding
    )
    attention_probs = torch.nn.functional.softmax(
        attention, dim=-1)  # (bsz, num_heads, seq_len, seq_len)
    context = attention_probs @ value  # (bsz, num_heads, seq_len, head_dim)
    return context


# unfold
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


def unfold_with_torch(query, key, value, nx, ny, W, B, H, M, N, D=1, useF=True):
    padding = W * D
    nlocal = (2 * W + 1) ** 2
    (q_img, k_img, v_img) = map(lambda t: t.view(B * H, N, M),
                                (query, key, value))
    k_img, v_img = map(lambda t: rearrange(t, 'b (h w) c -> b c h w', h=nx),
                       (k_img, v_img))
    if useF:
        k_img, v_img = map(
            lambda t: F.unfold(t, 2 * W + 1, padding=padding, dilation=D),
            (k_img, v_img))
        k_img, v_img = map(lambda t: rearrange(t, 'b (d j) i -> b i j d', j=nlocal),
                           (k_img, v_img))
    else:
        (k_img, v_img) = map(
            lambda t: F.pad(t, (padding,)*4), (k_img, v_img)
        )
        (k_img, v_img) = map(
            lambda t: t.unfold(2, 2*W+1, 1).unfold(3, 2*W+1, 1), (k_img, v_img) # bh * c * nx * ny * 2w1 * 2w1
        )
        k_img, v_img = map(
            lambda t: rearrange(t, 'b c h w x y -> b (h w) (x y) c'),
            (k_img, v_img))

    dots_image = einsum('b i d, b i j d -> b i j', q_img, k_img)
    mask_invalid_locations_offical(dots_image, nx, ny, W, D, False)
    attention_probs = torch.nn.functional.softmax(dots_image, dim=-1)
    context = einsum('b i j, b i j d -> b i d', attention_probs, v_img).view(
        B, H, N, M)
    return context


def benchmark_visionlongformer(args, img_size):
    method = args.method
    is_autograd = method not in ['cuda', 'scwbackward']
    exact_sliding = args.exact
    M = args.M  # hidden size
    W = args.W  # one sided. Actual window size = (3*W)**2
    B = args.B
    D = args.D  # no dilation
    H = args.H  # number of heads

    nx = img_size
    ny = img_size
    N = nx * ny  # * 16
    autoregressive = False  # not autoregressive
    device = 'cuda'
    dtype = torch.float32

    nexps = 100
    cost = np.zeros(nexps)
    memory = np.zeros(nexps)
    for i in range(nexps):
        query = torch.randn(B * H * N * M, requires_grad=True,
                            device=device, dtype=dtype).view(B, H, N, M)
        query.retain_grad()
        key = torch.randn(B * H * N * M, requires_grad=True, device=device,
                          dtype=dtype).flip(dims=(0,)).view(B, H, N, M)
        key.retain_grad()
        value = torch.randn(B * H * N * M, requires_grad=True,
                            device=device, dtype=dtype).view(B, H, N, M)
        value.retain_grad()

        # start forward
        torch.cuda.synchronize()
        start = time.time()
        if method in ['scwautograd', 'scwbackward']:
            context = slidingchunk(query, key, value, nx, ny, W, B, H, M, N, exact_sliding, is_autograd)
        elif method == 'full':
            context = full_and_mask(query, key, value, nx, ny, W, B, H, M, N, exact_sliding)
        elif method in ['unfoldtensor', 'unfoldF']:
            context = unfold_with_torch(query, key, value, nx, ny, W, B, H, M, N, D=D, useF=method=='unfoldF')
        elif method == 'cuda':
            context = vil_mm2d(query, key, value, nx, ny, W, B, H, M, N, D=D)
        else:
            raise ValueError("Unsupported method: {}".format(method))
        # start backward
        context.sum().backward()
        torch.cuda.synchronize()
        # end of forward-backward
        end = time.time()
        cost[i] = end - start
        memory[i] = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        torch.cuda.empty_cache()
    # output ms and MB. Ignore the first few exps due to their large variance
    return np.mean(cost[10:])*1000, np.mean(memory[10:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="benchmark different implementations of vil")
    parser.add_argument('--method', default='scwbackward', help="the method: full, unfoldtensor, unfoldF, cuda, scwautograd, scwbackward")
    parser.add_argument('--exact', default=0, type=int,
                        help="0: no padding, 1: exact, -1: cyclic padding")
    parser.add_argument('--W', default=8, type=int,
                        help="one-sided window size. Actual window size = (3*W)**2")
    parser.add_argument('--H', default=12, type=int,
                        help="number of heads")
    parser.add_argument('--M', default=64, type=int,
                        help="hidden size per head")
    parser.add_argument('--B', default=2, type=int,
                        help="batchsize")
    parser.add_argument('--D', default=1, type=int,
                        help="dilation")
    args = parser.parse_args()

    np.random.seed(300)
    random.seed(300)
    torch.manual_seed(300)
    torch.cuda.manual_seed(300)
    torch.cuda.manual_seed_all(300)
    torch.set_printoptions(sci_mode=False)

    img_sizes = [16, 24, 32, 40] + list(range(48, 300, 24))
    # img_sizes = list(range(24, 300, 24))
    instances = []
    for img_size in img_sizes:
        try:
            print("compute the image size: ", img_size)
            cost, memory = benchmark_visionlongformer(args, img_size)
            print("Mean cost, memory: ", cost, memory)
            instances.append([img_size, cost, memory])
        except:
            print("Image size, time, memory:")
            print(instances)
            break
    print("Image size, time, memory:")
    print(instances)

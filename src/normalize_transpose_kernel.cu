/*
takes float32 RGB from HWC to planar fp16 CHW format for network input
*/

#include "config.hpp"
#include "checkcuda.hpp"
//#include "preprocessor.hpp"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace frame_preprocessor
{
    __global__ void normalize_transpose_rgb_f32_to_fp16(const float *__restrict__ src, __half *__restrict__ dst, int W, int H)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x >= W || y >= H)
            return;

        const size_t pix = static_cast<size_t>(y) * W + x;
        const size_t plane = static_cast<size_t>(H) * W;
        const size_t si = pix * 3;


        #pragma unroll
        for(int c = 0; c < 3; ++c)
        {
            const float v = src[si + c] * dNormScale(c) - dNormBias(c);
            dst[c * plane + pix] = __float2half_rn(v);
        }
    }

    __global__ void resize_normalize_transpose_fused(const uint8_t* __restrict__ src, __half *__restrict__ dst, int in_w, int in_h, int out_w, int out_h)
    {
        const int ox = blockIdx.x * blockDim.x + threadIdx.x;
        const int oy = blockIdx.y * blockDim.y + threadIdx.y;

        if(ox >= out_w || oy >= out_h)
            return;

        const float scale_x = static_cast<float>(in_w) / static_cast<float>(out_w);
        const float scale_y = static_cast<float>(in_h) / static_cast<float>(out_h);
        const float sx = (static_cast<float>(ox) + 0.5f) * scale_x - 0.5f;
        const float sy = (static_cast<float>(oy) + 0.5f) * scale_y - 0.5f;
        const float fx0 = floorf(sx), fy0 = floorf(sy);
        const float wx = sx - fx0, wy = sy - fy0;

        int x0 = static_cast<int>(fx0), x1 = x0 + 1;
        int y0 = static_cast<int>(fy0), y1 = y0 + 1;
        x0 = min(max(x0, 0), in_w - 1);
        x1 = min(max(x1, 0), in_w - 1);

        y0 = min(max(y0, 0), in_h - 1);
        y1 = min(max(y1, 0), in_h - 1);

        const size_t row0 = static_cast<size_t>(y0) * in_w, row1 = static_cast<size_t>(y1) * in_w;
        const size_t plane = static_cast<size_t>(out_h) * out_w;
        const size_t o = static_cast<size_t>(oy) * out_w + ox;

        #pragma unroll
        for(int c = 0; c < 3; ++c)
        {
            const float p00 = __ldg(&src[(row0 + x0) * 3 + c]);
            const float p01 = __ldg(&src[(row0 + x1) * 3 + c]);
            const float p10 = __ldg(&src[(row1 + x0) * 3 + c]);
            const float p11 = __ldg(&src[(row1 + x1) * 3 + c]);
            const float top = p00 + (p01 - p00) * wx;
            const float bot = p10 + (p11 - p10) * wx;
            const float v   = (top + (bot - top) * wy) * dim_normScale(c) - dim_normBias(c);
            dst[c * plane + o] = __float2half_rn(v);  
        }
    }

    void launch_resize_normalize_transpose(const std::uint8_t *src, __half *dst, int in_w, int in_h, int out_w, int out_h, cudaStream_t stream)
    {
        const dim3 block(dim_tileX, dim_tileY);
        const dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

        resize_normalize_transpose_fused<<<grid, block, 0, stream>>>(src, dst, in_w, in_h, out_w, out_h);
        CHECK_CUDA(cudaGetLastError());
    }
}

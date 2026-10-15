#include "config.hpp"
#include "checkcuda.hpp"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace frame_preprocessor
{
    __global__ void launch_resize_normalize_transpose_fused(const uint8_t *__restrict__ src, __half *__restrict__ dst, int in_w, int in_h, int out_w, int out_h)
    {
        const int ox = blockIdx.x * blockDim.x + threadIdx.x;
        const int oy = blockIdx.y * blockDim.y + threadIdx.y;

        if(ox >= out_w || oy >= out_h)
            return;

        const float scale_x = static_cast<float>(in_w) / static_cast<float>(out_w);
        const float scale_y = static_cast<float>(in_h) / static_cast<float>(out_h);

        const float sx = (static_cast<float>(ox) + 0.5f) * scale_x - 0.5f;
        const float sy = (static_cast<float>(oy) + 0.5f) * scale_y - 0.5f;

        const float fx0 = floorf(sx);
        const float fy0 = floorf(sy);
        const float wx = sx - fx0;
        const float wy = sy - fy0;

        int x0 = static_cast<int>(fx0), x1 = x0 + 1;
        int y0 = static_cast<int>(fy0), y1 = y0 + 1;
        x0 = min(max(x0, 0), in_w - 1);
        y0 = min(max(y0, 0), in_h - 1);
        x1 = min(max(x1, 0), in_w - 1);
        y1 = min(max(y1, 0), in_h - 1);

        const size_t row0 = static_cast<size_t>(y0) * in_w;
        const size_t row1 = static_cast<size_t>(y1) * in_w;
        const size_t o    = (static_cast<size_t>(oy) * out_w + ox) * 3;

        #pragma unroll
        for(int c = 0; c < 3; ++c)
        {
            const float p00 = static_cast<float>(src[(row0 + x0) * 3 + c]);
            const float p01 = static_cast<float>(src[(row0 + x1) * 3 + c]);
            const float p10 = static_cast<float>(src[(row1 + x0) * 3 + c]);
            const float p11 = static_cast<float>(src[(row1 + x1) * 3 + c]);
            const float top = p00 + (p01 - p00) * wx;
            const float bot = p10 + (p11 - p10) * wx;
            dst[o + c]      = top + (bot - top) * wy;
        }
    }
}

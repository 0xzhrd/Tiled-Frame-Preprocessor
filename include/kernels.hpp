/*

*/

#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace frame_processor
{
    void launch_nv12_to_rgb(const std::uint8_t *nv12, std::uint8_t *rgb, int width, int height, cudaStream_t stream);
    void launch_resize_normalize_transpose(const std::uint8_t *src, __half *dst, int in_w, int in_h, int out_w, int out_h, cudaStream_t stream);
}

/*
 header file declaring fixed constants for the preprocessor, defines the programs max allowable input and output frame dimensions
*/

#pragma once

#include <cstddef>

namespace frame_preprocessor
{
    //max input dimension is 4K
    //constexpr === compile time constant to be used for checks
    constexpr int dim_maxInputWidth  = 3840;
    constexpr int dim_maxInputHeight = 2160;
    //output dimensions 
    constexpr int dim_outputWidth  = 640;
    constexpr int dim_outputHeight = 640;
    constexpr int dim_channels     = 3;
    //tile sizing
    constexpr int dim_tileX = 16;
    constexpr int dim_tileY = 16;

    constexpr std::size_t dim_fp16Bytes = 2;

    __host__ __device__ inline float dim_normScale(int c)
    {
        const float s[3] = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
        return s[c]; 
    }

    __host__ __device__ inline float dim_normBias(int c)
    {
        const float b[3] = {0.0f, 0.0f, 0.0f};
        return b[c];
    }
    //extra compile time checks
    static_assert(dim_maxInputWidth > 0 && (dim_maxInputWidth % 2) == 0, "width must be an even positive integer.");
    static_assert(dim_maxInputHeight > 0 && (dim_maxInputHeight % 2) == 0, "height must be an even positive integer.");
    static_assert(dim_outputWidth > 0 && dim_outputHeight > 0, "height must be an even positive integer.");
    static_assert(dim_channels == 3, "the preprocessor pipeline must be 3 channels");

    constexpr std::size_t nv12_size_bytes(int width, int height)
    {
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) + static_cast<std::size_t>(width) * static_cast<std::size_t>(height/2);
    }

    constexpr std::size_t rgb_f32_size_bytes(int width, int height)
    {
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(dim_channels) * sizeof(float);
    }

    constexpr std::size_t output_size_bytes()
    {
        return static_cast<std::size_t>(dim_outputWidth) * static_cast<size_t>(dim_outputHeight) * static_cast<std::size_t>(dim_channels) * dim_fp16Bytes;
    }
}

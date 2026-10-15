/*
    first kernel to execute in the preprocessor program, takes in the NV12 bytes and then converts them to RGB(YUV 4:2:0)
*/

#include "config.hpp"
#include "checkcuda.hpp"
#include <cstdint>
#include <cuda_runtime.h>


namespace frame_preprocessor
{
    __global__ void nv12_to_rgb_u8(const uint8_t *__restrict__ nv12, uint8_t *__restrict__ rgb, int width, int height)
    {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        //simple boundary check
        if(row >= height || col >= width)
            return;
    
        //ITU-R BT.601(limited/studio range) coefficients for converting between YUV and RGB
        const int CY = 1220542, CUB = 2116026, CUG = -409993, CVG = -852492, CVR = 1673527;
        //fixed point arithimetic, rather than using fp 
        const int SHIFT = 20, ROUND = 1 << (SHIFT - 1);

        //conversion from nv12 to yuv 
        const uint8_t *chroma_plane = nv12 + static_cast<size_t>(width) * height;
        const size_t chroma_offset = static_cast<size_t>(row >> 1) * width + (col >> 1) * 2;

        const int Y = static_cast<int>(nv12[static_cast<size_t>(row) * width + col]);
        const int U = static_cast<int>(chroma_plane[chroma_offset]) - 128;
        const int V = static_cast<int>(chroma_plane[chroma_offset + 1]) - 128;

        const int y = max(0, Y -16) * CY;
        const int r = (y + CVR * V + ROUND) >> SHIFT;
        const int g = (y + CVG * V + CUG * U + ROUND) >> SHIFT;
        const int b = (y + CUB * U + ROUND) >> SHIFT;

        const size_t rgb_offset = (static_cast<size_t>(row) * width + col) * 3;
        rgb[rgb_offset + 0] = static_cast<uint8_t>(min(max(r, 0), 255));
        rgb[rgb_offset + 1] = static_cast<uint8_t>(min(max(g, 0), 255));
        rgb[rgb_offset + 2] = static_cast<uint8_t>(min(max(b, 0), 255));     
    }

    void launch_nv12_to_rgb(const uint8_t *nv12, uint8_t *rgb, int width, int height, cudaStream_t stream)
    {
        const dim3 block(dim_tileX, dim_tileY);
        const dim3 grid((width + dim_tileX - 1) / dim_tileX, (height + dim_tileY - 1) / dim_tileY);
        nv12_to_rgb_u8<<<grid, block, 0, stream>>>(nv12, rgb, width, height);
        CHECK_CUDA(cudaGetLastError());
    }

}
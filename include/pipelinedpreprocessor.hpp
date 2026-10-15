/*
double buffered preprocessing pipeline, runs the two kernels from (kernels.hpp) and returns a ready tensor
*/

#pragma once

#include "kernels.hpp"
#include "checkcuda.hpp"
#include "preprocessor.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

namespace frame_preprocessor
{
    class PipelinedPreprocessor
    {
        public:
            PipelinedPreprocessor(int input_width, int input_height);
            
            void submit(const std::uint8_t *src, std::size_t bytes);
            void submit_staged();
            std::uint8_t *staging();
            const __half *process();
            void synchronize();

            cudaStream_t compute_stream() const 
            {
                return compute_stream_.get();
            }
            cudaStream_t copy_stream() const
            {
                return copy_stream_.get();
            }

        private:
            static int validated_dim(int value, int max_value, const char *name);

            static constexpr int kSlots = 2;
            const int input_width_;
            const int input_height_;
            std::size_t nv12_bytes_;

            DeviceBuffer d_rgb_;
            HostBuffer pinned_[kSlots];
            DeviceBuffer d_in_[kSlots];
            DeviceBuffer d_out_[kSlots];

            Stream copy_stream_;
            Stream compute_stream_;
            Event copy_done_[kSlots];
            Event compute_done_[kSlots];

            unsigned long long head_;
            unsigned long long tail_;
            bool primed_[kSlots];
    };
}

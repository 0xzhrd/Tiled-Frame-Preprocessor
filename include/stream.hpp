/*
RAII wrapper around cudaStream_t
*/

#pragma once

#include "checkcuda.hpp"
#include <cuda_runtime.h>

namespace frame_processor
{
    class Stream
    {
        public:
            Stream()
            {
                CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
            }

            ~Stream()
            {
                if(stream_ != nullptr)
                    cudaStreamDestroy(stream_);
            }

            Stream(const Stream &) = delete;
            Stream &operator = (const Stream &) = delete;

            Stream(Stream &&other) noexcept : stream_(other.stream_)
            {
                other.stream_ = nullptr;
            }
            Stream &operator = (Stream &&other) noexcept
            {
                if(this != &other)
                {
                    if(stream_ != nullptr)
                        cudaStreamDestroy(stream_);
                    
                    stream_ = other.stream_;
                    other.stream_ = nullptr;
                }
                return *this;
            }

            cudaStream_t get() const noexcept
            {
                return stream_;
            } 

            void sync() const
            {
                CHECK_CUDA(cudaStreamSynchronize(stream_));
            }

        private:
            cudaStream_t stream_ = nullptr;
    };
}

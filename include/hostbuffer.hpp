/*
RAII wrapper for pinned host memory enables asynchronous host to device copies
*/

#pragma once

#include "checkcuda.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace frame_preprocessor
{
    class HostBuffer
    {
        public:
            HostBuffer() noexcept : ptr_(nullptr), size_(0) {}

            explicit HostBuffer(std::size_t size_bytes) : ptr_(nullptr), size_(size_bytes)
            {
                if(size_bytes != 0)
                {
                    CHECK_CUDA(cudaMallocHost(&ptr_, size_bytes));
                }
            }

            ~HostBuffer()
            {
                if(ptr_ != nullptr)
                {
                    cudaFreeHost(ptr_);
                }
            }

            HostBuffer(const HostBuffer &)              = delete;
            HostBuffer &operator = (const HostBuffer &) = delete;

            HostBuffer(HostBuffer &&other) noexcept : ptr_(other.ptr_), size_(other.size_)
            {
                other.ptr_  = nullptr;
                other.size_ = 0;
            }

            HostBuffer &operator = (HostBuffer&& other) noexcept 
            {
                if(this != &other)
                {
                    if(ptr_ != nullptr)
                    {
                        cudaFreeHost(ptr_);
                    }

                    ptr_  = other.ptr_;
                    size_ = other.size_;
                    other.ptr_  = nullptr;
                    other.size_ = 0;
                }
                return *this;
            }
            
            void *get() noexcept
            {
                return ptr_;
            }
            const void *get() const noexcept
            {
                return ptr_;
            }
            std::size_t size() const noexcept
            {
                return size_;
            }

            template <typename T> T *as() noexcept
            {
                return static_cast<T *>(ptr_);
            }
        
        private:
            void *ptr_;
            std::size_t size_;
    };
}

/*
RAII warapper for allocating the device memory buffer, failure throws unwinding the stack
*/

#pragma once

#include "checkcuda.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace frame_preprocessor
{
    class DeviceBuffer
    {
        public:
            DeviceBuffer() noexcept: ptr_(nullptr), size_(0) {}

            explicit DeviceBuffer(std::size_t size_bytes) : ptr_(nullptr), size_(size_bytes)
            {
                if(size_bytes != 0)
                {
                    CHECK_CUDA(cudaMalloc(&ptr_, size_bytes));
                }
            }

            ~DeviceBuffer()
            {
                if(ptr_ != nullptr)
                {
                    cudaFree(ptr_);
                }
            }

            DeviceBuffer(const DeviceBuffer &) = delete;
            DeviceBuffer &operator = (const DeviceBuffer &) = delete;

            DeviceBuffer(DeviceBuffer &&other) noexcept : ptr_(other.ptr_), size_(other.size_)
            {
                other.ptr_ = nullptr;
                other.size_ = 0;
            }

            DeviceBuffer &operator = (DeviceBuffer &&other) noexcept
            {
                if(this != &other)
                {
                    if(ptr_ != nullptr)
                        cudaFree(ptr_);

                    ptr_ = other.ptr_;
                    size_ = other.size_;
                    other.ptr_ = nullptr;
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

            template <typename T>T* as() noexcept
            {
                return static_cast<T *>(ptr_);
            }

        private:
            void *ptr_;
            std::size_t size_;
    };
}
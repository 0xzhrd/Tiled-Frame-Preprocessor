/*
RAII wrapper around cudaEvent_t measure the time it takes for kernel execution 
*/

#pragma once

#include "checkcuda.hpp"
#include <cuda_runtime.h>

namespace frame_preprocessor
{
    class Event
    {
        public:
            Event()
            {
                CHECK_CUDA(cudaEventCreateWithFlags(&event_, cudaEventDefault));
            }

            ~Event()
            {
                if(event_ != nullptr)
                {
                    cudaEventDestroy(event_);
                }
            }

            Event(const Event&) = delete;
            Event &operator = (const Event &) = delete;

            Event(Event&& other) noexcept: event_(other.event_)
            {
                other.event_ = nullptr;
            }

            Event &operator = (Event&& other) noexcept
            {
                if(this != &other)
                {
                    if(event_ != nullptr)
                        cudaEventDestroy(event_);

                    event_ = other.event_;
                    other.event_ = nullptr;
                }
                return *this;
            }

            void record(cudaStream_t stream) const
            {
                CHECK_CUDA(cudaEventRecord(event_, stream));
            }

            void sync() const
            {
                CHECK_CUDA(cudaEventSynchronize(event_));
            }

            static float elapsed_ms(const Event &start, const Event &stop)
            {
                float ms = 0.0f;
                CHECK_CUDA(cudaEventElapsedTime(&ms, start.event_, stop.event_));
                return ms;
            }

            cudaEvent_t get() const noexcept
            {
                return event_;
            }

        private:
            cudaEvent_t event_ = nullptr;
    };
}
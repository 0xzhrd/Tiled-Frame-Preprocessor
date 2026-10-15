/*
RAII wrapper for single stream preprocessing pipeline nv12 - rgb - resize-tranpose - fp16 network tensor 
 */

#pragma once

#include "devicebuffer.hpp"
#include "hostbuffer.hpp"
#include "config.hpp"
#include "stream.hpp"
#include "events.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace frame_processor
{
    class Preprocessor
    {
        public:
            Preprocessor(int input_width, int input_height);

            Preprocessor(const Preprocessor &) = delete;
            Preprocessor &operator = (const Preprocessor &) = delete;
            Preprocessor(Preprocessor &&) = delete;
            Preprocessor &operator = (Preprocessor &&) = delete;

            ~Preprocessor() = default;

            const __half *process(const std::uint8_t *nv12_frame, std::size_t nv12_frame_bytes);

            const __half *process_prestaged();

            std::uint8_t* input_buffer() noexcept
            {
                return h_nv12_.as<std::uint8_t>();
            }

            const __half *output() const noexcept 
            {
                return static_cast<const __half *>(d_output_.get());
            }

            cudaStream_t stream() const noexcept
            {
                return stream_.get();
            }

            int input_width() const noexcept
            {
                return input_width_;
            }
            int input_height() const noexcept
            {
                return input_height_;
            }

        private:
            const int input_width_;
            const int input_height_;

            DeviceBuffer  d_nv12_;
            DeviceBuffer  d_rgb_;
            DeviceBuffer d_resized_;
            DeviceBuffer d_output_;

            HostBuffer h_nv12_;

            Stream stream_;
            Event start_event_;
            Event stop_event_;
    };
}

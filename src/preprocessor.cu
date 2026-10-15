#include "kernels.hpp"
#include "preprocessor.hpp"
#include <string>
#include <cstring>
#include <stdexcept>

namespace frame_preprocessor
{
    namespace
    {
        int validated_dim(int value, int max_value, const char *name)
        {
            if(value <= 0)
            {
                throw std::invalid_argument(std::string("Preprocessor: ") + name + " must be positive (got " + std::to_string(value) + ")");
            }
            if((value % 2) != 0)
            {
                throw std::invalid_argument(std::string("Preprocessor: ") + name + " must be even for NV12 4:2:0 (got " + std::to_string(value) + ")");
            }
            if(value > max_value)
            {
                throw std::invalid_argument(std::string("Preprocessor: ") + name + " " +
                    std::to_string(value) + " exceeds compile-time maximum " + std::to_string(max_value));
            }
            return value;
        }
    }

    Preprocessor::Preprocessor(int input_width, int input_height) : input_width_(validated_dim(input_width, dMaxInputWidth, "input width"))
                                                              , input_height_(validated_dim(input_height, dMaxInputHeight, "input height"))
                                                              , d_nv12_()
                                                              , d_rgb_()
                                                              , d_resized_()
                                                              , d_output_()
                                                              , h_nv12_()
                                                              , stream_()
                                                              , start_event_()
                                                              , stop_event_()
    {
        d_nv12_    = DeviceBuffer(nv12_size_bytes(input_width, input_height_));
        d_rgb_     = DeviceBuffer(rgb_f32_size_bytes(input_width_, input_height_));
        d_resized_ = DeviceBuffer(rgb_f32_size_bytes(dOutputWidth, dOutputHeight));
        d_output_  = DeviceBuffer(output_size_bytes());
        h_nv12_    = HostBuffer(nv12_size_bytes(input_width_, input_height_));
    }

    const __half *Preprocessor::process_prestaged()
    {
        CHECK_CUDA(cudaMemcpyAsync(d_nv12_.get(), h_nv12_.get(), nv12_size_bytes(input_width_, input_height_), cudaMemcpyHostToDevice, stream_.get()));

        launch_nv12_to_rgb(d_nv12_.as<std::uint8_t>(), d_rgb_.as<std::uint8_t>(), input_width_, input_height_, stream_.get());

        launch_resize_normalize_transpose(d_rgb_.as<std::uint8_t>(), d_output_.as<__half>(), input_width_, input_height_, dOutputWidth, dOutputHeight, stream_.get());

        return d_output_.as<__half>();
    }

    const __half *Preprocessor::process(const std::uint8_t *nv12_frame, std::size_t nv12_frame_bytes)
    {
        const std::size_t expected = nv12_size_bytes(input_width_, input_height_);
        if(nv12_frame_bytes != expected)
        {
            throw std::invalid_argument(
                "Preprocessor::process: frame is " +std::to_string(nv12_frame_bytes) + " bytes, expected " + std::to_string(expected) + " for " + 
                std::to_string(input_width_) + "x" + std::to_string(input_height_) + " NV12");
        }
        std::memcpy(h_nv12_.get(), nv12_frame, nv12_frame_bytes);
        return process_prestaged();
    }
}

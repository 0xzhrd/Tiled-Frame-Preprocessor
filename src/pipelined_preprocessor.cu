#include "pipelinedpreprocessor.hpp"

#include <cstring>
#include <string>
#include <stdexcept>

namespace frame_preprocessor
{
    int PipelinedPreprocessor::validated_dim(int value, int max_value, const char *name)
    {
        if(value <= 0)
            throw std::invalid_argument(std::string("PipelinedPreprocessor: ") + name + "must be positive (got " + std::to_string(value) + ")");
        
        if((value % 2) != 0)
            throw std::invalid_argument(std::string("PipelinedProcessor: ") + name + "must be even for NV12 4:2:0 (got " + std::to_string(value) + ")");

        if(value > max_value)
            throw std::invalid_argument(std::string("PipelinedPreprocessor: ") + name + " " + std::to_string(value) + "exceeds maximum " + std::to_string(max_value));        

        return value;
    }

    PipelinedPreprocessor::PipelinedPreprocessor(int input_width, int input_height) : input_width_ (validated_dim(input_width, dim_maxInputWidth, "input width")) 
                                                                                    , input_height_(validated_dim(input_height, dim_maxInputHeight, "input height"))
                                                                                    , nv12_bytes_(nv12_size_bytes(input_width_, input_height_)) 
                                                                                    , d_rgb_(), d_in_{}, d_out_{}
                                                                                    , pinned_{}, copy_stream_(), compute_stream_(), copy_done_{} 
                                                                                    , head_(0), tail_(0)
    {
        d_rgb_ = DeviceBuffer(rgb_f32_size_bytes(input_width_, input_height_));
        for(int s = 0; s < kSlots; ++s)
        { 
            d_in_[s]   = DeviceBuffer(nv12_bytes_);
            d_out_[s]  = DeviceBuffer(output_size_bytes());
            pinned_[s] = HostBuffer(nv12_bytes_);
            primed_[s] = false;
        }
    }
    
    std::uint8_t *PipelinedPreprocessor::staging()
    {
        return pinned_[head_ % kSlots].as<std::uint8_t>();
    }

    void PipelinedPreprocessor::submit_staged()
    {
        const int slot = static_cast<int>(head_ % kSlots);

        if(primed_[slot])
            CHECK_CUDA(cudaStreamWaitEvent(copy_stream_.get(), compute_done_[slot].get(), 0));

        CHECK_CUDA(cudaMemcpyAsync(d_in_[slot].get(), pinned_[slot].get(), nv12_bytes_, cudaMemcpyHostToDevice, copy_stream_.get()));
        
        CHECK_CUDA(cudaEventRecord(copy_done_[slot].get(), copy_stream_.get()));
        ++head_;
    }

    void PipelinedPreprocessor::submit(const std::uint8_t *src, std::size_t bytes)
    {
        if(bytes != nv12_bytes_)
            throw std::runtime_error("PipelinedPreprocessor::submit: frame size mismatch");
        
        std::memcpy(pinned_[head_ % kSlots].get(), src, bytes);
        submit_staged();
    }

    const __half *PipelinedPreprocessor::process()
    {
        const int slot = static_cast<int>(tail_ % kSlots);

        CHECK_CUDA(cudaStreamWaitEvent(compute_stream_.get(), copy_done_[slot].get(), 0));

        launch_nv12_to_rgb(d_in_[slot].as<std::uint8_t>(), d_rgb_.as<std::uint8_t>(), input_width_, input_height_, compute_stream_.get());

        launch_resize_normalize_transpose(d_rgb_.as<std::uint8_t>(), d_out_[slot].as<__half>(), input_width_, input_height_, dim_outputWidth, dim_outputHeight, compute_stream_.get());

        CHECK_CUDA(cudaEventRecord(compute_done_[slot].get(), compute_stream_.get()));
        primed_[slot] = true;
        ++tail_;
        return d_out_[slot].as<__half>();
    }

    void PipelinedPreprocessor::synchronize()
    {
        const int slot = static_cast<int>((tail_ - 1) % kSlots);
        CHECK_CUDA(cudaEventSynchronize(compute_done_[slot].get()));
    }                                                                          
}

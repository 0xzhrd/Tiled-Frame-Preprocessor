#include "checkcuda.hpp"
#include "yolodetector.hpp"

#include <cstdio>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <cuda_fp16.h>
#include <NvInferRuntime.h>



namespace frame_preprocessor
{
    namespace
    {
        __global__ void half_to_float_kernel(const __half *src, float *dst, std::size_t n)
        {
            const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if(i < n)
                dst[i] = __half2float(src[i]);
        }

        class Logger : public nvinfer1::ILogger
        {
            void log(Severity s, const char *msg) noexcept override
            {
                if(s <= Severity::kWARNING) 
                    std::fprintf(stderr, "[TRT] %s\n", msg);

            }
        };
        Logger g_logger;

        std::vector<char> read_file(const std::string &p)
        {
            std::ifstream f(p, std::ios::binary | std::ios::ate);
            if(!f)
                throw std::runtime_error("YoloDetector: cannot open engine " + p);
            
            const std::streamsize n = f.tellg(); f.seekg(0);
            std::vector<char> b(n);
            if(!f.read(b.data(), n))
                throw std::runtime_error("YoloDetector: read failed " + p);

            return b;
        }
    }    

    YoloDetector::YoloDetector(const std::string &engine_path)
    {
        const std::vector<char> blob = read_file(engine_path);

        runtime_.reset(nvinfer1::createInferRuntime(g_logger));
        if(!runtime_)
            throw std::runtime_error("YoloDetector: createInferRuntime failed");

        engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
        if(!engine_)
            throw std::runtime_error("YoloDetector: deserialize failed");
        
        context_.reset(engine_->createExecutionContext());
        if(!context_)
            throw std::runtime_error("YoloDetector: createExecutionContext failed");

        for(int i = 0; i < engine_->getNbIOTensors(); ++i)
        {
            const char *name = engine_->getIOTensorName(i);
            const auto dims = engine_->getTensorShape(name);

            if(engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            {
                input_name_ = name;
                input_dim_ = dims.d[dims.nbDims - 1];
                const auto dt = engine_->getTensorDataType(name);
                input_is_half_ = (dt == nvinfer1::DataType::kHALF);
                if(!input_is_half_)
                {
                    if(dt != nvinfer1::DataType::kFLOAT)
                        throw std::runtime_error("YoloDetector: engine input is neither FP16 nor FP32");

                    const std::size_t n = std::size_t(3) * input_dim_ * input_dim_;
                    CHECK_CUDA(cudaMalloc(&d_input_f32_, n * sizeof(float)));
                }
            }
            else
            {
                output_name_ = name;
                num_attrs_ = dims.d[1];
                num_anchors_ = dims.d[2];
                num_classes_ = num_attrs_ - 4;
                output_elems_ = size_t(num_attrs_) * num_anchors_;
            }
        }
        if(input_name_.empty() || output_name_.empty())
            throw std::runtime_error("YoloDetector: could not find I/O tensors");

        CHECK_CUDA(cudaMalloc(&d_output_, output_elems_ * sizeof(float)));
        h_output_.resize(output_elems_);
        context_->setTensorAddress(output_name_.c_str(), d_output_);
    }

    YoloDetector::~YoloDetector()
    {
        if(d_output_)
            cudaFree(d_output_);

        if(d_input_f32_)
            cudaFree(d_input_f32_);
    }

    void YoloDetector::infer(const __half *d_input, cudaStream_t stream)
    {
        const void *in_ptr;
        if(input_is_half_)
        {
            in_ptr = d_input;
        } 
        else
        {
            const std::size_t n = std::size_t(3) * input_dim_ * input_dim_;
            const int threads = 256;
            const int blocks = static_cast<int>((n + threads - 1) / threads);
            half_to_float_kernel<<<blocks, threads, 0, stream>>>(d_input, static_cast<float *>(d_input_f32_), n);
            CHECK_CUDA(cudaGetLastError());
            in_ptr = d_input_f32_;
        }

        context_->setTensorAddress(input_name_.c_str(), const_cast<void *>(in_ptr));
        if(!context_->enqueueV3(stream))
            throw std::runtime_error("YoloDetector: enqueueV3 failed");
    }

    void YoloDetector::download(cudaStream_t stream)
    {
        CHECK_CUDA(cudaMemcpyAsync(h_output_.data(), d_output_, output_elems_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }

    std::vector<Detection> YoloDetector::decode(int orig_w, int orig_h, float conf_thresh, float iou_thresh, int keep_class) const
    {
        const float *out = h_output_.data();
        const int A = num_anchors_;
        const float sx = float(orig_w) / input_dim_;
        const float sy = float(orig_h) / input_dim_;

        std::vector<Detection> cand; cand.reserve(256);

        for(int i = 0; i < A; ++i)
        {
            int bc = (keep_class >= 0) ? keep_class : 0;
            float bs = (keep_class >= 0) ? out[(4 + keep_class) * A + i] : 0.0f;
            if(keep_class < 0)
            {
                for(int c = 0; c < num_classes_; ++c)
                {
                    const float s = out[(4 + c) * A + i];
                    if(s > bs)
                    {
                        bs = s;
                        bc = c;
                    }
                }
            } 
            if(bs < conf_thresh)
                continue;

            const float cx = out[0 * A + i], cy = out[1 * A + i], w = out[2 * A + i], h = out[3 * A + i];
            cand.push_back({ (cx - 0.5f *  w) * sx, (cy - 0.5f * h) * sy, (cx + 0.5f * w) * sx, (cy + 0.5f * h) * sy, bs, bc});
        }

        std::sort(cand.begin(), cand.end(), [](const Detection &a, const Detection &b){return a.score > b.score;});

        auto iou = [](const Detection &a, const Detection &b)
        {
            const float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
            const float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
            const float in = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
            const float un = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - in;
            return un > 0.f ? in/un : 0.f;
        };

        std::vector<Detection> keep;
        std::vector<char> dead(cand.size(), 0);

        for(size_t i = 0; i < cand.size(); ++i)
        {
            if(dead[i])
                continue;

            keep.push_back(cand[i]);

            for(size_t j = i + 1; j < cand.size(); ++j)
            {
                if(!dead[j] && cand[i].cls == cand[j].cls && iou(cand[i], cand[j]) > iou_thresh)
                    dead[j] = 1;
            }
        }
        return keep;
    }
}

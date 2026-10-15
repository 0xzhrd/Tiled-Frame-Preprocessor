/*
loads serialized engine and runs network inference on device tensor 
*/

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace frame_preprocessor
{
    struct Detection
    {
        float x1, y1, x2, y2, score;
        int cls;
    };

    class YoloDetector
    {
        public: 
            explicit YoloDetector(const std::string &engine_path);
            ~YoloDetector();
            void infer(const __half *d_input, cudaStream_t stream);
            void download(cudaStream_t stream);
            std::vector<Detection> decode(int orig_w, int orig_h
                                        , float conf_thresh = 0.25f
                                        , float iou_thresh  = 0.45f
                                        , int keep_class    = 0) const;

        private:
            struct  Deleter{ template<class T> void operator()(T *p) const {delete p;} };
            template<class T> using TrtPtr = std::unique_ptr<T, Deleter>;

            TrtPtr<nvinfer1::IRuntime> runtime_;
            TrtPtr<nvinfer1::ICudaEngine> engine_;
            TrtPtr<nvinfer1::IExecutionContext> context_;

            bool input_is_half_ = true;
            void *d_input_f32_ = nullptr;

            std::string input_name_, output_name_;
            int input_dim_ = 640, num_attrs_ = 84, num_anchors_ = 8400, num_classes_ = 80;
            void *d_output_ = nullptr;
            mutable std::vector<float> h_output_;
            size_t output_elems_ = 0;
    };
}

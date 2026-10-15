#include "config.hpp"
#include "checkcuda.hpp"
#include "preprocessor.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>

using namespace frame_preprocessor;

static void make_nv12_gradient(std::vector<uint8_t> &nv12, int W, int H)
{
    nv12.resize(nv12_size_bytes(W, H));
    uint8_t *Y = nv12.data();
    uint8_t *UV = nv12.data() + static_cast<size_t>(W) * H;
    for(int r = 0; r < H; ++r)
    {
        for(int c = 0; c < W; ++c)
        {
            Y[static_cast<size_t>(r) * W + c] = static_cast<uint8_t>(16 + 219 * (r + c) / (H + W));
        }
    }

    for(int r = 0; r < H / 2; ++r)
    {
        for(int c = 0; c < W / 2; ++c)
        {
            UV[static_cast<size_t>(r) * W + c * 2 + 0] = static_cast<uint8_t>(16 + 224 * c / (W / 2));
            UV[static_cast<size_t>(r) * W + c * 2 + 1] = static_cast<uint8_t>(16 + 224 * r / (H / 2));
        }
    }
}

static void make_nv12_random(std::vector<uint8_t> &nv12, int W, int H, unsigned seed)
{
    nv12.resize(nv12_size_bytes(W, H));
    std::mt19937 rng(seed);
    for(auto &b : nv12) b = static_cast<uint8_t>(rng() & 0xFF);
}

static std::vector<__half> reference_fp16(const std::vector<uint8_t> &nv12, int W, int H)
{
    cv::Mat nv12_mat(H * 3 / 2, W, CV_8UC1, const_cast<uint8_t *>(nv12.data()));
    cv::Mat rgb8, rgbf, resizedf;
    cv::cvtColor(nv12_mat, rgb8, cv::COLOR_YUV2RGB_NV12);
    rgb8.convertTo(rgbf, CV_32F);
    cv::resize(rgbf, resizedf, cv::Size(dim_outputWidth, dim_outputHeight), 0, 0, cv::INTER_LINEAR);

    const int Wp = dim_outputWidth, Hp = dim_outputHeight;
    const size_t plane = static_cast<size_t>(Hp) * Wp;
    std::vector<__half> out(3 * plane);
    for(int y = 0; y < Hp; ++y)
    {
        for(int x = 0; x < Wp; ++x)
        {
            const cv::Vec3f px = resizedf.at<cv::Vec3f>(y, x);
            for(int c = 0; c < 3; ++c)
            {
                out[c * plane + static_cast<size_t>(y) * Wp + x] = __float2half_rn(px[c] * dim_normScale(c) - dim_normBias(c));
            }
        }
    }
    return out;
}

static int32_t fp16_ordered(uint16_t h)
{
    return (h & 0x8000) ? -static_cast<int32_t>(h & 0x7FFF) : static_cast<int32_t>(h);
}

static int ulp_distance(__half a, __half b)
{
    uint16_t ha, hb;
    std::memcpy(&ha, &a, 2);
    std::memcpy(&hb, &b, 2);
    return std::abs(fp16_ordered(ha) - fp16_ordered(hb));
}

static bool compare(const std::vector<__half> &got, const std::vector<__half> &ref, const char *label)
{
    int max_ulp = 0; size_t worst = 0, hist[3] = {0, 0, 0};
    for(size_t i = 0; i < ref.size(); ++i)
    {
        const int d = ulp_distance(got[i], ref[i]);
        if(d > max_ulp) {max_ulp = d; worst = i; }
        ++hist[d == 0 ? 0 : (d == 1 ? 1 : 2)];
    }
    const bool pass = (max_ulp<= 1);
    std::printf("[%s] %-9s max_ulp = %d  (0:%zu   1:%zu)  >=2:%zu)   worst@%zu\n", 
                pass ? "PASS" : "FAIL", label, max_ulp, hist[0], hist[1], hist[2], worst);
    return pass;
}

int main()
{
    const int W = 1920, H = 1080;
    Preprocessor pp(W, H);
    const size_t out_elems = 3 * static_cast<size_t>(dim_outputWidth) * dim_outputHeight;

    auto run = [&](std::vector<uint8_t> &nv12, const char *label) -> bool
    {
        const __half *d_out = pp.process(nv12.data(), nv12.size());
        CHECK_CUDA(cudaStreamSynchronize(pp.stream()));
        std::vector<__half> got(out_elems);
        CHECK_CUDA(cudaMemcpy(got.data(), d_out, out_elems * sizeof(__half), cudaMemcpyDeviceToHost));

        return compare(got, reference_fp16(nv12, W, H), label);
    };

    bool ok = true;
    std::vector<uint8_t> g, r;
    make_nv12_gradient(g, W, H);          ok &= run(g, "gradient");
    make_nv12_random(r, W, H, 0x12345678);   ok &= run(r, "random");
    std::printf("%s\n", ok ? "ALL_PASS" : "FAILURES PRESENT");
    return ok ? 0 : 1;
}
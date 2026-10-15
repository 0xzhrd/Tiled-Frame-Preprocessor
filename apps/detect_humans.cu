#include "config.hpp"
#include "checkcuda.hpp"
#include "preprocessor.hpp"
#include "yolodetector.hpp"

#include <cstdio>
#include <chrono>
#include <vector>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>

using namespace frame_preprocessor;
using clk = std::chrono::steady_clock;

static float pct(std::vector<float> v, double p)
{
    if(v.empty())
        return 0.f;

    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(p/100.0 * (v.size() - 1) + 0.5))];
}

static void report(const char *n, const std::vector<float> &v)
{
    if(v.empty())
    {
        std::printf("%-12s (none)\n", n);
        return;
    }
    double m = 0;
    for(float x : v)
        m += x;
        m /= v.size();

    std::printf("%-12s mean=%.3f p50=%.3f p90=%.3f p99=%.3f p99.9=%.3f max=%.3f\n", n, m, pct(v, 50), pct(v, 90), pct(v, 99), pct(v, 99.9), *std::max_element(v.begin(), v.end()));
}

int main(int argc, char **argv)
{
    if(argc < 5)
    {
        std::fprintf(stderr, "usage: %s <in.nv12> <W> <H> <yolo.engine> [conf] [iou]\n", argv[0]);
        return(1);
    }

    const char *nv12 = argv[1];
    const int W = std::atoi(argv[2]) 
            , H = std::atoi(argv[3]);

    const float conf = (argc > 5) ? std::atof(argv[5]):0.25f;
    const float iou = (argc > 6) ? std::atof(argv[6]):0.45f;

    Preprocessor pp(W, H);
    YoloDetector det(argv[4]);

    const size_t fb = nv12_size_bytes(W, H);
    std::ifstream in(nv12, std::ios::binary);
    if(!in)
    {
        std::fprintf(stderr, "Cannot open %s\n", nv12);
        return(1);
    } 
    std::vector<uint8_t> frame(fb);

    cudaEvent_t e0, e1, e2;
    CHECK_CUDA(cudaEventCreate(&e0)); 
    CHECK_CUDA(cudaEventCreate(&e1));
    CHECK_CUDA(cudaEventCreate(&e2));

    std::vector<float> t_pre, t_inf, t_post, t_tot;
    size_t n = 0, people = 0;

    while(in.read(reinterpret_cast<char *>(frame.data()), fb))
    {
        const auto c0 = clk::now();
        CHECK_CUDA(cudaEventRecord(e0, pp.stream()));
        const __half *d_in = pp.process(frame.data(), frame.size());
        CHECK_CUDA(cudaEventRecord(e1, pp.stream()));
        det.infer(d_in, pp.stream());
        CHECK_CUDA(cudaEventRecord(e2, pp.stream()));
        det.download(pp.stream());
        CHECK_CUDA(cudaStreamSynchronize(pp.stream()));

        const auto c1 = clk::now();
        auto dets = det.decode(W, H, conf, iou, 0);
        const auto c2 = clk::now();

        float pre = 0, inf = 0;
        CHECK_CUDA(cudaEventElapsedTime(&pre, e0, e1));
        CHECK_CUDA(cudaEventElapsedTime(&inf, e1, e2));
        t_pre.push_back(pre);
        t_inf.push_back(inf);
        t_post.push_back(std::chrono::duration<float, std::milli>(c2 - c1).count());
        t_tot.push_back(std::chrono::duration<float, std::milli>(c2 - c0).count());

        people += dets.size();
        if(n < 5 || n % 100 == 0)
        {
            std::printf("frame %5zu : %2zu people | pre %.2f inf %.2f post %.2f tot %.2f ms\n", n, dets.size(), pre, inf, t_post.back(), t_tot.back());
            if(n < 5) for(auto &d : dets)
                std::printf("person %.2f [%.0f %.0f %.0f %.0f]\n", d.score, d.x1, d.y1, d.x2, d.y2);
        }
        ++n;
    }

    std::printf("---- %zu frames, %zu person detections ----\n", n, people);
    report("preprocess", t_pre);
    report("inference", t_inf);
    report("postprocess", t_post);
    report("end_to_end", t_tot);

    if(!t_tot.empty())
        std::printf("throughput @p99.9 end-to-end: %.1f fps\n", 1000.0f / pct(t_tot, 99.9));

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);

    return(0);
}

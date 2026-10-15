#include "config.hpp"
#include "checkcuda.hpp"
#include "preprocessor.hpp"

#include <cmath>
#include <cstdio>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

using namespace frame_preprocessor;

static void fill_frame(std::vector<uint8_t> &nv12, int W, int H)
{
    nv12.resize(nv12_size_bytes(W, H));
    for(size_t i = 0; i < nv12.size(); ++i)
        nv12[i] = static_cast<uint8_t>((i * 131u + 17u) &0xFFu);
}

struct Stats
{
    double min, 
           mean,
           p50,
           p90,
           p99,
           p999,
           max, 
           sd;
};

static Stats summarize(std::vector<double> &x)
{
    std::sort(x.begin(), x.end());
    const size_t n = x.size();
    auto pct = [&](double p)
    {
        return x[std::min(n - 1, static_cast<size_t>(p * (n - 1) + 0.5))];
    };

    double s = 0.0;
    for(double v : x)
        s += v;
    const double mean = s / n;
    double var = 0.0;
    for(double v : x)
        var += (v - mean) * (v - mean);
    return{x.front(), mean, pct(0.50), pct(0.90), pct(0.99), pct(0.999), x.back(), std::sqrt(var / n)};
}

static void report(const char *label, std::vector<double> &x)
{
    const Stats s = summarize(x);
    std::printf("%-13s min=%.3f mean=%.3f p50=%.3f p90=%.3f p99=%.3f p99.9=%.3f max=%.3f sd=%.3f\n", 
                 label, s.min, s.mean, s.p50, s.p90, s.p99, s.p999, s.max, s.sd);
}


int main(int argc, char **argv)
{
    const int W = (argc > 1) ? std::atoi(argv[1]) : 3840;
    const int H = (argc > 2) ? std::atoi(argv[2]) : 2160;
    const int iters = (argc > 3) ? std::atoi(argv[3]) : 20000;
    const int warmup = 300;
    
    Preprocessor pp(W, H);
    
    std::vector<uint8_t> nv12;
    fill_frame(nv12, W, H);
    std::memcpy(pp.input_buffer(), nv12.data(), nv12.size());

    Event start, stop;
    for(int i = 0; i < warmup; ++i) 
        pp.process_prestaged();
    
    CHECK_CUDA(cudaStreamSynchronize(pp.stream()));

    std::vector<double> gpu_ms; gpu_ms.reserve(iters);
    for(int i = 0; i < iters; ++i)
    {
        start.record(pp.stream());
        pp.process_prestaged();
        stop.record(pp.stream());
        CHECK_CUDA(cudaStreamSynchronize(pp.stream()));
        gpu_ms.push_back(Event::elapsed_ms(start, stop));
    }

    std::vector<double> e2e_ms;
    e2e_ms.reserve(iters);
    for(int i = 0; i < iters; ++i)
    {
        const auto t0 = std::chrono::steady_clock::now();
        pp.process(nv12.data(), nv12.size());
        CHECK_CUDA(cudaStreamSynchronize(pp.stream()));
        const auto t1 = std::chrono::steady_clock::now();
        e2e_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::printf("Tiled Frame Preprocessor  %dx%d -> %dx%d iters=%d warmup=%d\n",
                W, H, dim_outputWidth, dim_outputHeight, iters, warmup);
    report("gpu_pipeline", gpu_ms);
    report("end_to_end", e2e_ms);
    std::printf("Throughput @p99.9 end-to-end: %.1f fps\n", 1000.0 / summarize(e2e_ms).p999);
    return(0);
}
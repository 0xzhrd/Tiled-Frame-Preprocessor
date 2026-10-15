#include "config.hpp"
#include "checkcuda.hpp"
#include "pipelinedpreprocessor.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace frame_preprocessor;
using steady = std::chrono::steady_clock;

static double ms_since(steady::time_point t0)
{
    return std::chrono::duration<double, std::milli>(steady::now() - t0).count();
}

static void synth_nv12(std::vector<std::uint8_t> &f, int W, int H)
{
    const size_t ysize = size_t(W) * H;
    for(int y = 0; y < H; ++y)
    {
        for(int x = 0; x < W; ++x)
            f[size_t(y) * W + x] = std::uint8_t((x + y) & 0xFF);
        
    }
    std::memset(f.data() + ysize, 128, f.size() - ysize);
}

static bool load_first_frame(const std::string &path, std::vector<std::uint8_t> &f)
{
    std::ifstream in(path, std::ios::binary);
    if(!in)
        return false;

    in.read(reinterpret_cast<char *>(f.data()), f.size());
    return bool(in);
}

int main(int argc, char **argv)
{
    if(argc < 3)
    {
        std::fprintf(stderr, "usage: %s <W> <H> [in.nv12] [iters] [warmup]\n", argv[0]);
        return(1);
    }

    const int W = std::atoi(argv[1]);
    const int H = std::atoi(argv[2]);

    const std::string in_path = (argc > 3) ? argv[3] : "";
    const int ITERS = (argc > 4) ? std::atoi(argv[4]) : 20000;
    const int WARMUP = (argc > 5) ? std::atoi(argv[5]) : 300;
    
    PipelinedPreprocessor pp(W, H);

    std::vector<std::uint8_t> frame(nv12_size_bytes(W, H));
    if(!in_path.empty() && load_first_frame(in_path, frame))
    {
        std::printf("frame source: %s\n", in_path.c_str());
    }
    else
    {
        synth_nv12(frame, W, H);
        std::printf("frame source: synthetic %dx%d NV12\n", W, H);
    }

    pp.submit(frame.data(), frame.size());
    for(int i = 0; i < WARMUP; ++i)
    {
        pp.submit(frame.data(), frame.size());
        pp.process();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    double serial_per = 0.0, overlapped_per = 0.0;

    {
        const auto t0 = steady::now();
        for(int i = 0; i < ITERS; ++i)
        {
            pp.submit_staged();
            pp.process();
            pp.synchronize();
        }
        serial_per = ms_since(t0) / ITERS;
    }

    for(int i = 0; i < 4; ++i)
    {
        pp.submit(frame.data(), frame.size());
        pp.process();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    {
        const auto t0 = steady::now();
        for(int i = 0; i < ITERS; ++i)
        {
            pp.submit_staged();
            pp.process();
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        overlapped_per = ms_since(t0) / ITERS;
    }

    std::printf("iters=%d  warmup=%d  %dx%d -> %dx%d\n", ITERS, WARMUP, W, H, dim_outputWidth, dim_outputHeight);
    std::printf("serial     : %.3f ms/frame     (%.1f fps)\n", serial_per, 1000.0 / serial_per);
    std::printf("overlapped : %.3f ms/frame     (%.1f fps)\n", overlapped_per, 1000.0 / overlapped_per);
    std::printf("speedup    : %.2fx\n", serial_per / overlapped_per);

    return(0);
}
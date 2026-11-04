#include "delta_core.h"
#include <unistd.h>
#include <cstdlib>

struct VectorStats {
    double ss=0,rr=0,min_cos=1,max_error=0; int count=0;
    void add(const float *actual,const float *ref,int size) {
        double aa=0,bb=0,ab=0;
        for(int i=0;i<size;i++) { double a=actual[i],b=ref[i],d=a-b;ss+=d*d;rr+=b*b;aa+=a*a;bb+=b*b;ab+=a*b;max_error=std::max(max_error,std::abs(d)); }
        if(aa>0&&bb>0)min_cos=std::min(min_cos,ab/std::sqrt(aa*bb)); count++;
    }
    void write(std::ostream &f) const { f<<"{\"nrmse\":"<<std::sqrt(ss/std::max(rr,1e-30))<<",\"min_cosine\":"<<min_cos<<",\"max_abs_error\":"<<max_error<<",\"samples\":"<<count<<"}"; }
};

static int8_t output_value(Engine &e,int index,int channel,int position) {
    const auto &a=e.out[index];const auto data=e.output(index);
    if(a.fmt==RKNN_TENSOR_NC1HWC2) {
        int c2=a.dims[4],w=a.w_stride?a.w_stride:a.dims[3];
        return data[((channel/c2)*w+position)*c2+channel%c2];
    }
    if(a.fmt==RKNN_TENSOR_NHWC) return data[position*a.dims[3]+channel];
    int w=a.w_stride?a.w_stride:a.dims[3];return data[channel*w+position];
}

struct DeltaRegister {
    Engine &e;
    std::vector<std::array<int8_t,256>> lut;
    std::vector<std::array<bool,256>> saturated;
    uint64_t commits=0,clipped=0,elements=0;
    explicit DeltaRegister(Engine &engine):e(engine),lut(DH),saturated(DH) {
        for(int h=0;h<DH;h++)for(int q=-128;q<128;q++) {
            auto &in=e.in[L+2+h];auto &out=e.out[L+h];
            float value=(q-out.zp)*out.scale;int next=std::lrintf(value/in.scale)+in.zp;
            lut[h][q+128]=static_cast<int8_t>(std::max(-128,std::min(127,next)));
            saturated[h][q+128]=next < -128 || next > 127;
        }
        reset();
    }
    void reset() { for(int h=0;h<DH;h++)std::memset(e.input(L+2+h),static_cast<unsigned char>(e.in[L+2+h].zp),DK*DK); }
    void update() {
        for(int h=0;h<DH;h++) {
            auto dst=e.input(L+2+h);const auto &a=e.out[L+h];const auto src=e.output(L+h);
            const auto &map=lut[h];const auto &sat=saturated[h];
            if(a.fmt==RKNN_TENSOR_NC1HWC2) {
                // Traverse native blocks directly: no division/modulo per element.
                const int c2=a.dims[4],c1=a.dims[1],stride=a.w_stride?a.w_stride:a.dims[3];
                for(int block=0;block<c1;block++)for(int v=0;v<DK;v++) {
                    const auto input=src+(block*stride+v)*c2;auto target=dst+v*DK+block*c2;
                    for(int j=0;j<c2;j++) {int q=int(input[j])+128;target[j]=map[q];clipped+=sat[q];}
                }
            } else {
                for(int v=0;v<DK;v++)for(int k=0;k<DK;k++) {
                    int q=int(output_value(e,L+h,k,v))+128;dst[v*DK+k]=map[q];clipped+=sat[q];
                }
            }
        }
        elements+=DE;
        commits++;
    }
    void read(float *dst) {
        for(int h=0;h<DH;h++)for(int k=0;k<DK;k++)for(int v=0;v<DK;v++) {
            auto &a=e.out[L+h];dst[h*DK*DK+k*DK+v]=(int(output_value(e,L+h,k,v))-a.zp)*a.scale;
        }
    }
};

int main(int argc,char **argv) {
    try {
        ::alarm(180);
        const int steps=argc>1?std::atoi(argv[1]):LOOPS;need(steps>0,"invalid benchmark steps");
        Memory before=memory();
        auto embedding=readvec<float>("data/embedding.f32");auto ids=readvec<int32_t>("data/sequence.i32");
        std::ifstream reference("data/reference.f32",std::ios::binary),state_reference("data/state-reference.f32",std::ios::binary);
        need(bool(reference),"missing logits reference");if(DH)need(bool(state_reference),"missing state reference");
        Memory loaded,active,cleanup;Stats logits_stats;VectorStats state_stats;
        double replay_ms=0,replay_api_ms=0,closed_ms=0,closed_api_ms=0,host_commit_ms=0;
        std::array<double,3> replay_rates{},closed_rates{};uint64_t checksum=0,clipped=0,elements=0;long major_faults=0;
        rknn_mem_size mem_size{};int mem_query_rc=-999;uint64_t input_cma=0,output_cma=0;double init_ms=0;
        {
            MemoryTable table;Engine e("models/tcn_engram.rknn",true);Run cache(e,&table,embedding);DeltaRegister delta(e);
            // Match a persistent service: release temporary FP32 lookup after INT8 tables are built.
            embedding.clear();embedding.shrink_to_fit();loaded=memory();init_ms=e.init_ms;
            mem_query_rc=rknn_query(e.ctx,RKNN_QUERY_MEM_SIZE,&mem_size,sizeof(mem_size));
            for(const auto &a:e.in)input_cma+=a.size_with_stride;for(const auto &a:e.out)output_cma+=a.size_with_stride;
            std::vector<float> actual(V),ref(V),state_actual(DE?DE:1),state_ref(DE?DE:1);
            std::ofstream logit_dump("results/logits.f32",std::ios::binary),state_dump("results/states.f32",std::ios::binary);
            cache.reset();delta.reset();
            for(int pos=0;pos<N;pos++) {
                cache.prepare(ids[pos]);e.run();cache.update(ids[pos]);delta.update();e.logits(actual.data());
                reference.read(reinterpret_cast<char*>(ref.data()),V*sizeof(float));need(bool(reference),"short logits reference");
                logits_stats.add(actual.data(),ref.data());logit_dump.write(reinterpret_cast<char*>(actual.data()),V*sizeof(float));
                if(DH) {
                    delta.read(state_actual.data());state_reference.read(reinterpret_cast<char*>(state_ref.data()),DE*sizeof(float));need(bool(state_reference),"short state reference");
                    state_stats.add(state_actual.data(),state_ref.data(),DE);state_dump.write(reinterpret_cast<char*>(state_actual.data()),DE*sizeof(float));
                }
            }
            logit_dump.close();state_dump.close();::sync();active=memory();
            { std::ofstream snapshot("results/init-memory.json");write_memory(snapshot,active); }
            rusage a{},b{};getrusage(RUSAGE_SELF,&a);
            for(int repeat=0;repeat<3;repeat++) {
                cache.reset();delta.reset();
                for(int j=0;j<64;j++){int token=ids[j%N];cache.prepare(token);e.run();cache.update(token);delta.update();}
                int start_commits=cache.commits;uint64_t delta_commits=delta.commits;double elapsed=0;
                for(int j=0;j<steps;j++) {
                    int token=ids[j%N];auto start=Clock::now();cache.prepare(token);replay_api_ms+=e.run();
                    auto commit=Clock::now();cache.update(token);delta.update();host_commit_ms+=ms(commit,Clock::now());checksum+=e.top1();elapsed+=ms(start,Clock::now());
                }
                need(cache.commits-start_commits==steps && delta.commits-delta_commits==unsigned(steps),"one commit per token");
                replay_ms+=elapsed;replay_rates[repeat]=1000*steps/elapsed;
                std::fprintf(stderr,"REPLAY_REPEAT %d %.3f steps/s\n",repeat,replay_rates[repeat]);
            }
            for(int repeat=0;repeat<3;repeat++) {
                cache.reset();delta.reset();for(int j=0;j<32;j++){int token=ids[j%N];cache.prepare(token);e.run();cache.update(token);delta.update();}
                int token=e.top1();double elapsed=0;
                for(int j=0;j<steps;j++) {
                    auto start=Clock::now();cache.prepare(token);closed_api_ms+=e.run();cache.update(token);delta.update();token=e.top1();checksum+=token;elapsed+=ms(start,Clock::now());
                }
                closed_ms+=elapsed;closed_rates[repeat]=1000*steps/elapsed;
                std::fprintf(stderr,"CLOSED_REPEAT %d %.3f steps/s\n",repeat,closed_rates[repeat]);
            }
            getrusage(RUSAGE_SELF,&b);major_faults=b.ru_majflt-a.ru_majflt;clipped=delta.clipped;elements=delta.elements;
        }
        cleanup=memory();std::ofstream f("results/metrics.json");
        bool logits_gate=std::sqrt(logits_stats.ss/logits_stats.rr)<.08 && logits_stats.min_cos>.995;
        f<<"{\"heads\":"<<DH<<",\"head_dimension\":"<<DK<<",\"raw_int8_state_bytes\":"<<DE<<",\"rknn_initializations\":1,\"runs_per_token\":1,\"benchmark_steps_per_repeat\":"<<steps<<",\"repeats\":3"
         <<",\"init_ms\":"<<init_ms<<",\"replay_steps_per_second\":"<<3000*steps/replay_ms<<",\"replay_api_ms_per_step\":"<<replay_api_ms/(3*steps)<<",\"host_cache_and_delta_commit_ms\":"<<host_commit_ms/(3*steps)
         <<",\"closed_loop_steps_per_second\":"<<3000*steps/closed_ms<<",\"closed_loop_api_ms_per_step\":"<<closed_api_ms/(3*steps)<<",\"major_faults\":"<<major_faults<<",\"checksum\":"<<checksum
         <<",\"state_requant_clipped\":"<<clipped<<",\"state_requant_elements\":"<<elements<<",\"io_input_bytes_with_stride\":"<<input_cma<<",\"io_output_bytes_with_stride\":"<<output_cma
         <<",\"rknn_mem_query_rc\":"<<mem_query_rc<<",\"rknn_weight_bytes\":"<<mem_size.total_weight_size<<",\"rknn_internal_bytes\":"<<mem_size.total_internal_size<<",\"rknn_total_dma_bytes\":"<<mem_size.total_dma_allocated_size
         <<",\"system_sram_bytes\":"<<mem_size.total_sram_size<<",\"free_sram_bytes\":"<<mem_size.free_sram_size<<",\"replay_repeat_rates\":[";
        for(int i=0;i<3;i++)f<<(i?",":"")<<replay_rates[i];f<<"],\"closed_loop_repeat_rates\":[";
        for(int i=0;i<3;i++)f<<(i?",":"")<<closed_rates[i];f<<"],\"logits_vs_fp32\":";logits_stats.write(f);
        f<<",\"state_vs_fp32\":";state_stats.write(f);f<<",\"logits_finite_gate_passed\":"<<(logits_gate?"true":"false")<<",\"before\":";write_memory(f,before);
        f<<",\"resident_after_fp32_release\":";write_memory(f,loaded);f<<",\"active\":";write_memory(f,active);f<<",\"after_cleanup\":";write_memory(f,cleanup);f<<"}\n";f.close();
        need(cleanup.dma==before.dma,"DMA not released");
        std::printf("DELTA_BENCH_COMPLETE heads=%d dim=%d replay=%.3f closed=%.3f DMA=%dKiB RSS=%dKiB logits_gate=%d\n",DH,DK,3000*steps/replay_ms,3000*steps/closed_ms,active.dma,active.rss,int(logits_gate));
        return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"DELTA_BENCH_ERROR %s\n",e.what());return 1;}
}

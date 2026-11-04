#include "rknn_api.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sys/resource.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "engram_config.h"
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a,Clock::time_point b) { return std::chrono::duration<double,std::milli>(b-a).count(); }
static void ck(int rc,const char *op) { if(rc) throw std::runtime_error(std::string(op)+": "+std::to_string(rc)); }
static void need(bool value,const char *what) { if(!value) throw std::runtime_error(what); }
template<class T> static std::vector<T> readvec(const std::string &path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate); need(bool(f),"cannot open input file");
    size_t bytes=f.tellg(); need(bytes%sizeof(T)==0,"bad file length");
    std::vector<T> v(bytes/sizeof(T)); f.seekg(0); f.read(reinterpret_cast<char*>(v.data()),bytes);
    need(bool(f),"short input read"); return v;
}
struct Memory { int dma=0,rss=0,hwm=0,available=0,cma_total=0,cma_allocated=0,cma_released=0,cma_free=0; };
static Memory memory() {
    Memory m;
    for(const char *p:{"/proc/rk_dma_heap/alloc_bitmap","/proc/self/status","/proc/meminfo"}) {
        std::ifstream f(p); std::string s;
        while(std::getline(f,s)) {
            std::sscanf(s.c_str()," Used: %d KiB",&m.dma);
            std::sscanf(s.c_str(),"VmRSS: %d kB",&m.rss);
            std::sscanf(s.c_str(),"VmHWM: %d kB",&m.hwm);
            std::sscanf(s.c_str(),"MemAvailable: %d kB",&m.available);
            std::sscanf(s.c_str(),"CmaTotal: %d kB",&m.cma_total);
            std::sscanf(s.c_str(),"CmaAllocated: %d kB",&m.cma_allocated);
            std::sscanf(s.c_str(),"CmaReleased: %d kB",&m.cma_released);
            std::sscanf(s.c_str(),"CmaFree: %d kB",&m.cma_free);
        }
    }
    return m;
}
static void write_memory(std::ostream &f,const Memory &m) {
    f<<"{\"dma_used_kib\":"<<m.dma<<",\"rss_kib\":"<<m.rss<<",\"hwm_kib\":"<<m.hwm<<",\"mem_available_kib\":"<<m.available<<",\"cma_total_kib\":"<<m.cma_total<<",\"cma_allocated_kib\":"<<m.cma_allocated<<",\"cma_released_kib\":"<<m.cma_released<<",\"cma_free_kib\":"<<m.cma_free<<"}";
}
static int8_t quant(float x,const rknn_tensor_attr &a) {
    int q=std::lrintf(x/a.scale)+a.zp; return static_cast<int8_t>(std::max(-128,std::min(127,q)));
}
struct Engine {
    rknn_context ctx=0;
    std::vector<rknn_tensor_attr> in,out;
    std::vector<rknn_tensor_mem*> im,om;
    std::vector<std::vector<int8_t>> logical;
    double init_ms=0;
    explicit Engine(const std::string &path,bool use_memory) {
        auto start=Clock::now();
        ck(rknn_init(&ctx,const_cast<char*>(path.c_str()),0,0,nullptr),"init");
        rknn_input_output_num n{}; ck(rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&n,sizeof(n)),"count");
        need(n.n_input==(use_memory?unsigned(L+2+DH):unsigned(L+1))&&n.n_output==unsigned(L+DH),"unexpected IO count");
        in.resize(n.n_input); out.resize(n.n_output);
        im.resize(n.n_input,nullptr); om.resize(n.n_output,nullptr); logical.resize(n.n_input);
        for(size_t i=0;i<in.size();i++) {
            auto &a=in[i]; a.index=i;
            ck(rknn_query(ctx,RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR,&a,sizeof(a)),"NHWC input attr");
            std::fprintf(stderr,"INPUT %zu %s fmt=%d type=%d n=%u stride=%u bytes=%u zp=%d scale=%.9g dims=",i,a.name,a.fmt,a.type,a.n_elems,a.w_stride,a.size_with_stride,a.zp,a.scale);
            for(unsigned d=0;d<a.n_dims;d++) std::fprintf(stderr,"%u,",a.dims[d]); std::fprintf(stderr,"\n");
            need(a.type==RKNN_TENSOR_INT8&&(a.fmt==RKNN_TENSOR_NHWC||a.fmt==RKNN_TENSOR_NC1HWC2),"input requires native INT8");
            need(a.n_elems==unsigned(i>=unsigned(L+2)?DK*DK:(i==0||i==unsigned(L+1)?C:2*C)),"input element count mismatch");
            if(a.fmt==RKNN_TENSOR_NC1HWC2) {
                need(a.n_dims==5&&a.dims[0]==1&&a.dims[2]==1&&a.dims[1]*a.dims[4]==unsigned(i>=unsigned(L+2)?DK:C),"unexpected blocked input shape");
                logical[i].assign(a.n_elems,static_cast<int8_t>(a.zp));
            }
            a.pass_through=1;
            im[i]=rknn_create_mem(ctx,a.size_with_stride); need(im[i]!=nullptr,"input CMA allocation failed");
            std::memset(im[i]->virt_addr,static_cast<unsigned char>(a.zp),a.size_with_stride);
            ck(rknn_set_io_mem(ctx,im[i],&a),"set input");
        }
        for(size_t i=0;i<out.size();i++) {
            auto &a=out[i]; a.index=i;
            ck(rknn_query(ctx,RKNN_QUERY_NATIVE_OUTPUT_ATTR,&a,sizeof(a)),"native output attr");
            std::fprintf(stderr,"OUTPUT %zu %s fmt=%d type=%d n=%u bytes=%u zp=%d scale=%.9g\n",i,a.name,a.fmt,a.type,a.n_elems,a.size_with_stride,a.zp,a.scale);
            need(a.type==RKNN_TENSOR_INT8,"output requires INT8");
            if(a.fmt==RKNN_TENSOR_NC1HWC2)
                need(a.n_dims==5&&a.dims[2]==1&&a.dims[3]==unsigned(i>=unsigned(L)?DK:1),"blocked output requires one spatial position");
            need(a.n_elems==unsigned(i>=unsigned(L)?DK*DK:(i==0?V:C)),"output element count mismatch");
            om[i]=rknn_create_mem(ctx,a.size_with_stride); need(om[i]!=nullptr,"output CMA allocation failed");
            ck(rknn_set_io_mem(ctx,om[i],&a),"set output");
        }
        init_ms=ms(start,Clock::now());
    }
    ~Engine() {
        if(ctx) { for(auto p:om) if(p) rknn_destroy_mem(ctx,p); for(auto p:im) if(p) rknn_destroy_mem(ctx,p); rknn_destroy(ctx); }
    }
    int8_t *input(int i) { return logical[i].empty()?static_cast<int8_t*>(im[i]->virt_addr):logical[i].data(); }
    int8_t *output(int i) { return static_cast<int8_t*>(om[i]->virt_addr); }
    double run(bool repack=true) {
        if(repack) for(size_t i=0;i<in.size();i++) if(!logical[i].empty()) {
            const auto &a=in[i]; unsigned c1=a.dims[1],w=a.dims[3],c2=a.dims[4],stride=a.w_stride?a.w_stride:w;
            auto dst=static_cast<int8_t*>(im[i]->virt_addr); const auto src=logical[i].data();
            need(c1*stride*c2<=a.size_with_stride,"blocked input buffer too small");
            for(unsigned b=0;b<c1;b++) for(unsigned t=0;t<w;t++)
                std::memcpy(dst+(b*stride+t)*c2,src+t*(c1*c2)+b*c2,c2);
        }
        auto t=Clock::now(); ck(rknn_run(ctx,nullptr),"run"); return ms(t,Clock::now());
    }
    int top1() { auto p=output(0); return std::max_element(p,p+V)-p; }
    void logits(float *dst) { auto p=output(0); auto &a=out[0]; for(int j=0;j<V;j++) dst[j]=(int(p[j])-a.zp)*a.scale; }
};
struct Stats {
    double ss=0,rr=0,max_abs=0,min_cos=1;
    int samples=0,matches=0;
    void add(const float *y,const float *ref) {
        double yy=0,ff=0,yf=0;
        for(int j=0;j<V;j++) { double e=y[j]-ref[j]; ss+=e*e; rr+=double(ref[j])*ref[j]; max_abs=std::max(max_abs,std::abs(e)); yy+=double(y[j])*y[j]; ff+=double(ref[j])*ref[j]; yf+=double(y[j])*ref[j]; }
        min_cos=std::min(min_cos,yf/std::sqrt(yy*ff));
        matches+=(std::max_element(y,y+V)-y)==(std::max_element(ref,ref+V)-ref); samples++;
    }
    void write(std::ostream &f) const {
        f<<"{\"nrmse\":"<<std::sqrt(ss/rr)<<",\"max_abs_error\":"<<max_abs<<",\"min_cosine\":"<<min_cos<<",\"top1_matches\":"<<matches<<",\"samples\":"<<samples<<"}";
    }
};
struct MemoryTable {
    std::vector<int8_t> data;
    std::vector<float> scale;
    std::array<int,HN> last_rows{};
    MemoryTable():data(readvec<int8_t>("data/memory.i8")),scale(readvec<float>("data/memory_scale.f32")) {
        int rows=0; for(int h=0;h<HN;h++) rows+=BUCKETS[h];
        need(data.size()==unsigned(rows*HD)&&scale.size()==unsigned(rows),"memory table dimensions mismatch");
    }
    void lookup(int token,int prev1,int prev2,const rknn_tensor_attr &a,int8_t *dst) {
        int offset=0;
        for(int h=0;h<HN;h++) {
            uint64_t mix=uint64_t(token)*HASH_MULT[h][0] ^ uint64_t(prev1)*HASH_MULT[h][1];
            if(ORDERS[h]==3) mix ^= uint64_t(prev2)*HASH_MULT[h][2];
            int row=int((mix&0x7fffffffU)%BUCKETS[h])+offset;
            last_rows[h]=row; offset+=BUCKETS[h];
            const int8_t *src=data.data()+row*HD;
            float factor=scale[row]/a.scale;
            for(int j=0;j<HD;j++) {
                int q=std::lrintf(float(src[j])*factor)+a.zp;
                dst[h*HD+j]=static_cast<int8_t>(std::max(-128,std::min(127,q)));
            }
        }
    }
};
struct Run {
    Engine &e; MemoryTable *table;
    std::vector<int8_t> eq,eqstate,window;
    std::array<std::vector<int8_t>,L> rings;
    std::array<std::array<int8_t,256>,L> lut;
    std::array<int,L> ptr{};
    int window_ptr=0;
    int prev1=0,prev2=0;
    bool zero_memory=false;
    double lookup_ms=0,last_lookup_ms=0;
    int commits=0;
    Run(Engine &engine,MemoryTable *memory_table,const std::vector<float> &embedding):e(engine),table(memory_table) {
        eq.resize(V*C); for(int i=0;i<V*C;i++) eq[i]=quant(embedding[i],e.in[0]);
        {
            eqstate.resize(V*C); for(int i=0;i<V*C;i++) eqstate[i]=quant(embedding[i],e.in[1]);
            for(int i=0;i<L;i++) {
                rings[i].assign(2*D[i]*C,static_cast<int8_t>(e.in[i+1].zp));
                if(i) for(int q=-128;q<128;q++) {
                    float value=(q-e.out[i].zp)*e.out[i].scale;
                    lut[i][q+128]=quant(value,e.in[i+1]);
                }
            }
        }
    }
    void prepare(int token) {
        need(token>=0&&token<V,"bad token id");
        auto vec=eq.data()+token*C;
        {
            std::memcpy(e.input(0),vec,C);
            for(int i=0;i<L;i++) {
                int p=ptr[i],d=D[i];
                std::memcpy(e.input(i+1),rings[i].data()+p*C,C);
                std::memcpy(e.input(i+1)+C,rings[i].data()+((p+d)%(2*d))*C,C);
            }
        }
        if(table) {
            auto start=Clock::now();
            if(zero_memory) std::memset(e.input(L+1),static_cast<unsigned char>(e.in[L+1].zp),C);
            else table->lookup(token,prev1,prev2,e.in[L+1],e.input(L+1));
            last_lookup_ms=ms(start,Clock::now()); lookup_ms+=last_lookup_ms;
        }
        prev2=prev1; prev1=token;
    }
    void update(int token) {
        for(int i=0;i<L;i++) {
            auto dst=rings[i].data()+ptr[i]*C;
            if(i==0) std::memcpy(dst,eqstate.data()+token*C,C);
            else { auto src=e.output(i); for(int c=0;c<C;c++) dst[c]=lut[i][int(src[c])+128]; }
            ptr[i]=(ptr[i]+1)%(2*D[i]);
        }
        commits++;
    }
    void reset() {
        prev1=prev2=0;
        for(int i=0;i<L;i++) { ptr[i]=0; std::fill(rings[i].begin(),rings[i].end(),static_cast<int8_t>(e.in[i+1].zp)); }
    }
};

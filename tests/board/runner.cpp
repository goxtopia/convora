#include "rknn_api.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <unistd.h>
static void ck(int r){if(r)throw std::runtime_error("RKNN rc="+std::to_string(r));}
struct Tensor {rknn_tensor_attr a{};rknn_tensor_mem *m=nullptr;int c,w;};
static size_t index(const Tensor&t,int c,int w){
 const auto&a=t.a;int stride=a.w_stride?a.w_stride:t.w;
 if(a.fmt==RKNN_TENSOR_NC1HWC2){int c2=a.dims[4];return ((c/c2)*stride+w)*c2+c%c2;}
 if(a.fmt==RKNN_TENSOR_NHWC)return w*a.dims[3]+c;
 return c*stride+w;
}
int main(int argc,char**argv){
 rknn_context ctx=0;std::vector<Tensor>in,out;
 try{
  alarm(180);int ni,no,rows;std::ifstream shapes("shapes.txt");shapes>>ni>>no>>rows;
  ck(rknn_init(&ctx,(void*)"model.rknn",0,0,nullptr));rknn_input_output_num count{};ck(rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&count,sizeof(count)));
  if(count.n_input!=(unsigned)ni||count.n_output!=(unsigned)no)throw std::runtime_error("IO count mismatch");
  in.resize(ni);out.resize(no);
  for(int kind=0;kind<2;kind++){
   auto&ts=kind?out:in;
   for(int i=0;i<(int)ts.size();i++){
    auto&t=ts[i];shapes>>t.c>>t.w;t.a.index=i;
    ck(rknn_query(ctx,kind?RKNN_QUERY_NATIVE_OUTPUT_ATTR:RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR,&t.a,sizeof(t.a)));
    fprintf(stderr,"%s %d %s fmt=%d type=%d c=%d w=%d scale=%.10g zp=%d bytes=%u stride=%u dims=",kind?"OUT":"IN",i,t.a.name,t.a.fmt,t.a.type,t.c,t.w,t.a.scale,t.a.zp,t.a.size_with_stride,t.a.w_stride);
    for(unsigned d=0;d<t.a.n_dims;d++)fprintf(stderr,"%u,",t.a.dims[d]);fprintf(stderr,"\n");
    if(t.a.type!=RKNN_TENSOR_INT8)throw std::runtime_error("requires native int8");
    t.a.pass_through=1;t.m=rknn_create_mem(ctx,t.a.size_with_stride);if(!t.m)throw std::runtime_error("CMA allocation");
    memset(t.m->virt_addr,(unsigned char)t.a.zp,t.a.size_with_stride);ck(rknn_set_io_mem(ctx,t.m,&t.a));
    if(index(t,t.c-1,t.w-1)>=t.a.size_with_stride)throw std::runtime_error("layout overflow");
   }
  }
  std::ifstream feeds("inputs.f32",std::ios::binary);std::ofstream output("actual.f32",std::ios::binary);
  std::vector<float>v;
  for(int row=0;row<rows;row++){
   for(auto&t:in){
    v.resize(t.c*t.w);feeds.read((char*)v.data(),v.size()*4);if(!feeds)throw std::runtime_error("input EOF");
    for(int c=0;c<t.c;c++)for(int w=0;w<t.w;w++){
     int q=std::lrintf(v[c*t.w+w]/t.a.scale)+t.a.zp;
     ((int8_t*)t.m->virt_addr)[index(t,c,w)]=std::max(-128,std::min(127,q));
    }
   }
   ck(rknn_run(ctx,nullptr));
   for(auto&t:out){v.resize(t.c*t.w);for(int c=0;c<t.c;c++)for(int w=0;w<t.w;w++)v[c*t.w+w]=(((int8_t*)t.m->virt_addr)[index(t,c,w)]-t.a.zp)*t.a.scale;output.write((char*)v.data(),v.size()*4);}
  }
  for(auto&t:out)rknn_destroy_mem(ctx,t.m);for(auto&t:in)rknn_destroy_mem(ctx,t.m);rknn_destroy(ctx);ctx=0;
  printf("COMPLETE rows=%d\n",rows);return 0;
 }catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());if(ctx){for(auto&t:out)if(t.m)rknn_destroy_mem(ctx,t.m);for(auto&t:in)if(t.m)rknn_destroy_mem(ctx,t.m);rknn_destroy(ctx);}return 1;}
}

#define main previous_benchmark_main
#include "bench.cpp"
#undef main

int main() {
  try {
    ::alarm(180);
    auto embedding=readvec<float>("data/embedding.f32");
    auto ids=readvec<int32_t>("data/diag-sequence.i32");
    auto flags=readvec<int32_t>("data/diag-flags.i32");
    need(ids.size()==flags.size(),"flags length mismatch");
    MemoryTable table;Engine e("models/tcn_engram.rknn",true);Run cache(e,&table,embedding);DeltaRegister delta(e);
    embedding.clear();embedding.shrink_to_fit();
    std::ofstream inputs("results/input-feeds.f32",std::ios::binary),outputs("results/output-feeds.f32",std::ios::binary);
    std::ofstream positions("results/feed-positions.i32",std::ios::binary);
    std::vector<float> buf(16384);int samples=0;
    for(int t=0;t<int(ids.size());t++) {
      if(flags[t]&1){cache.reset();delta.reset();}
      cache.prepare(ids[t]);
      if(flags[t]&2) {
        for(int i=0;i<int(e.in.size());i++) {
          int c=i>=L+2?DK:C,w=i>=L+2?DK:(i>=1&&i<=L?2:1);
          auto src=e.input(i);auto &a=e.in[i];
          for(int ch=0;ch<c;ch++)for(int p=0;p<w;p++)buf[ch*w+p]=(int(src[p*c+ch])-a.zp)*a.scale;
          inputs.write(reinterpret_cast<const char*>(buf.data()),c*w*sizeof(float));
        }
      }
      e.run();
      if(flags[t]&2) {
        for(int i=0;i<int(e.out.size());i++) {
          int c=i>=L?DK:(i==0?V:C),w=i>=L?DK:1;auto &a=e.out[i];
          for(int ch=0;ch<c;ch++)for(int p=0;p<w;p++)buf[ch*w+p]=(int(output_value(e,i,ch,p))-a.zp)*a.scale;
          outputs.write(reinterpret_cast<const char*>(buf.data()),c*w*sizeof(float));
        }
        positions.write(reinterpret_cast<const char*>(&t),sizeof(t));samples++;
      }
      cache.update(ids[t]);delta.update();
    }
    std::printf("DUMP_COMPLETE tokens=%zu samples=%d\n",ids.size(),samples);
  } catch(const std::exception &error){std::fprintf(stderr,"DUMP_ERROR %s\n",error.what());return 1;}
  return 0;
}

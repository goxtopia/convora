#pragma once
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

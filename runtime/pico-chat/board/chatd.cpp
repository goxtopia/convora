#include "delta_core.h"
#include "delta_register.h"
static constexpr int CONTEXT=512;
#include <arpa/inet.h>
#include <cerrno>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <netinet/tcp.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

static volatile sig_atomic_t running=1;
static void stopped(int) { running=0; }
static const char *checkpoint="9db7a2fa06b2687bfaa8091b9814bbf6ce52c7aba64d45bd66af933a902adef5";
static const char *model_name="delta4x32-lccc-sft-qat-compatible-20261029";
static const char *rknn_sha="6942f512bbc74c4d1ddd309fbd8cc580d959520d772c56c6098e0e38d8291175";
static unsigned requests=0,resets=0;
static bool send_all(int fd,const std::string &s) {
    size_t offset=0;
    while(offset<s.size()) {
        ssize_t n=::send(fd,s.data()+offset,s.size()-offset,MSG_NOSIGNAL);
        if(n<0&&errno==EINTR&&running) continue;
        if(n<=0) return false;
        offset+=size_t(n);
    }
    return true;
}
static void json(int fd,int status,const std::string &body) {
    send_all(fd,"HTTP/1.1 "+std::to_string(status)+(status==200?" OK":" Error")+
        "\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "+
        std::to_string(body.size())+"\r\n\r\n"+body);
}
static bool event(int fd,const std::string &name,const std::string &data) {
    std::string body="event: "+name+"\ndata: "+data+"\n\n";
    std::ostringstream size;size<<std::hex<<body.size();
    return send_all(fd,size.str()+"\r\n"+body+"\r\n");
}
struct Request { std::string method,path,body; };
static Request receive(int fd) {
    std::string data;char buffer[2048];size_t end=std::string::npos;
    while((end=data.find("\r\n\r\n"))==std::string::npos) {
        ssize_t n=recv(fd,buffer,sizeof(buffer),0);need(n>0,"incomplete HTTP headers");
        data.append(buffer,size_t(n));need(data.size()<=16384,"HTTP headers too large");
    }
    std::istringstream head(data.substr(0,end));Request req;std::string version,line;
    head>>req.method>>req.path>>version;need(version=="HTTP/1.1"||version=="HTTP/1.0","bad HTTP version");
    std::getline(head,line);size_t length=0;
    while(std::getline(head,line)) {
        std::transform(line.begin(),line.end(),line.begin(),[](unsigned char c){return char(std::tolower(c));});
        if(line.rfind("content-length:",0)==0) {
            char *stop=nullptr;long value=strtol(line.c_str()+15,&stop,10);
            need(value>=0&&value<=8192,"HTTP body too large");length=size_t(value);
        }
        need(line.rfind("transfer-encoding:",0)!=0,"chunked request bodies are not supported");
    }
    req.body=data.substr(end+4);
    while(req.body.size()<length) {
        ssize_t n=recv(fd,buffer,std::min(sizeof(buffer),length-req.body.size()),0);
        need(n>0,"incomplete HTTP body");req.body.append(buffer,size_t(n));
    }
    req.body.resize(length);return req;
}
static size_t field(const std::string &s,const char *name) {
    size_t p=s.find(std::string("\"")+name+"\"");need(p!=std::string::npos,"missing input_ids");
    p=s.find(':',p);need(p!=std::string::npos,"invalid JSON field");return p+1;
}
static void ws(const std::string &s,size_t &p) { while(p<s.size()&&std::isspace((unsigned char)s[p]))p++; }
static long number(const std::string &s,size_t &p) {
    ws(s,p);need(p<s.size()&&(std::isdigit((unsigned char)s[p])||s[p]=='-'),"integer expected");
    char *stop=nullptr;errno=0;long n=strtol(s.c_str()+p,&stop,10);need(errno==0,"integer overflow");
    p=size_t(stop-s.c_str());return n;
}
static std::vector<int> input_ids(const std::string &body) {
    size_t p=field(body,"input_ids");ws(body,p);need(p<body.size()&&body[p++]=='[',"input_ids must be an array");
    std::vector<int> ids;
    for(;;) {
        ws(body,p);need(p<body.size(),"incomplete input_ids");
        if(body[p]==']'){p++;break;}
        long n=number(body,p);need(n>=0&&n<V,"token out of vocabulary");ids.push_back(int(n));
        need(ids.size()<=CONTEXT,"input exceeds model context");ws(body,p);need(p<body.size(),"incomplete input_ids");
        if(body[p]==']'){p++;break;}
        need(body[p++]==',',"invalid token array");ws(body,p);need(p<body.size()&&body[p]!=']',"trailing comma");
    }
    need(!ids.empty(),"empty input_ids");return ids;
}
struct Boost {
    std::string prior;
    Boost() {
        std::ifstream f("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");f>>prior;
        if(!prior.empty()) {std::ofstream o("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");o<<"performance";}
    }
    ~Boost() {if(!prior.empty()){std::ofstream o("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");o<<prior;}}
};
static bool connected(int fd) {
    char c;ssize_t n=recv(fd,&c,1,MSG_PEEK|MSG_DONTWAIT);
    return n!=0&&(n>0||errno==EAGAIN||errno==EWOULDBLOCK||errno==EINTR);
}
static void serve(int fd,Engine &engine,Run &state,DeltaRegister &delta) {
    Request req=receive(fd);
    if(req.method=="GET"&&req.path=="/health") {
        Memory m=memory();
        json(fd,200,"{\"ready\":true,\"model\":\""+std::string(model_name)+"\",\"rknn_sha256\":\""+std::string(rknn_sha)+"\",\"checkpoint_sha256\":\""+std::string(checkpoint)+
            "\",\"pid\":"+std::to_string(getpid())+",\"model_loads\":1,\"context_tokens\":"+std::to_string(CONTEXT)+
            ",\"requests\":"+std::to_string(requests)+",\"cache_resets\":"+std::to_string(resets)+
            ",\"rss_kib\":"+std::to_string(m.rss)+",\"dma_kib\":"+std::to_string(m.dma)+"}");return;
    }
    if(req.method=="POST"&&req.path=="/reset") {
        state.reset();delta.reset();resets++;json(fd,200,"{\"ok\":true,\"cache_resets\":"+std::to_string(resets)+"}");return;
    }
    if(req.method!="POST"||req.path!="/generate") {json(fd,404,"{\"error\":\"not found\"}");return;}
    auto ids=input_ids(req.body);int budget=80;
    if(req.body.find("\"max_new_tokens\"")!=std::string::npos) {size_t p=field(req.body,"max_new_tokens");long n=number(req.body,p);need(n>=1&&n<=160,"generation budget must be 1..160");budget=int(n);}
    state.reset();delta.reset();resets++;requests++;
    if(!send_all(fd,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"))return;
    if(!event(fd,"start","{\"input_tokens\":"+std::to_string(ids.size())+",\"model_loads\":1}"))return;
    Boost boost;auto start=Clock::now();int count=0,calls=0;bool eos=false,cancelled=false;
    double prefill_ms=0,decode_ms=0;
    for(int token:ids) {
        if(!running||!connected(fd)){cancelled=true;break;}
        state.prepare(token);engine.run();state.update(token);delta.update();
    }
    prefill_ms=ms(start,Clock::now());auto decode_start=Clock::now();
    if(!cancelled)for(int i=0;i<budget;i++) {
        if(!running||!connected(fd)){cancelled=true;break;}
        int token=engine.top1();count++;eos=token==2;
        if(!event(fd,"token","{\"id\":"+std::to_string(token)+",\"index\":"+std::to_string(i)+"}")){cancelled=true;break;}
        if(eos||i==budget-1)break;
        state.prepare(token);engine.run();state.update(token);delta.update();calls++;
    }
    decode_ms=ms(decode_start,Clock::now());
    std::ostringstream done;done<<"{\"output_tokens\":"<<count<<",\"decode_calls\":"<<calls<<",\"prefill_ms\":"<<prefill_ms
        <<",\"decode_ms\":"<<decode_ms<<",\"tokens_per_second\":"<<(decode_ms>0?1000*calls/decode_ms:0)
        <<",\"eos\":"<<(eos?"true":"false")<<",\"cancelled\":"<<(cancelled?"true":"false")<<"}";
    event(fd,"done",done.str());send_all(fd,"0\r\n\r\n");
}
int main(int argc,char **argv) {
    try {
        std::freopen("/userdata/pico-chat/service.log","a",stderr);
        const char *root=argc>1?argv[1]:"/mnt/sdcard/pico-models/delta-sft-qat-compatible-20261029";
        int port=argc>2?std::atoi(argv[2]):8781;need(port>1024&&port<65536,"bad port");need(chdir(root)==0,"model directory unavailable");
        struct sigaction action{};action.sa_handler=stopped;sigemptyset(&action.sa_mask);sigaction(SIGTERM,&action,nullptr);sigaction(SIGINT,&action,nullptr);signal(SIGPIPE,SIG_IGN);
        int listener=socket(AF_INET,SOCK_STREAM,0);need(listener>=0,"socket failed");int yes=1;
        setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
        sockaddr_in address{};address.sin_family=AF_INET;address.sin_port=htons(port);address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
        need(bind(listener,reinterpret_cast<sockaddr*>(&address),sizeof(address))==0,"board port occupied");need(listen(listener,16)==0,"listen failed");
        auto embedding=readvec<float>("data/embedding.f32");MemoryTable table;Engine engine("models/tcn_engram.rknn",true);Run state(engine,&table,embedding);DeltaRegister delta(engine);
        std::vector<float>().swap(embedding);
        std::fprintf(stderr,"PICO_CHAT_READY pid=%d port=%d rss_kib=%d\n",getpid(),port,memory().rss);std::fflush(stderr);
        while(running) {
            int fd=accept(listener,nullptr,nullptr);if(fd<0){if(errno==EINTR)continue;throw std::runtime_error("accept failed");}
            setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&yes,sizeof(yes));timeval timeout{5,0};
            setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
            try {serve(fd,engine,state,delta);}catch(const std::exception &e){std::fprintf(stderr,"REQUEST_ERROR %s\n",e.what());json(fd,400,"{\"error\":\"invalid or incomplete request\"}");}
            close(fd);
        }
        close(listener);return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"PICO_CHAT_ERROR %s\n",e.what());return 1;}
}

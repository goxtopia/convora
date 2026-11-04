"""Local chat UI and tokenizer bridge to the resident RV1103 service."""
import argparse
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import threading
import time
import select
import ipaddress
from collections import deque
from urllib.parse import urlsplit

os.environ['TOKENIZERS_PARALLELISM']='false'
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent
TOKENIZER_PATH=ROOT/'tokenizer.json'
TOKENIZER_SHA='71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668'
ASSISTANT='<|im_start|>assistant\n<think>\n\n</think>\n\n'
CONFIG=json.loads((ROOT/'model.json').read_text(encoding='utf-8'))
CONTEXT=CONFIG['context_tokens']
MODEL=CONFIG['model']
CHECKPOINT_SHA=CONFIG['checkpoint_sha256']


class PromptBuilder:
    def __init__(self,path=TOKENIZER_PATH):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=TOKENIZER_SHA:raise ValueError('Tokenizer SHA mismatch')
        self.tokenizer=Tokenizer.from_file(str(path));self.tokenizer.no_padding();self.tokenizer.no_truncation()

    def encode(self,text):return self.tokenizer.encode(text,add_special_tokens=False).ids

    @staticmethod
    def render(messages):
        pieces=[]
        for m in messages:
            header=ASSISTANT if m['role']=='assistant' else '<|im_start|>user\n'
            pieces.append(header+m['content']+'<|im_end|>\n')
        return ''.join(pieces)+ASSISTANT

    def build(self,messages):
        if not isinstance(messages,list) or not 1<=len(messages)<=200:raise ValueError('请发送1至200条对话消息。')
        for i,m in enumerate(messages):
            if not isinstance(m,dict) or m.get('role')!=('user' if i%2==0 else 'assistant'):raise ValueError('对话顺序应为用户、助手交替，并以用户问题结束。')
            if not isinstance(m.get('content'),str) or len(m['content'])>4000:raise ValueError('每条消息最多4000字。')
        if len(messages)%2!=1 or not messages[-1]['content'].strip():raise ValueError('请输入问题。')
        # Retain whole recent user/assistant pairs. Encode the final assembled
        # string in one call so boundary merges match the training encoder.
        kept=[messages[-1]]
        ids=self.encode(self.render(kept))
        if len(ids)>CONTEXT:
            content=self.encode(kept[0]['content'])
            allowance=CONTEXT-len(self.encode(self.render([dict(role='user',content='')])))
            content=content[-max(1,allowance):]
            while content:
                text=self.tokenizer.decode(content,skip_special_tokens=True).lstrip('\ufffd')
                ids=self.encode(self.render([dict(role='user',content=text)]))
                if len(ids)<=CONTEXT:break
                content=content[1:]
            return ids,dict(input_tokens=len(ids),retained_messages=1,dropped_messages=len(messages)-1,truncated=True)
        for end in range(len(messages)-1,1,-2):
            candidate=messages[end-2:end]+kept
            encoded=self.encode(self.render(candidate))
            if len(encoded)>CONTEXT:break
            kept=candidate;ids=encoded
        return ids,dict(input_tokens=len(ids),retained_messages=len(kept),dropped_messages=len(messages)-len(kept),truncated=len(kept)<len(messages))


class Board:
    def __init__(self,port):self.port=port
    def request(self,path,body=None,timeout=4):
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=timeout)
        try:
            data=None if body is None else json.dumps(body).encode()
            conn.request('GET' if body is None else 'POST',path,body=data,headers={'Content-Type':'application/json'})
            response=conn.getresponse()
            value=json.loads(response.read())
            if response.status!=200:raise ConnectionError(value.get('error','板端请求失败。'))
            return value
        finally:conn.close()
    def stream(self,ids,budget):
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=5)
        try:
            body=json.dumps(dict(input_ids=ids,max_new_tokens=budget)).encode()
            conn.request('POST','/generate',body,{'Content-Type':'application/json'})
            response=conn.getresponse()
            if response.status!=200:raise ConnectionError('板端拒绝了生成请求。')
            event='message';parts=[];finished=False
            while True:
                line=response.readline()
                if not line:break
                text=line.decode('utf-8').strip('\r\n')
                if text.startswith('event: '):event=text[7:]
                elif text.startswith('data: '):parts.append(text[6:])
                elif not text and parts:
                    value=json.loads('\n'.join(parts));parts=[]
                    if event=='done':finished=True
                    yield event,value
            if not finished:raise ConnectionError('板端流式连接中断。')
        finally:conn.close()


class Scheduler:
    """One NPU worker, bounded FIFO, no shared conversation state."""
    def __init__(self,capacity=12):
        self.lock=threading.Lock();self.waiting=deque();self.active=None;self.capacity=capacity
    def enqueue(self):
        with self.lock:
            if len(self.waiting)>=self.capacity:return None
            ticket=object();self.waiting.append(ticket);return ticket
    def start(self,ticket):
        with self.lock:
            if self.active is None and self.waiting and self.waiting[0] is ticket:
                self.active=self.waiting.popleft();return True
            return False
    def position(self,ticket):
        with self.lock:return self.waiting.index(ticket)+1
    def release(self,ticket):
        with self.lock:
            if self.active is ticket:self.active=None
            elif ticket in self.waiting:self.waiting.remove(ticket)
    def status(self):
        with self.lock:return dict(busy=self.active is not None,waiting=len(self.waiting),capacity=self.capacity)


class Server(ThreadingHTTPServer):
    daemon_threads=True
    allow_reuse_address=os.name!='nt'
    def server_bind(self):
        if os.name=='nt':self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()
    def __init__(self,address,board_port):
        self.builder=PromptBuilder();self.board=Board(board_port);self.scheduler=Scheduler()
        self.health_lock=threading.Lock();self.cached_health=None;self.health_at=0.
        super().__init__(address,Handler)
    def health(self):
        queue=self.scheduler.status()
        with self.health_lock:
            # The board daemon serves one request at a time. Polling it during
            # generation would queue health calls behind the current answer.
            if not queue['busy'] and time.monotonic()-self.health_at>5:
                try:
                    value=self.board.request('/health',timeout=2)
                    if (value.get('checkpoint_sha256')!=CHECKPOINT_SHA or value.get('context_tokens')!=CONTEXT
                            or value.get('model')!=MODEL or value.get('rknn_sha256')!=CONFIG['rknn_sha256']):raise ValueError('Configured model mismatch')
                    self.cached_health=dict(online=True,board=value)
                except (OSError,ValueError,http.client.HTTPException):
                    self.cached_health=dict(online=False,error='板子暂未连接，或当前模型尚未就绪。')
                self.health_at=time.monotonic()
            return dict(self.cached_health or dict(online=False),queue=queue,model=CONFIG,web_pid=os.getpid(),listen_host=self.server_address[0])


class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def setup(self):
        super().setup();self.connection.settimeout(10)
    def log_message(self,format,*args):
        # Keep request metadata only; conversation text is never logged.
        print('%s %s'%(self.log_date_time_string(),format%args),flush=True)
    def reply(self,status,value):
        data=json.dumps(value,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def origin_ok(self):
        origin=self.headers.get('Origin')
        if not origin:return self.headers.get('Sec-Fetch-Site') not in ('cross-site',)
        parsed=urlsplit(origin)
        return parsed.scheme=='http' and parsed.netloc.lower()==self.headers.get('Host','').lower()
    def connected(self):
        ready,_,_=select.select([self.connection],[],[],0)
        return not ready or bool(self.connection.recv(1,socket.MSG_PEEK))
    def lan_client(self):
        address=ipaddress.ip_address(self.client_address[0])
        return address.is_private or address.is_loopback or address.is_link_local
    def do_GET(self):
        if not self.lan_client():self.reply(403,dict(error='仅供局域网体验。'));return
        path=urlsplit(self.path).path
        if path=='/api/health':
            value=self.server.health();self.reply(200 if value['online'] else 503,value)
            return
        names={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
        if path not in names:self.reply(404,dict(error='not found'));return
        file=ROOT/'dist'/names[path]
        types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8'}
        data=file.read_bytes();self.send_response(200);self.send_header('Content-Type',types[file.suffix]);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-cache');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(data)
    def do_POST(self):
        if not self.lan_client():self.close_connection=True;self.reply(403,dict(error='仅供局域网体验。'));return
        if not self.origin_ok():self.close_connection=True;self.reply(403,dict(error='请从当前Pico Chat页面发起请求。'));return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=1024*1024:raise ValueError('请求大小超出限制。')
            body=json.loads(self.rfile.read(length).decode('utf-8'))
            if not isinstance(body,dict):raise ValueError('无效请求。')
        except (ValueError,UnicodeError):self.reply(400,dict(error='无效或过长的请求。'));return
        if self.path=='/api/reset':
            # Clearing one browser must never reset somebody else's generation.
            # Every /generate already resets all board caches before prefill.
            self.reply(200,dict(ok=True,scope='this_browser'))
            return
        if self.path!='/api/chat':self.reply(404,dict(error='not found'));return
        try:
            budget=body.get('max_new_tokens',80)
            if type(budget)!=int or not 1<=budget<=160:raise ValueError('回答长度应在1至160 token之间。')
            ids,meta=self.server.builder.build(body.get('messages'))
        except ValueError as exc:self.reply(400,dict(error=str(exc)));return
        ticket=self.server.scheduler.enqueue()
        if ticket is None:self.reply(429,dict(error='排队人数已满，请稍后再试。'));return
        started=time.perf_counter();generated=[];first_ms=None;last_text=''
        def emit(event,value):
            encoded=('event: '+event+'\ndata: '+json.dumps(value,ensure_ascii=False)+'\n\n').encode()
            self.wfile.write(encoded);self.wfile.flush()
        stream=None
        try:
            self.send_response(200);self.send_header('Content-Type','text/event-stream; charset=utf-8');self.send_header('Cache-Control','no-cache')
            self.send_header('Connection','close');self.send_header('X-Accel-Buffering','no');self.end_headers();self.close_connection=True
            self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            emit('meta',dict(**meta,model=MODEL,context_tokens=CONTEXT))
            while not self.server.scheduler.start(ticket):
                if not self.connected():return
                if time.perf_counter()-started>90:
                    emit('error',dict(error='排队等待超时，请重试。'));return
                emit('queue',dict(position=self.server.scheduler.position(ticket)))
                time.sleep(.5)
            if not self.connected():return
            queue_ms=(time.perf_counter()-started)*1000
            emit('start',dict(queue_ms=queue_ms))
            stream=self.server.board.stream(ids,budget)
            for event,value in stream:
                if not self.connected():break
                if event=='token':
                    token=value['id'];generated.append(token)
                    text=self.server.builder.tokenizer.decode(generated,skip_special_tokens=True).rstrip('\ufffd')
                    if text!=last_text:
                        if first_ms is None:first_ms=(time.perf_counter()-started)*1000
                        last_text=text;emit('delta',dict(text=text,tokens=len(generated)))
                elif event=='done':
                    final_text=self.server.builder.tokenizer.decode(generated,skip_special_tokens=True)
                    emit('done',dict(**value,text=final_text,first_token_ms=first_ms,queue_ms=queue_ms,total_ms=(time.perf_counter()-started)*1000))
        except (BrokenPipeError,ConnectionResetError):pass
        except (OSError,ValueError,http.client.HTTPException) as exc:
            try:emit('error',dict(error='板端连接中断，请检查USB后重试。'))
            except (OSError,ValueError):pass
        finally:
            if stream is not None:stream.close()
            self.server.scheduler.release(ticket)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=9110);parser.add_argument('--board-port',type=int,default=18765);parser.add_argument('--host',default='0.0.0.0')
    args=parser.parse_args();server=Server((args.host,args.port),args.board_port)
    print(f'PICO_CHAT_WEB_READY http://127.0.0.1:{args.port} pid={os.getpid()}',flush=True)
    try:server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()

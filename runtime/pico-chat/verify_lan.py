"""Actual board + independent HTTP clients; run against the deployed service."""
import array,concurrent.futures,http.client,json,sys,threading,time
from pathlib import Path
from server import Board,PromptBuilder,Scheduler,CONFIG,ROOT

import os
RELEASE=ROOT.parents[1]
OUT=RELEASE/'outputs/lan-verification'
LAN=os.environ.get('PICO_CHAT_HOST','127.0.0.1');PORT=int(os.environ.get('PICO_CHAT_PORT','9110'))
def request(path,body=None,origin=None,host=LAN):
    c=http.client.HTTPConnection(host,PORT,timeout=10)
    headers={'Content-Type':'application/json'}
    if origin:headers['Origin']=origin
    c.request('GET' if body is None else 'POST',path,None if body is None else json.dumps(body).encode(),headers)
    r=c.getresponse();status=r.status;d=json.loads(r.read());c.close();return status,d

def events(messages,budget=32,on_event=None,abort_kind=None):
    c=http.client.HTTPConnection(LAN,PORT,timeout=15)
    c.request('POST','/api/chat',json.dumps(dict(messages=messages,max_new_tokens=budget)).encode(),{'Content-Type':'application/json','Origin':f'http://{LAN}:{PORT}'})
    r=c.getresponse();assert r.status==200,(r.status,r.read())
    data=[];kind='';start=time.monotonic()
    try:
        while True:
            line=r.readline()
            if not line:break
            line=line.decode().strip()
            if line.startswith('event: '):kind=line[7:]
            elif line.startswith('data: '):
                value=json.loads(line[6:]);data.append((kind,value,time.monotonic()-start))
                if on_event:on_event(kind,value)
                if kind==abort_kind:break
    finally:r.close();c.close()
    return data

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks={};s=Scheduler(2);a=s.enqueue();b=s.enqueue();assert s.enqueue() is None
    assert not s.start(b) and s.start(a);s.release(b);c=s.enqueue();assert s.position(c)==1
    s.release(a);assert s.start(c);s.release(c);assert s.status()['waiting']==0
    checks['scheduler_fifo_capacity_and_cancel']=True
    status,health=request('/api/health');assert status==200 and health['online'] and health['listen_host']=='0.0.0.0'
    checks['lan_health']=health
    assert request('/api/reset',{},origin=f'http://{LAN}:{PORT}')[0]==200
    assert request('/api/reset',{},origin='http://example.org')[0]==403
    assert request('/api/chat',{'messages':[]})[0]==400
    checks['lan_same_origin_and_cross_origin_rejection']=True
    board=Board(18765);before=board.request('/health');builder=PromptBuilder()
    golden=json.loads((ROOT/'template-golden.json').read_text(encoding='utf-8'))
    golden_ids,golden_meta=builder.build(golden['messages'])
    assert golden_ids==golden['prefix_ids'] and golden_meta['retained_messages']==3
    checks['multi_turn_matches_training_encoder']=True
    probe=RELEASE/'tests/fixtures'
    ids=array.array('i');ids.frombytes((probe/'sequence.i32').read_bytes())
    logits=array.array('f');logits.frombytes((probe/'npu-closed-logits.f32').read_bytes())
    if sys.byteorder!='little':ids.byteswap();logits.byteswap()
    compared=[]
    for n in (1,32,133,260):
        expected=max(range(6400),key=lambda i:logits[(n-1)*6400+i])
        result=list(board.stream(list(ids[:n]),1));actual=next(v['id'] for e,v in result if e=='token')
        assert actual==expected,(n,actual,expected);compared.append({'prefix_tokens':n,'first_token':actual})
    checks['resident_matches_independent_board_benchmark']=compared
    messages=[dict(role='user',content='你好！')]
    prompt,_=builder.build(messages)
    clean=[v['id'] for e,v in board.stream(prompt,32) if e=='token']
    started=threading.Event()
    long=[dict(role='user',content='请记住这是一条多人排队验证消息。'*150)]
    def signal(e,v):
        if e=='start':started.set()
    count_before=board.request('/health')['requests']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        first=pool.submit(events,long,160,signal)
        assert started.wait(5)
        health_status,active_health=request('/api/health')
        assert health_status==200 and active_health['online'] and active_health['queue']['busy']
        checks['health_stays_online_during_generation']=True
        cancelled=events(messages,32,abort_kind='queue');assert any(e=='queue' for e,_,_ in cancelled)
        third=pool.submit(events,messages,32)
        assert request('/api/reset',{},origin=f'http://{LAN}:{PORT}')[0]==200
        first_data=first.result(15);third_data=third.result(15)
    done=next(v for e,v,_ in first_data if e=='done');assert not done['cancelled']
    assert next(v for e,v,_ in first_data if e=='meta')['input_tokens']==512
    assert any(e=='queue' for e,_,_ in third_data) and any(e=='done' for e,_,_ in third_data)
    # Cached web health must not be used to count native requests.
    after=board.request('/health');assert after['requests']-count_before==2,(count_before,after)
    clean_again=[v['id'] for e,v in board.stream(prompt,32) if e=='token'];assert clean_again==clean
    checks['concurrent_clients']={'long_request':first_data,'queued_request':third_data,'cancelled_queued_request':cancelled,'native_requests':2,'one_browser_clear_does_not_interrupt_other':True,'cancelled_queue_never_reached_npu':True,'clean_answer_reproduced_after_other_client':True}
    events(long,160,abort_kind='delta')
    time.sleep(.3)
    clean_after_cancel=[v['id'] for e,v in board.stream(prompt,32) if e=='token'];assert clean_after_cancel==clean
    checks['active_cancel_then_clean_generation']=True
    rejected=False
    try:board.request('/generate',{'input_ids':[6400]})
    except ConnectionError:rejected=True
    assert rejected;checks['invalid_token_rejected']=True
    final=board.request('/health')
    assert final['model_loads']==1 and final['checkpoint_sha256']==CONFIG['checkpoint_sha256'] and final['context_tokens']==512
    assert final['model']==CONFIG['model'] and final['rknn_sha256']==CONFIG['rknn_sha256']
    checks['board_before']=before;checks['board_after']=final;checks['status']='passed'
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':'passed','benchmark_prefixes':len(compared),'lan':f'http://{LAN}:{PORT}','concurrent_clients':True,'queue_cancel':True,'clear_isolation':True,'native_model_loads':1,'board_rss_kib':final['rss_kib'],'board_dma_kib':final['dma_kib']}))

if __name__=='__main__':
    # Invalid-token rejection is checked separately because Board.request raises
    # for the expected 400 response.
    try:main()
    except Exception:
        import traceback;traceback.print_exc();raise

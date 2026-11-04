import argparse,json,os,subprocess,sys,time,datetime
from pathlib import Path
from setup import prepare,ROOT,HERE,atomic,sha
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def execute(run,resume=False,steps=0):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(run/'source'),TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_CACHE_DIR=str(ROOT/'research/delta-lccc-sft-20261029/inductor-cache'),TRITON_CACHE_DIR=str(ROOT/'research/delta-lccc-sft-20261029/triton-cache'))
    cmd=[sys.executable,'-u','-m','training.train','--config',str(run/'config.json'),'--stop-after-updates',str(steps)]
    if resume:cmd+=['--resume','auto']
    with (run/'pipeline.log').open('a',buffering=1) as log:
        child=subprocess.Popen(cmd,cwd=run/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
        atomic(run/'pipeline-status.json',{'status':'training','pid':child.pid,'started_utc':now(),'command':cmd,'ssh_foreground_required':True})
        try:
            while child.poll() is None:
                time.sleep(15)
                path=run/'sft/status.json'
                if path.exists():
                    r=json.loads(path.read_text());print(json.dumps({k:r.get(k) for k in ('status','event','step','tokens','loss','epochs_completed','tokens_per_second')}),flush=True)
        except BaseException:
            child.terminate()
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    if child.returncode:
        atomic(run/'pipeline-status.json',{'status':'failed','returncode':child.returncode,'ended_utc':now()});print((run/'pipeline.log').read_text()[-9000:],flush=True);raise SystemExit(child.returncode)
    atomic(run/'pipeline-status.json',{'status':'train_complete' if not steps else 'diagnostic_complete','returncode':0,'ended_utc':now()})
    if not steps:
        for name in ('audit.py','compare.py'):
            atomic(run/'pipeline-status.json',{'status':'audit' if name=='audit.py' else 'comparison','started_utc':now()})
            with (run/(name+'.log')).open('w',buffering=1) as log:
                result=subprocess.run([sys.executable,'-u',str(run/name),str(run)],cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                print((run/(name+'.log')).read_text()[-9000:],flush=True)
                atomic(run/'pipeline-status.json',{'status':'postprocess_failed','script':name,'returncode':result.returncode,'training_complete':True});raise SystemExit(result.returncode)
        atomic(run/'pipeline-status.json',{'status':'complete','returncode':0,'ended_utc':now(),'training_complete':True,'comparison_complete':True})
        print((run/'comparison.json').read_text(),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--steps',type=int,default=0);p.add_argument('--resume',action='store_true');p.add_argument('--diagnostic',action='store_true');args=p.parse_args()
    assert '/' not in args.name and '..' not in args.name
    run=ROOT/'training-runs'/args.name
    if not args.resume:
        if not args.steps:
            assert json.loads((HERE/'preflight-report.json').read_text())['status']=='passed'
            assert json.loads((HERE/'resume-validation.json').read_text())['status']=='passed'
        prepare(args.name,args.steps,args.diagnostic)
    execute(run,args.resume,args.steps)
if __name__=='__main__':main()

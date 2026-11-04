import argparse,json,hashlib,shutil,os,subprocess,sys,time,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=ROOT/'training-runs/delta-lccc-sft-2epoch-20261029'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,r):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(r,indent=2));q.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--max-updates',type=int,default=0);p.add_argument('--resume',action='store_true');a=p.parse_args()
    assert '/' not in a.name and '..' not in a.name
    run=ROOT/'training-runs'/a.name
    if not a.resume:
        assert not run.exists();assert sha(PARENT/'sft/last.pt')=='7ca29f788efaf7cbdd6300e8c326d887617373b6dabc4483a6fa034afdad90aa'
        run.mkdir();shutil.copytree(PARENT/'source',run/'source')
        for name in ('train.py','qat_model.py'):shutil.copy2(HERE/name,run/'source'/name)
        file=run/'source/sft_data.py';text=file.read_text();old='SPECS={160:(256,160,4096),256:(64,256,4096),512:(32,256,4096),768:(8,256,1024)}';assert text.count(old)==1
        file.write_text(text.replace(old,'SPECS={160:(128,160,2048),256:(32,256,4096),512:(16,256,4096),768:(8,256,2048)}'))
        file=run/'source/sft_graph.py';text=file.read_text();old="with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):";assert text.count(old)==1
        file.write_text(text.replace(old,"with torch.autocast('cuda',enabled=False):"))
        sources={str(f.relative_to(run/'source')):sha(f) for f in sorted((run/'source').rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}
        atomic(run/'source-manifest.json',sources);shutil.copy2(HERE/'launch.py',run/'launcher-frozen.py')
        atomic(run/'preparation.json',{'parent_sha256':sha(PARENT/'sft/last.pt'),'source_files':len(sources),'seconds':1800,'new_grids_from_train':True,'full_model_trainable':True,'fake_quant_scope':'Delta only','optimizer':'inherited SFT','lr':1e-5,'allocator_limit_gib':12})
    cmd=[sys.executable,'-u',str(run/'source/train.py'),'--seconds','1800']
    if a.max_updates:cmd+=['--max-updates',str(a.max_updates)]
    if a.resume:cmd+=['--resume']
    env=os.environ.copy();env['PYTHONPATH']=str(run/'source')
    with (run/'pipeline.log').open('a',buffering=1) as log:
        child=subprocess.Popen(cmd,env=env,cwd=run/'source',stdout=log,stderr=subprocess.STDOUT)
        atomic(run/'pipeline-status.json',{'status':'running','pid':child.pid,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'ssh_foreground_required':True})
        try:
            while child.poll() is None:
                time.sleep(15)
                f=run/'train/status.json'
                if f.exists():
                    r=json.loads(f.read_text());print(json.dumps({k:r.get(k) for k in ('status','event','step','updates','targets','loss','elapsed_seconds')}),flush=True)
        except BaseException:
            child.terminate()
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    if child.returncode:
        print((run/'pipeline.log').read_text()[-10000:],flush=True);atomic(run/'pipeline-status.json',{'status':'failed','returncode':child.returncode});raise SystemExit(child.returncode)
    manifest=json.loads((run/'source-manifest.json').read_text());assert all(sha(run/'source'/n)==v for n,v in manifest.items())
    r=json.loads((run/'train/completion.json').read_text());assert all(sha(run/'train'/n)==v for n,v in r['artifact_sha256'].items())
    atomic(run/'completion-audit.json',{'status':'passed','source_files_verified':len(manifest),'training':r})
    atomic(run/'pipeline-status.json',{'status':'complete','returncode':0,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'board_pending':True})
    print(json.dumps(r),flush=True)
if __name__=='__main__':main()

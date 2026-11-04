"""Prepare portable run layouts for the original, SHA-verified trainers.

Preparation only. Training is an explicit second command. Original large data
shards and full parent checkpoints must be supplied externally.
"""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8',newline='\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pretrain','sft','qat']);p.add_argument('--data-manifest',type=Path,required=True);p.add_argument('--parent-checkpoint',type=Path);p.add_argument('--asset-manifest',type=Path);p.add_argument('--work',type=Path,default=ROOT/'work');p.add_argument('--name',default='');a=p.parse_args()
    stage=ROOT/'training'/a.stage;work=a.work.resolve();name=a.name or 'release-'+a.stage
    assert name and name.replace('-','').replace('_','').isalnum()
    manifest=a.data_manifest.resolve();assert manifest.is_file()
    data=json.loads(manifest.read_text());expected='316a97a0f816eeacbf7a9ca79646e90dd971c2be842d849bdea1f8c9970ab078' if a.stage=='pretrain' else 'a0b7193c2f8f391f93dfe2f461f79cfd5d825fb8480686d89c64f3149af2e4c8'
    assert data['fingerprint']==expected,'Dataset fingerprint mismatch'
    for shard in data['shards']:
        for split in shard['splits'].values():
            for file,size in split['files'].items():assert Path(file).is_file() and Path(file).stat().st_size==size,file
    parent_sha={'sft':'0fc4252d0245a516a76dca6e23395c1609b85800844bc1b753ea9a1be5d9d6d6','qat':'7ca29f788efaf7cbdd6300e8c326d887617373b6dabc4483a6fa034afdad90aa'}
    if a.stage!='pretrain':assert a.parent_checkpoint and sha(a.parent_checkpoint)==parent_sha[a.stage],'Requires the original full parent checkpoint, not an optimizer-free deployment package'
    if a.stage=='sft':assert a.asset_manifest and a.asset_manifest.is_file(),'Supply the frozen data asset manifest'
    run=work/'training-runs'/name;assert not run.exists(),'Refusing existing run'
    shutil.copytree(stage/'source',run/'source');shutil.copyfile(stage/'source-manifest.json',run/'source-manifest.json')
    manifest_for_run=manifest
    if a.stage=='sft':
        directory=run/'data';directory.mkdir();shutil.copyfile(manifest,directory/'manifest.json');shutil.copyfile(a.asset_manifest,directory/'asset-sha256.json');manifest_for_run=directory/'manifest.json'
    if a.stage=='qat':
        # The exact frozen QAT trainer expects these relative project paths.
        parent=work/'training-runs/delta-lccc-sft-2epoch-20261029/sft/last.pt';parent.parent.mkdir(parents=True,exist_ok=True)
        if parent.exists():assert sha(parent)==parent_sha['qat']
        else:shutil.copyfile(a.parent_checkpoint,parent)
        target=work/'research/delta-lccc-sft-20261029/data/manifest.json';target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==manifest.read_bytes()
        else:shutil.copyfile(manifest,target)
        command=[sys.executable,'-u',str(run/'source/train.py'),'--seconds','1800']
    else:
        cfg=json.loads((stage/'historical'/('train-config.json' if a.stage=='pretrain' else 'config.json')).read_text())
        cfg.update(data=str(manifest_for_run),out=str(run/('pretrain' if a.stage=='pretrain' else 'sft')),model_config=str(stage/'model-config.json'),resume='none')
        if a.stage=='sft':cfg['init_from']=str(a.parent_checkpoint.resolve())
        write(run/'config.json',cfg);command=[sys.executable,'-u','-m','training.train','--config',str(run/'config.json')]
    write(run/'preparation.json',{'stage':a.stage,'historical_source_subset':True,'source_subset_sha256':sha(run/'source-manifest.json'),'historical_manifest':str(ROOT/'provenance'/f'{a.stage}-historical-source-manifest.json'),'data_manifest':str(manifest_for_run),'command':command,'cwd':str(run/'source'),'note':'Portable packaging changes source inventory and paths. This is a fresh run contract, not a historical resume.'})
    print(json.dumps({'prepared':str(run),'command':command,'cwd':str(run/'source'),'training_started':False},indent=2))
if __name__=='__main__':main()

"""Verify Convora source; optionally require external Hugging Face artifacts."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--artifacts',action='store_true',help='Require all Hugging Face model/diagnostic artifacts');a=p.parse_args();root=a.root.resolve()
    manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'));errors=[]
    artifacts=json.loads((root/'ARTIFACTS.json').read_text(encoding='utf-8'))
    for name,info in manifest['files'].items():
        file=root/name
        if not file.is_file() or file.stat().st_size!=info['bytes'] or sha(file)!=info['sha256']:errors.append(name)
    ignored={'.git','__pycache__','.venv','.cache','work','outputs','build'}
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and not any(x in p.parts for x in ignored)}
    extra=actual-set(manifest['files'])-set(artifacts['files'])-{'MANIFEST.json'}
    errors+=['unlisted: '+name for name in sorted(extra)]
    available=0;missing=[]
    for name,info in artifacts['files'].items():
        file=root/name
        if file.exists():
            available+=1
            if file.stat().st_size!=info['bytes'] or sha(file)!=info['sha256']:errors.append('artifact mismatch: '+name)
        elif info['distribution']=='huggingface':
            missing.append(name)
            if a.artifacts:errors.append('missing artifact: '+name)
    sources={}
    for stage in ('pretrain','sft','qat'):
        subset=json.loads((root/f'training/{stage}/source-manifest.json').read_text())
        historical=json.loads((root/f'provenance/{stage}-historical-source-manifest.json').read_text())
        for name,digest in subset.items():
            if sha(root/f'training/{stage}/source'/name)!=digest or digest!=historical.get(name):errors.append('frozen source mismatch: '+stage+'/'+name)
        sources[stage]=len(subset)
    for file in root.rglob('*.py'):
        if any(x in file.parts for x in ignored):continue
        ast.parse(file.read_text(encoding='utf-8-sig'),filename=str(file))
    print(json.dumps({'passed':not errors,'code_files':len(manifest['files']),'code_bytes':sum(v['bytes'] for v in manifest['files'].values()),'frozen_source_files':sources,'available_external_files':available,'missing_optional_artifacts':len(missing),'model_repository':artifacts['model_repository'],'errors':errors},indent=2))
    return bool(errors)
if __name__=='__main__':sys.exit(main())

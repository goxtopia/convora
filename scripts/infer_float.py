"""CPU greedy generation with the pinned MiniMind tokenizer/template."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'modeling'))
import torch
from load import load_float,FloatDecoder

def main():
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True);group.add_argument('--prompt');group.add_argument('--ids-json',type=Path);p.add_argument('--weights',type=Path,default=ROOT/'models/float/qat-final.pt');p.add_argument('--max-new-tokens',type=int,default=80);a=p.parse_args()
    assert 1<=a.max_new_tokens<=512
    torch.set_num_threads(2);tokenizer=None
    # The existing chat bridge owns the exact multi-turn, no-think template.
    sys.path.insert(0,str(ROOT/'runtime/pico-chat'))
    if a.prompt is not None:
        from server import PromptBuilder
        builder=PromptBuilder();ids,_=builder.build([{'role':'user','content':a.prompt}]);tokenizer=builder.tokenizer
    else:ids=json.loads(a.ids_json.read_text(encoding='utf-8'))
    assert isinstance(ids,list) and ids and all(type(t) is int and 0<=t<6400 for t in ids)
    backbone,layer,_=load_float(a.weights);decoder=FloatDecoder(backbone,layer)
    for token in ids:logits=decoder.step(token)
    generated=[]
    for _ in range(a.max_new_tokens):
        token=int(logits.argmax());generated.append(token)
        if token==2:break
        logits=decoder.step(token)
    print(json.dumps({'text':tokenizer.decode(generated,skip_special_tokens=True) if tokenizer else None,'generated_ids':generated},ensure_ascii=False))
if __name__=='__main__':main()

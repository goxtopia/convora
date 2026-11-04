"""Same-parent eight-layer TCN plus one nonzero, untrained delta layer."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'modeling'))
from delta_layer import DeltaLayer
from training.download import sha256
from training.export import CachedStep
from training.model import ModelConfig, TCNLanguageModel
from training.train import load_checkpoint


class HybridStep(nn.Module):
    def __init__(self, model, delta):
        super().__init__()
        self.model, self.delta = model, delta
        self.register_buffer("head_weight", model.output_weight().detach().clone().unsqueeze(2))

    def forward(self, x, *args):
        h, hidden = x, []
        for i, block in enumerate(self.model.blocks):
            temporal = F.conv2d(torch.cat((args[i], h), dim=3), block.weight.unsqueeze(2), block.bias)
            h = h+F.conv2d(F.relu(temporal), self.model.pointwise[i].weight.unsqueeze(2))
            if i == self.model.config.hook:
                h = CachedStep.inject(h, args[8], self.model.engram)
            if i < 7: hidden.append(h)
        states = ()
        if self.delta is not None:
            h, *states = self.delta(h, *args[9:])
        return (F.conv2d(h, self.head_weight), *hidden, *states)


class Runtime:
    def __init__(self, root, cfg, delta):
        options=ort.SessionOptions(); options.intra_op_num_threads=2
        self.session=ort.InferenceSession(str(root/"step.onnx"),sess_options=options,providers=["CPUExecutionProvider"])
        self.cfg,self.delta=cfg,delta
        self.embedding=np.fromfile(root/"data/embedding.f32","<f4").reshape(6400,256)
        self.table=np.fromfile(root/"data/memory.i8","i1").reshape(-1,64).astype(np.float32)
        self.table*=np.fromfile(root/"data/memory_scale.f32","<f4")[:,None]
        self.offsets=np.cumsum([0]+list(cfg.buckets[:-1]));self.reset()

    def reset(self):
        self.rings=[np.zeros((1,256,1,2*d),np.float32) for d in self.cfg.dilations]
        self.ptr=[0]*8;self.previous=[0,0]
        self.states=[np.zeros((1,self.delta.dimension,1,self.delta.dimension),np.float32) for _ in range(4)] if self.delta else []

    def step(self,token,collect=False):
        rows=[]
        for n,m,b,o in zip(self.cfg.orders,self.cfg.hash_multipliers,self.cfg.buckets,self.offsets):
            key=token*m[0]^self.previous[0]*m[1]
            if n==3:key^=self.previous[1]*m[2]
            rows.append((key&0x7fffffff)%b+int(o))
        x=self.embedding[token].reshape(1,256,1,1)
        feeds={"x":x,"memory":self.table[rows].reshape(x.shape)}
        for i,(p,d) in enumerate(zip(self.ptr,self.cfg.dilations)):
            feeds[f"state{i}"]=self.rings[i][:,:,:,[p,(p+d)%(2*d)]]
        for i,s in enumerate(self.states): feeds[f"delta{i}"]=s
        sample={k:v.copy() for k,v in feeds.items()} if collect else None
        result=self.session.run(None,feeds)
        for i,d in enumerate(self.cfg.dilations):
            p=self.ptr[i];self.rings[i][:,:,:,p:p+1]=x if i==0 else result[i]
            self.ptr[i]=(p+1)%(2*d)
        self.states=[v.copy() for v in result[8:]]
        self.previous=[token,self.previous[0]]
        return result[0].reshape(6400),sample,self.states


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--dimension",type=int,default=64)
    p.add_argument("--normalize",choices=["bounded","l2"],default="bounded")
    p.add_argument("--baseline",action="store_true")
    p.add_argument("--parent",type=Path,default=Path("/home/wuyupeng/lccc-1pass-deployment-assets-20261028/checkpoint.pt"))
    p.add_argument("--inputs",type=Path,default=Path("lccc-1pass-board-artifacts/deployment-inputs.json"))
    args=p.parse_args();root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
    for folder in ("data","calibration","models","results"): (root/folder).mkdir()
    torch.set_num_threads(2);torch.manual_seed(110328)
    initial=load_checkpoint(args.parent);cfg=ModelConfig.from_dict(initial["model_config"])
    model=TCNLanguageModel(cfg).eval();model.load_state_dict(initial["model"])
    torch.manual_seed(110328+args.dimension)
    delta=None if args.baseline else DeltaLayer(dimension=args.dimension,normalize=args.normalize).eval()
    hybrid=HybridStep(model,delta).eval()
    names=["x"]+[f"state{i}" for i in range(8)]+["memory"]
    outputs=["logits"]+[f"h{i}" for i in range(7)]
    examples=[torch.zeros(1,256,1,1)]+[torch.zeros(1,256,1,2) for _ in range(8)]+[torch.zeros(1,256,1,1)]
    if delta:
        names += [f"delta{i}" for i in range(4)];outputs += [f"next{i}" for i in range(4)]
        examples += [torch.zeros(1,args.dimension,1,args.dimension) for _ in range(4)]
    torch.onnx.export(hybrid,tuple(examples),str(root/"step.onnx"),opset_version=12,input_names=names,output_names=outputs,do_constant_folding=True)
    (model.embedding.weight.detach()*model.embedding_scale).numpy().astype("<f4").tofile(root/"data/embedding.f32")
    table=torch.cat([t.weight.detach() for t in model.engram.tables]).numpy()
    scales=np.maximum(np.abs(table).max(1)/127,1e-8).astype("<f4")
    np.clip(np.rint(table/scales[:,None]),-127,127).astype("i1").tofile(root/"data/memory.i8")
    scales.tofile(root/"data/memory_scale.f32")
    runtime=Runtime(root,cfg,delta);inputs=json.loads(args.inputs.read_text());lines=[];state_nonzero=0
    for doc in inputs["calibration"][:24]:
        assert doc["split"]=="train";runtime.reset()
        ids=doc["ids"]
        for t,token in enumerate(ids):
            _,sample,_=runtime.step(token, t%max(1,len(ids)//8)==0)
            if sample is not None:
                state_nonzero+=int("delta0" in sample and bool(np.any(sample["delta0"])))
                paths=[]
                for name in names:
                    path=root/"calibration"/f"{len(lines):04d}-{name}.npy"
                    np.save(path,sample[name]);paths.append(str(path))
                lines.append(" ".join(paths))
    (root/"calibration/dataset.txt").write_text("\n".join(lines)+"\n")
    reference=inputs["reference"]["ids"];np.asarray(reference,"<i4").tofile(root/"data/sequence.i32")
    runtime.reset();logits=[];states=[]
    for token in reference:
        logit,_,state=runtime.step(token);logits.append(logit)
        if delta:states.append(np.stack(state).reshape(-1))
    np.asarray(logits,dtype="<f4").tofile(root/"data/reference.f32")
    if delta: np.asarray(states,dtype="<f4").tofile(root/"data/state-reference.f32")
    # Independent vectorized TCN + matrix-multiply delta recurrence.
    with torch.no_grad():
        offset=0
        for layer in model.engram.tables:
            n=len(layer.weight);layer.weight.copy_(torch.from_numpy(runtime.table[offset:offset+n]));offset+=n
        captured={}
        hook=model.head.register_forward_pre_hook(lambda module,values:captured.update(h=values[0].detach().clone()))
        base=model(torch.tensor([reference])).numpy()[0];hook.remove()
        feature=captured["h"];current=[torch.zeros(1,args.dimension,1,args.dimension) for _ in range(4)] if delta else []
        ref=[];raw=[];new=[]
        for t in range(len(reference)):
            h=feature[:,:,t:t+1].unsqueeze(-1);raw.append(h.numpy().reshape(-1))
            if delta:
                h,*current=delta.reference(h,current)
                np.testing.assert_allclose(np.stack([s.numpy() for s in current]).reshape(-1),states[t],rtol=2e-4,atol=3e-5)
            new.append(h.numpy().reshape(-1))
            ref.append(F.conv2d(h,hybrid.head_weight).numpy().reshape(6400))
        np.testing.assert_allclose(np.stack(logits),np.stack(ref),rtol=2e-4,atol=1e-4)
    raw,new=np.stack(raw),np.stack(new)
    ops={}
    for node in onnx.load(str(root/"step.onnx")).graph.node:ops[node.op_type]=ops.get(node.op_type,0)+1
    header=["#pragma once",f"constexpr int C=256,V=6400,L=8,R=133,N={len(reference)},FIRST=0,ROWS={len(reference)},LOOPS=2048;",
            "constexpr int D[L]={1,2,4,8,16,32,1,2};",f"constexpr int DH={4 if delta else 0},DK={args.dimension if delta else 0},DE={4*args.dimension**2 if delta else 0};",
            "constexpr int HN=4,HD=64;","constexpr int BUCKETS[HN]={19997,20011,20021,20023};","constexpr int ORDERS[HN]={2,2,3,3};",
            "constexpr uint64_t HASH_MULT[HN][3]={{841577377,1427818425,1713831177},{1403703963,542880493,52321687},{799045305,480294803,756998953},{1028136769,665279961,1782731651}};"]
    (root/"engram_config.h").write_text("\n".join(header)+"\n")
    report={"parent_sha256":sha256(args.parent),"trained_delta":False,"baseline":args.baseline,"heads":4 if delta else 0,
            "dimension":args.dimension if delta else 0,"normalization":args.normalize,"inputs":len(names),"outputs":len(outputs),"ops":ops,
            "calibration_samples":len(lines),"nonzero_state_samples":state_nonzero,"reference_positions":len(reference),
            "onnx_vs_independent_max_abs":float(np.abs(np.stack(logits)-np.stack(ref)).max()),
            "delta_residual_relative_l2":float(np.linalg.norm(new-raw)/np.linalg.norm(raw)),
            "delta_parameters":sum(p.numel() for p in delta.parameters()) if delta else 0,
            "external_state_int8_bytes":4*args.dimension**2 if delta else 0,
            "external_state_fp32_bytes":16*args.dimension**2 if delta else 0,
            "warning":"Existing trained TCN + untrained nonzero delta-rule layer; no task quality claim"}
    (root/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    from rknn.api import RKNN
    rknn=RKNN(verbose=False)
    try:
        assert rknn.config(target_platform="rv1103",optimization_level=3,quantized_dtype="asymmetric_quantized-8",quantized_method="channel")==0
        assert rknn.load_onnx(model=str(root/"step.onnx"))==0
        assert rknn.build(do_quantization=True,dataset=str(root/"calibration/dataset.txt"))==0
        assert rknn.export_rknn(str(root/"models/tcn_engram.rknn"))==0
    finally:rknn.release()
    report.update(rknn_sha256=sha256(root/"models/tcn_engram.rknn"),rknn_bytes=(root/"models/tcn_engram.rknn").stat().st_size,status="compile_passed")
    (root/"report.json").write_text(json.dumps(report,indent=2));print("HYBRID_COMPILE_PASSED",flush=True)


if __name__=="__main__":main()

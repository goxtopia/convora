"""Exact legacy supervised window IDs/labels; retain every real prefix token.

Regrouping happens ONLY within a legacy effective update of 256 windows.
No loss mask, sample weight or optimizer boundary changes.
"""
import numpy as np
import torch
from training.data import TokenCorpus

SPECS={160:(256,160,4096),256:(64,256,4096),512:(32,256,4096),768:(8,256,1024)}

class ChatCorpus(TokenCorpus):
    def epoch_order(self,seed):
        if not hasattr(self,'orders'):self.orders={};self.target_count=0
        if seed in self.orders:return self.orders[seed]
        selected=[];base=0;total=0
        for tokens,index,cumulative,mask in self.shards:
            count=int(cumulative[-1]);local=np.arange(count,dtype=np.int64)
            docs=np.searchsorted(cumulative,local,side='right')
            previous=np.concatenate(([0],cumulative[:-1]))
            starts=index[docs].astype(np.int64)+(local-previous[docs])*256+1
            assert not np.any(mask[index[:-1]])
            # Adjacent windows partition the loss mask. The one intervening
            # first token of a new document is asserted unsupervised above.
            counts=np.add.reduceat(mask,starts,dtype=np.int64)
            selected.append(base+local[counts>0]);total+=int(counts.sum());base+=count
        order=np.concatenate(selected);np.random.default_rng(seed).shuffle(order)
        self.orders[seed]=order;self.target_count=total
        return order

    def describe(self,windows):
        rows=[]
        for window in windows:
            sid=int(np.searchsorted(self.cumulative,window,side='right'))
            local=int(window)-(int(self.cumulative[sid-1]) if sid else 0)
            tokens,index,cumulative,mask=self.shards[sid]
            doc=int(np.searchsorted(cumulative,local,side='right'))
            previous=int(cumulative[doc-1]) if doc else 0
            start,end=int(index[doc]),int(index[doc+1]);offset=(local-previous)*256
            body=min(256,end-start-1-offset)
            assert offset in (0,256,512) and body>0
            labels=np.array(tokens[start+offset+1:start+offset+body+1],dtype=np.int64)
            if mask is not None:labels[np.asarray(mask[start+offset+1:start+offset+body+1])==0]=-100
            assert (labels!=-100).any()
            width=160 if offset==0 and body<=160 else offset+256
            rows.append((width,tokens,start,offset,body,labels))
        return rows

    @staticmethod
    def pack(rows,width):
        micro,block,capacity=SPECS[width]
        x=np.zeros((len(rows),width),np.int64);valid=np.zeros_like(x,np.bool_);y=np.full((len(rows),block),-100,np.int64)
        for row,(_,tokens,start,offset,body,labels) in enumerate(rows):
            dest=width-block-offset;assert dest>=0
            x[row,dest:width-block+body]=tokens[start:start+offset+body]
            valid[row,dest:width-block+body]=True;y[row,:body]=labels
        rr,cc=np.nonzero(y!=-100)
        indices=rr*width+(width-block)+cc
        targets=y[rr,cc].copy()
        return tuple(torch.from_numpy(t) for t in (x,valid,y,indices,targets))

    def prepare_macro(self,windows):
        groups={width:[] for width in SPECS}
        for row in self.describe(windows):groups[row[0]].append(row)
        packed=[]
        def emit(rows,width):
            if not rows:return
            capacity=SPECS[width][2]
            count=sum(int((r[-1]!=-100).sum()) for r in rows)
            if count>capacity:
                assert len(rows)>1
                middle=len(rows)//2;emit(rows[:middle],width);emit(rows[middle:],width)
            else:packed.append((width,self.pack(rows,width)))
        for width,rows in groups.items():
            micro=SPECS[width][0]
            for i in range(0,len(rows),micro):emit(rows[i:i+micro],width)
        assert sum(len(p[1][0]) for p in packed)==len(windows)
        return packed,sum(len(p[1][-1]) for p in packed)

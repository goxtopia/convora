"""Same-template CPU FP32 diagnostics of three 1pass checkpoints; no teacher."""
import argparse,hashlib,json,sys,time
from pathlib import Path


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):value.update(chunk)
    return value.hexdigest()


def cases():
    casual=[
      ('greeting','你好！'),('identity','你是谁？'),('thanks','谢谢你的帮助。'),
      ('tired','我今天有点累，陪我聊两句吧。'),('weekend','周末想在家放松，有什么简单的建议？'),
      ('cooking','我刚开始学做饭，有点紧张。'),('encourage','能用一句话鼓励我一下吗？'),
      ('rain','外面下雨了，今天不想出门。'),('noodles','我刚煮了一碗面，味道还不错。'),
      ('book','想找本轻松的书看看，你有什么想法？'),('sleep','最近总是熬夜，想早点睡。'),
      ('walk','下班后想散散步，你觉得怎么样？'),('hobby','最近没什么爱好，想试点简单的新东西。'),
      ('nervous','明天第一次参加面试，我有点紧张。'),('friend','朋友今天没回我消息，我有点失落。'),
      ('tea','我喜欢喝茶，不太喜欢咖啡。'),('smalltalk','今天没发生什么特别的事，就是想聊聊天。'),
      ('music','我想听点安静的音乐，放松一下。'),('food','中午不知道吃什么，想吃清淡一点。'),
      ('exercise','我很久没运动了，想从简单的开始。'),('happy','今天终于把房间收拾干净了，有点开心。'),
      ('mistake','今天做错了一件小事，一直有点自责。'),('hello_en','Hello! How are you today?'),
      ('tired_en','I had a tiring day. Can we chat for a moment?')]
    rows=[dict(id=i,category='casual',messages=[dict(role='user',content=q)]) for i,q in casual]
    strict=[
      ('blue','请只回复：蓝色',['蓝色']),('thank_only','只输出“谢谢”两个字，不要解释。',['谢谢']),
      ('banana','从苹果、香蕉中选择香蕉，只输出这个词。',['香蕉']),
      ('yellow','这把雨伞是黄色的。雨伞是什么颜色？只回答颜色。',['黄色']),
      ('one_plus_one','1加1等于几？只回答数字。',['2']),('two_plus_three','2加3等于几？只回答数字。',['5']),
      ('animal','猫是动物吗？只回答“是”或“否”。',['是']),
      ('translation','把“早安”翻译成英语，只输出译文。',['Good morning','Good morning.']),
      ('json','只输出这个JSON对象：{"ok":true}',None),('colors','用顿号连接红、绿、蓝，不添加其他字。',['红、绿、蓝']),
      ('goodnight','不要解释，只回复：晚安',['晚安']),('seven','只输出数字7。',['7']),
      ('letter','从A、B、C中选择B，只输出B。',['B']),('reverse','把“春天”两个字倒过来，只输出结果。',['天春']),
      ('count','“小猫”有几个汉字？只输出数字。',['2']),('code','编号为K7，只输出编号。',['K7'])]
    rows += [dict(id=i,category='strict',messages=[dict(role='user',content=q)],expected=e,check='json' if i=='json' else 'exact') for i,q,e in strict]
    histories=[
      ('name',[('user','请记住我叫阿禾。'),('assistant','好的，阿禾。')],[('name_recall','我叫什么名字？只回答名字。',['阿禾']),('name_copy','请只重复我的名字。',['阿禾'])]),
      ('drink',[('user','我喜欢喝茶，不喜欢咖啡。'),('assistant','知道了，你更喜欢喝茶。')],[('drink_like','我喜欢喝什么？只输出名称。',['茶']),('drink_dislike','我不喜欢喝什么？只输出名称。',['咖啡'])]),
      ('umbrella',[('user','我的包里有钥匙和红色雨伞。'),('assistant','好的，我记住了。')],[('umbrella_color','雨伞是什么颜色？只回答颜色。',['红色']),('umbrella_item','包里除了雨伞还有什么？只回答物品名称。',['钥匙'])]),
      ('fruit',[('user','我今天买了两个苹果，没有买香蕉。'),('assistant','知道了，你买了两个苹果。')],[('fruit_count','我买了几个苹果？只输出数字。',['2']),('fruit_absent','我没有买哪种水果？只输出水果名称。',['香蕉'])])]
    for family,history,questions in histories:
        for i,q,e in questions:rows.append(dict(id=i,category='multi_strict',family=family,messages=[dict(role=r,content=c) for r,c in history]+[dict(role='user',content=q)],expected=e,check='exact'))
    assert len(rows)==48
    return rows


def prefix(tokenizer,messages):
    pieces=[]
    for m in messages:
        header='<|im_start|>'+m['role']+'\n'
        if m['role']=='assistant':header+='<think>\n\n</think>\n\n'
        pieces.append(header+m['content']+'<|im_end|>\n')
    pieces.append('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    return tokenizer.encode(''.join(pieces),add_special_tokens=False).ids


def correct(row,text):
    if row.get('check')=='exact':return text.strip() in row['expected']
    if row.get('check')=='json':
        try:return json.loads(text)=={'ok':True} and type(json.loads(text).get('ok')) is bool
        except (ValueError,AttributeError):return False
    return None


def repetition(ids):
    longest=run=1
    for a,b in zip(ids,ids[1:]):
        run=run+1 if a==b else 1;longest=max(longest,run)
    grams=[tuple(ids[i:i+4]) for i in range(max(0,len(ids)-3))]
    return dict(max_identical_token_run=longest if ids else 0,duplicate_4gram_fraction=1-len(set(grams))/len(grams) if grams else 0.)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();sys.path.insert(0,str(root/'source'))
    import torch
    from tokenizers import Tokenizer
    from training.model import ModelConfig,TCNLanguageModel
    from training.streaming import CachedDecoder
    from training.train import load_checkpoint
    torch.set_num_threads(4);torch.manual_seed(42)
    tokfile=root/'tokenizer.json';tokenizer=Tokenizer.from_file(str(tokfile));tokenizer.no_padding();tokenizer.no_truncation()
    tsha=sha(tokfile);questions=cases();(root/'cases.json').write_text(json.dumps(questions,ensure_ascii=False,indent=2))
    names={'old-final':('old-final.pt','c62e0cb4781e4e080eef7f518a053010ff07908cf6b2fa219f7a8ee3cc8ebbfa'),
           'lccc-mixed':('mixed-last.pt','3720565dc4f08f9c5a2da11a729b212adbe50ba83d451c1b2c4d5de4bede68f6'),
           'lccc-final':('finetune-last.pt','0d7d9fe8deeb9f8533e6327d844764e0bf9f1116df38c0a46a371ee761018f92')}
    report=dict(scope='48 illustrative same-input diagnostics, not unseen blind accuracy or equal-budget data ablation',precision='CPU FP32',template='minimind3_nothink_v1',max_new_tokens=80,decoding='greedy, no penalties',tokenizer_sha256=tsha,models={})
    for name,(filename,expected_sha) in names.items():
        checkpoint=root/filename;digest=sha(checkpoint);assert digest==expected_sha
        saved=load_checkpoint(checkpoint);assert saved['tokenizer_sha256']==tsha
        cfg=ModelConfig.from_dict(saved['model_config']);assert cfg.num_passes==1
        model=TCNLanguageModel(cfg).eval();model.load_state_dict(saved['model']);del saved
        predictions=[];checks=[];started=time.time()
        with torch.inference_mode():
            for row in questions:
                prompt=prefix(tokenizer,row['messages']);cache=CachedDecoder(model)
                for token in prompt:logits=cache.step(token)
                history=list(prompt);generated=[]
                for step in range(80):
                    if row['id'] in ['greeting','tired','blue','name_recall'] and step in [0,1,7,15]:
                        reference=model(torch.tensor([history[-(cfg.context+1):]],dtype=torch.long))[:,-1,:]
                        error=float((reference-logits).abs().max());checks.append(dict(id=row['id'],step=step,max_logit_difference=error,same_argmax=int(reference.argmax())==int(logits.argmax())))
                    token=int(logits.argmax());generated.append(token);history.append(token)
                    if token==2:break
                    logits=cache.step(token)
                text=tokenizer.decode(generated,skip_special_tokens=True)
                predictions.append(dict(row,text=text,generated_ids=generated,eos=generated[-1]==2,new_tokens=len(generated),prompt_tokens=len(prompt),strict_correct=correct(row,text),**repetition(generated)))
        single=[r for r in predictions if r['category']=='strict'];multi=[r for r in predictions if r['category']=='multi_strict'];casual=[r for r in predictions if r['category']=='casual']
        summary=dict(checkpoint_sha256=digest,parameters=model.parameter_counts(),strict_single=dict(correct=sum(r['strict_correct'] for r in single),total=len(single)),strict_multi=dict(correct=sum(r['strict_correct'] for r in multi),total=len(multi)),eos_answers=sum(r['eos'] for r in predictions),answers=len(predictions),mean_new_tokens=sum(r['new_tokens'] for r in predictions)/len(predictions),casual_mean_duplicate_4gram_fraction=sum(r['duplicate_4gram_fraction'] for r in casual)/len(casual),max_cache_logit_difference=max(r['max_logit_difference'] for r in checks),cache_argmax_matches=sum(r['same_argmax'] for r in checks),cache_checks=len(checks),elapsed_seconds=time.time()-started)
        (root/(name+'-predictions.json')).write_text(json.dumps(predictions,ensure_ascii=False,indent=2));(root/(name+'-cache-checks.json')).write_text(json.dumps(checks,indent=2));report['models'][name]=summary
        (root/'generation-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(name,json.dumps(summary),flush=True)
    print('LOCAL_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()

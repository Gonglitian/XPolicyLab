"""Exercise X-VLA's native trainer on real benchmark delta-action labels.

This explicit new action-space adapter preserves the 20-D pretrained projections.
Only true benchmark channels enter the loss; padding is never a training target.
It is a training plumbing check, not the published absolute-EEF evaluation recipe.
"""
import argparse
import json
from pathlib import Path
import random
import sys

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset
from PIL import Image
from torchvision import transforms as tv

XVLA_ROOT=Path(__file__).parent/'xvla'
sys.path.insert(0,str(XVLA_ROOT))
import train as upstream_train
from models.modeling_xvla import XVLA
from models.action_hub import BaseActionSpace, register_action


class DeltaActionSpace(BaseActionSpace):
    dim_action=20
    actual_dim=7
    binary=(6,)
    def compute_loss(self,pred,target):
        assert pred.shape==target.shape
        continuous=[i for i in range(self.actual_dim) if i not in self.binary]
        return {'delta_action_loss':torch.nn.functional.mse_loss(pred[...,continuous],target[...,continuous]),
                'binary_control_loss':torch.nn.functional.binary_cross_entropy_with_logits(pred[...,self.binary],target[...,self.binary])}
    def preprocess(self,proprio,action,mode='train'):
        action=action.clone(); action[...,self.binary]=0
        return proprio,action
    def postprocess(self,action):
        action=action.clone(); action[...,self.binary]=action[...,self.binary].sigmoid()
        return action

@register_action('smoke_libero_delta')
class LiberoDeltaActionSpace(DeltaActionSpace): pass

@register_action('smoke_robocasa_delta')
class RoboCasaDeltaActionSpace(DeltaActionSpace):
    actual_dim=12
    binary=(6,11)


class BenchmarkSamples(IterableDataset):
    def __init__(self,path,benchmark,output):
        self.benchmark=benchmark
        self.domain_id=3 if benchmark=='libero' else 7
        self.episodes=[]
        self.image_transform=tv.Compose([tv.Resize((224,224)),tv.ToTensor(),tv.Normalize((.485,.456,.406),(.229,.224,.225))])
        tasks={r['task_index']:r['task'] for r in map(json.loads,(path/'meta/tasks.jsonl').read_text().splitlines())}
        ids=[r['episode_index'] for r in map(json.loads,(path/'meta/episodes.jsonl').read_text().splitlines())]
        if benchmark=='robocasa': random.Random(0).shuffle(ids)
        self.ids=ids[:8]
        keys=['image','wrist_image'] if benchmark=='libero' else ['robot0_agentview_left','robot0_eye_in_hand','robot0_agentview_right']
        for eid in self.ids:
            frame=pd.read_parquet(path/f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet')
            state=np.stack(frame['observation.state']).astype('float32')
            action=np.stack(frame['action']).astype('float32')
            if benchmark=='robocasa': action=action[:,[5,6,7,8,9,10,11,0,1,2,3,4]]
            binary=(6,) if benchmark=='libero' else (6,11)
            action[:,binary]=(action[:,binary]+1)/2
            assert np.all((action[:,binary]>=0)&(action[:,binary]<=1))
            videos=[]
            for key in keys:
                video=path/f'videos/chunk-{eid//1000:03d}/observation.images.{key}/episode_{eid:06d}.mp4'
                with av.open(str(video)) as container:
                    frames=np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])
                assert len(frames)==len(frame)
                videos.append(frames)
            task_key='task_index' if benchmark=='libero' else 'annotation.human.task_description'
            instructions=[tasks[int(i)] for i in frame[task_key]]
            self.episodes.append((state,action,videos,instructions))
        states=np.concatenate([e[0] for e in self.episodes]); actions=np.concatenate([e[1] for e in self.episodes])
        state_mean,state_std=states.mean(0),states.std(0)
        state_std=np.where(state_std>1e-3,state_std,1)
        action_mean,action_std=actions.mean(0),actions.std(0)
        action_std=np.where(action_std>1e-3,action_std,1)
        action_mean[list(binary)]=0; action_std[list(binary)]=1
        self.norm={'state_mean':state_mean.tolist(),'state_std':state_std.tolist(),
                   'action_mean':action_mean.tolist(),'action_std':action_std.tolist()}
        for i,(state,action,videos,instructions) in enumerate(self.episodes):
            self.episodes[i]=((state-state_mean)/state_std,(action-action_mean)/action_std,videos,instructions)
        self.records=[(e,t) for e,episode in enumerate(self.episodes) for t in range(len(episode[0]))]
        (output/'normalization.json').write_text(json.dumps(self.norm,indent=2)+'\n')
    def sample(self,e,t,num_actions=30):
        states,actions,videos,instructions=self.episodes[e]
        indices=np.minimum(np.arange(t,t+num_actions),len(actions)-1)
        images=[self.image_transform(Image.fromarray(v[t])) for v in videos]
        image_mask=torch.tensor([True]*len(images)+[False]*(3-len(images)))
        while len(images)<3: images.append(torch.zeros_like(images[0]))
        return {'domain_id':torch.tensor(self.domain_id), 'language_instruction':instructions[t],
            'image_input':torch.stack(images),'image_mask':image_mask,
            'proprio':torch.from_numpy(np.pad(states[t],(0,20-states.shape[1]))),
            'action':torch.from_numpy(np.pad(actions[indices],((0,0),(0,20-actions.shape[1]))))}
    def __iter__(self):
        while True:
            order=list(self.records); random.shuffle(order)
            for e,t in order: yield self.sample(e,t)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',choices=['libero','robocasa'],required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--data-only',action='store_true')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    dataset=BenchmarkSamples(args.dataset,args.benchmark,args.output)
    sample=dataset.sample(0,0)
    record={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    record.update(seed=42,demo_ids=dataset.ids,frames=len(dataset.records),domain_id=dataset.domain_id,
        action_mode='smoke_'+args.benchmark+'_delta',
        note='Custom delta-action adapter; native XVLA optimizer/backward/saving; no padded channels in loss; not published absolute-EEF recipe',
        sample_shapes={k:list(v.shape) for k,v in sample.items() if hasattr(v,'shape')})
    (args.output/'run_config.json').write_text(json.dumps(record,indent=2)+'\n')
    print('DATA_READY',json.dumps(record),flush=True)
    if args.data_only:return
    assert (args.base/'download_verified.json').exists()
    holder={}
    def model_factory(path):
        model=XVLA.from_pretrained(path)
        model.action_space=LiberoDeltaActionSpace() if args.benchmark=='libero' else RoboCasaDeltaActionSpace()
        model.action_mode=model.config.action_mode=record['action_mode']
        model.config.smoke_normalization=dataset.norm
        model.config.smoke_domain_id=dataset.domain_id
        holder['model']=model
        name,parameter=next((n,v) for n,v in model.named_parameters() if 'action_decoder' in n and n.endswith('weight'))
        holder.update(probe_name=name,probe=parameter.detach().float().cpu().clone(),parameter=parameter)
        return model
    def loader_factory(**kwargs):
        return DataLoader(dataset,batch_size=kwargs['batch_size'],num_workers=0,pin_memory=True)
    train_args=upstream_train.get_args_parser().parse_args([
        '--models',str(args.base),'--output_dir',str(args.output),'--train_metas_path',str(args.dataset),
        '--iters',str(args.steps),'--batch_size',str(args.batch_size),'--learning_rate',str(args.lr),
        '--learning_coef','0.1','--freeze_steps','20','--warmup_steps','10',
        '--save_interval',str(args.steps),'--log_interval','5','--seed','42'])
    upstream_train.main(train_args,dataloader_factory=loader_factory,model_factory=model_factory)
    saved=args.output/f'ckpt-{args.steps}'
    upstream_train.XVLAProcessor.from_pretrained(args.base).save_pretrained(saved)
    # Reload with the explicit delta action spaces registered by this module.
    reloaded=XVLA.from_pretrained(saved)
    reloaded_probe=dict(reloaded.named_parameters())[holder['probe_name']].detach().float().cpu()
    assert torch.equal(reloaded_probe,holder['parameter'].detach().float().cpu())
    (args.output/'reload_check.json').write_text(json.dumps({
        'status':'passed','method':'XVLA.from_pretrained(saved checkpoint) with training_smoke action registry',
        'parameter_exact_match':True,'checkpoint':str(saved)},indent=2)+'\n')
    del reloaded,reloaded_probe
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events=list(args.output.rglob('events.out.tfevents.*'))
    rows={}
    for event in events:
        accumulator=EventAccumulator(str(event)).Reload()
        for key in accumulator.Tags()['scalars']:
            if 'loss' in key or key.startswith('lr_'):
                for item in accumulator.Scalars(key):rows.setdefault(item.step,{'step':item.step})[key]=item.value
    history=[rows[k] for k in sorted(rows)]
    (args.output/'loss.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in history))
    loss=[r['loss_total'] for r in history]
    assert loss and np.isfinite(loss).all()
    record.update(status='completed',first_window_mean=float(np.mean(loss[:5])),last_window_mean=float(np.mean(loss[-5:])),
        parameter_probe_name=holder['probe_name'],parameter_probe_max_change=float((holder['parameter'].detach().float().cpu()-holder['probe']).abs().max()))
    assert record['parameter_probe_max_change']>0
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print('TRAINING_COMPLETE',json.dumps(record),flush=True)

if __name__=='__main__':main()

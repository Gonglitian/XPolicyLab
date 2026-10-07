"""Real-data short LoRA training through OpenPI's existing JAX train.py."""
import argparse
import dataclasses
import importlib.util
import json
import math
from pathlib import Path
import random

import numpy as np
import pandas as pd
from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.policies import libero_policy
from openpi.shared.normalize import NormStats
from openpi.training import config as cfg, optimizer, weight_loaders, data_loader


@dataclasses.dataclass(frozen=True)
class SmokeLiberoDataConfig(cfg.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                'observation/image':'observation.images.image',
                'observation/wrist_image':'observation.images.wrist_image',
                'observation/state':'observation.state','actions':'action','prompt':'prompt'})]),
            data_transforms=transforms.Group(inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
                                             outputs=[libero_policy.LiberoOutputs()]),
            model_transforms=cfg.ModelTransformFactory()(model_config),
            action_sequence_keys=('action',),use_quantile_norm=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',choices=['libero','robocasa'],required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--openpi-root',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--data-only',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    episodes=[json.loads(line)['episode_index'] for line in (args.dataset/'meta/episodes.jsonl').read_text().splitlines()]
    if args.benchmark=='robocasa':
        random.Random(0).shuffle(episodes)
    selected=episodes[:8]
    frames=[pd.read_parquet(args.dataset/f'data/chunk-{e//1000:03d}/episode_{e:06d}.parquet') for e in selected]
    states=np.concatenate([np.stack(f['observation.state']) for f in frames])
    actions=np.concatenate([np.stack(f['action']) for f in frames])
    if args.benchmark=='robocasa':
        states=states[:,[7,8,9,10,11,12,13,0,1,2,3,4,5,6,14,15]]
        actions=actions[:,[5,6,7,8,9,10,11,0,1,2,3,4]]
    def norm(x):
        # Real dimensions preserve zero std; official Normalize adds epsilon.
        return NormStats(mean=np.pad(x.mean(0),(0,32-x.shape[1])),
                         std=np.pad(x.std(0),(0,32-x.shape[1]),constant_values=1))
    norm_stats={'state':norm(states),'actions':norm(actions)}
    base=cfg.DataConfig(norm_stats=norm_stats,prompt_from_task=args.benchmark=='libero',use_quantile_norm=False)
    model=Pi0Config(pi05=True,action_horizon=10 if args.benchmark=='libero' else 50,
                   discrete_state_input=args.benchmark=='robocasa',
                   paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
    if args.benchmark=='libero':
        data=SmokeLiberoDataConfig(repo_id='smoke/libero',base_config=base)
    else:
        data=cfg.LeRobotRobocasaDataConfig(data_dirs=[{'path':str(args.dataset),'filter_key':'8_demos'}],base_config=base)
    config=cfg.TrainConfig(name='pi05_'+args.benchmark+'_smoke',exp_name='attempt01',
        model=model,data=data,freeze_filter=model.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(args.base/'params')),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=10,peak_lr=args.lr,decay_steps=args.steps,decay_lr=args.lr*.1),
        ema_decay=None,batch_size=args.batch_size,num_workers=0,num_train_steps=args.steps,
        log_interval=5,save_interval=args.steps,keep_period=None,wandb_enabled=False,
        checkpoint_base_dir=str(args.output/'checkpoints'),assets_base_dir=str(args.output/'assets'),seed=42,fsdp_devices=1)
    manifest={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest.update(seed=42,demo_ids=selected,frames=len(states),method='OpenPI JAX LoRA',
        normalization='mean/std computed only from the selected eight training demonstrations',
        action_horizon=model.action_horizon,initialization='fresh pi05_base JAX parameters',
        checkpoint_dir=str(config.checkpoint_dir),freeze_filter=str(config.freeze_filter))
    (args.output/'run_config.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.data_only:
        loader=data_loader.create_data_loader(config,shuffle=True,num_batches=1)
        observation,action=next(iter(loader))
        print('DATA_READY',args.benchmark,observation.state.shape,action.shape,
              {k:v.shape for k,v in observation.images.items()},flush=True)
        assert np.isfinite(np.asarray(action)).all()
        return
    assert (args.base/'download_verified.json').exists()
    import wandb
    history=[]
    original_log=wandb.log
    def record_log(values,*a,**kw):
        if 'loss' in values:
            row={'step':kw.get('step'),**{k:float(v) for k,v in values.items()}}
            if not all(math.isfinite(v) for k,v in row.items() if k!='step'):
                raise RuntimeError('Non-finite training metric')
            history.append(row)
            with (args.output/'loss.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        return original_log(values,*a,**kw)
    spec=importlib.util.spec_from_file_location('openpi_smoke_train',args.openpi_root/'scripts/train.py')
    trainer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    original_init=trainer.init_wandb
    def init_and_record(*a,**kw):
        nonlocal original_log
        original_init(*a,**kw)
        # wandb.init replaces module-level wandb.log, even in disabled mode.
        original_log=wandb.log
        wandb.log=record_log
    trainer.init_wandb=init_and_record
    trainer.main(config)
    losses=[v['loss'] for v in history]
    if not losses:
        raise RuntimeError('Training finished but metric recorder received no losses; inspect native stdout')
    manifest.update(status='completed',first_window_mean=sum(losses[:5])/len(losses[:5]),
        last_window_mean=sum(losses[-5:])/len(losses[-5:]),loss_logs=len(losses))
    (args.output/'result.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('TRAINING_COMPLETE',json.dumps(manifest),flush=True)

if __name__=='__main__': main()

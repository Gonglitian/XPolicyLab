"""Bounded training check using NVIDIA's real dataset, model and Trainer."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch
from transformers import TrainingArguments, TrainerCallback, set_seed
from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['libero', 'robocasa'], required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--libero-config', type=Path)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    if args.benchmark == 'robocasa':
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        config = DATA_CONFIG_MAP['panda_omron']
    else:
        spec = importlib.util.spec_from_file_location('smoke_libero_config', args.libero_config)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class SmokeLiberoDataConfig(module.LiberoDataConfig):
            # The RoboCasa fork makes this method abstract; preserve NVIDIA's
            # LIBERO keys/indices explicitly when using its training runtime.
            def modality_config(self):
                return {name: ModalityConfig(delta_indices=indices, modality_keys=keys)
                        for name,indices,keys in [
                            ('video',self.observation_indices,self.video_keys),
                            ('state',self.observation_indices,self.state_keys),
                            ('action',self.action_indices,self.action_keys),
                            ('language',self.observation_indices,self.language_keys)]}
        config = SmokeLiberoDataConfig()
    dataset = LeRobotSingleDataset(dataset_path=args.dataset,
        modality_configs=config.modality_config(), transforms=config.transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, video_backend='opencv',
        filter_key='8_demos', filter_key_seed=0)
    sample = dataset[0]
    manifest = {**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'seed':42,'dataset_frames':len(dataset),'subset_demos':dataset.subset_demos,
        'initialization':'fresh foundation checkpoint; no benchmark-finetuned weights',
        'tune_llm':False,'tune_visual':False,'tune_projector':True,'tune_diffusion_model':True,
        'sample_shapes':{k:list(v.shape) for k,v in sample.items() if hasattr(v,'shape')}}
    (args.output/'run_config.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('DATA_READY', json.dumps(manifest), flush=True)
    if args.data_only:
        return
    assert (args.base/'download_verified.json').exists(), 'Base weights must be verified first'
    model = GR00T_N1_5.from_pretrained(str(args.base), tune_llm=False,tune_visual=False,
                                     tune_projector=True,tune_diffusion_model=True)
    assert model.action_head.config.action_horizon == len(config.action_indices)
    model.compute_dtype = model.config.compute_dtype = 'bfloat16'
    trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    manifest['trainable_parameters'] = sum(p.numel() for _,p in trainable)
    probe_name, probe_parameter = trainable[-1]
    probe = probe_parameter.detach().float().cpu().clone()
    history=[]
    class RecordLoss(TrainerCallback):
        def on_log(self, arguments, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                if not math.isfinite(logs['loss']):
                    raise RuntimeError('Non-finite training loss')
                record={'step':state.global_step,**logs}
                history.append(record)
                with (args.output/'loss.jsonl').open('a') as f:
                    f.write(json.dumps(record)+'\n')
    training_args = TrainingArguments(output_dir=str(args.output),remove_unused_columns=False,
        bf16=True,tf32=True,per_device_train_batch_size=args.batch_size,
        dataloader_num_workers=2,dataloader_pin_memory=False,dataloader_persistent_workers=True,
        optim='adamw_torch',adam_beta1=.95,adam_beta2=.999,adam_epsilon=1e-8,
        learning_rate=args.lr,weight_decay=1e-5,warmup_ratio=.05,lr_scheduler_type='cosine',
        logging_steps=5,logging_nan_inf_filter=False,max_steps=args.steps,
        save_strategy='steps',save_steps=args.steps,save_total_limit=1,
        report_to='tensorboard',seed=42,ddp_find_unused_parameters=False)
    runner = TrainRunner(model=model,training_args=training_args,train_dataset=dataset)
    runner.trainer.add_callback(RecordLoss())
    runner.train()
    manifest['parameter_probe_name']=probe_name
    manifest['parameter_probe_max_change']=float((probe_parameter.detach().float().cpu()-probe).abs().max())
    if manifest['parameter_probe_max_change'] == 0:
        raise RuntimeError('Probed trainable parameter did not update')
    losses=[v['loss'] for v in history]
    manifest.update(status='completed',first_window_mean=sum(losses[:5])/len(losses[:5]),
                    last_window_mean=sum(losses[-5:])/len(losses[-5:]),loss_logs=len(losses))
    (args.output/'result.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('TRAINING_COMPLETE',json.dumps(manifest),flush=True)

if __name__ == '__main__':
    main()

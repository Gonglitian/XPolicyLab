"""Read-only timing summary. Excludes technical preflight and JIT startup windows."""
import json
import statistics
from common import RUN, SUITES, HORIZONS
from settings import METHODS

def main():
    speed8, speed16, rollouts = [], [], []
    completed8 = completed16 = completed_stages = cells = 0
    target8 = len(SUITES) * (100000 * ('sf' in METHODS) + 10000 * ('er' in METHODS))
    target16 = len(SUITES) * 90000 * ('er' in METHODS)
    target_cells = len(SUITES) * len(METHODS) * 55
    for method in METHODS:
        for suite in SUITES:
            for task in range(10):
                stage = RUN / method / suite / f'task{task:02d}'
                path = stage / 'metrics.jsonl'
                records = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
                big = method == 'er' and task > 0
                done = max((r['step'] for r in records), default=0)
                # Contention can save between the 50-step metric boundaries.
                checkpoint_dir = stage / 'checkpoints/pi05_libero_cl/stage'
                saved = [int(p.name) for p in checkpoint_dir.glob('*')
                         if p.is_dir() and p.name.isdigit()]
                done = max([done, *saved])
                if big:
                    completed16 += done
                else:
                    completed8 += done
                for r in records:
                    if r['step'] >= 100 and r.get('idle_gpu_timing_valid') is True:
                        (speed16 if big else speed8).append(r['seconds_per_step'])
                completed_stages += int((stage / 'trained.json').exists())
                cells += task + 1 if (stage / 'evaluated.json').exists() else 0
                evaluated = stage / 'evaluated.json'
                valid_eval = evaluated.exists() and json.loads(evaluated.read_text()).get('idle_gpu_timing_valid') is True
                for file in (stage / 'evaluation').glob('worker*.json') if valid_eval else []:
                    rows = json.loads(file.read_text())['episodes']
                    # First rollout per worker contains inference JIT; don't extrapolate it.
                    rollouts.extend(dict(suite=suite, **r) for r in rows[1:])
    result = dict(status=json.loads((RUN / 'status.json').read_text()),
        completed_updates=completed8+completed16, target_updates=target8+target16,
        completed_training_stages=completed_stages, completed_cells=cells, target_cells=target_cells,
        batch8_seconds_per_step=statistics.median(speed8) if speed8 else None,
        batch16_seconds_per_step=statistics.median(speed16) if speed16 else None,
        measured_timing_windows=dict(batch8=len(speed8), batch16=len(speed16)),
        measured_eval_episodes=len(rollouts),
        timing_basis='Only four-idle-GPU observations; legacy shared-GPU windows excluded',
        limitations=['Excludes GPU waiting time, model load, checkpoint I/O, evaluation startup, and future failures.',
                     'Unmeasured batch16 speed is a 1x-2x scenario, not a measurement.',
                     'Later-suite episode lengths are uncertain until evaluated.'])
    if speed8:
        t8 = statistics.median(speed8)
        t16 = [statistics.median(speed16)]*2 if speed16 else [t8, 2*t8]
        result['remaining_training_hours_scenario'] = [((target8-completed8)*t8+(target16-completed16)*v)/3600 for v in t16]
        result['total_training_hours_scenario'] = [(target8*t8+target16*v)/3600 for v in t16]
    if rollouts:
        seconds_per_control = statistics.median(r['seconds']/(r['steps']+10) for r in rollouts)
        result['median_episode_seconds'] = statistics.median(r['seconds'] for r in rollouts)
        result['eval_hours_all_episodes_at_suite_horizon_scenario'] = sum(len(METHODS)*55*13*(HORIZONS[s]+10)*seconds_per_control for s in SUITES)/3600
        result['eval_hours_same_episode_duration_scenario'] = target_cells*13*result['median_episode_seconds']/3600
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

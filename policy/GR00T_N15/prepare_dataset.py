"""Validate prepared LeRobot data and create a source-preserving metadata view."""
import argparse
import json
import shutil
from pathlib import Path


def validate_dataset(source, benchmark, modality):
    source = Path(source).resolve(strict=True)
    info = json.loads((source / 'meta/info.json').read_text())
    if info.get('codebase_version') not in {'v2.0', 'v2.1'}:
        raise ValueError('N1.5 requires LeRobot v2.0/v2.1; convert v3 separately')
    features = info.get('features', {})
    expected = (8, 7) if benchmark == 'libero' else (16, 12)
    for group, key, dim in zip(('state', 'action'), ('observation.state', 'action'), expected):
        if features.get(key, {}).get('shape') != [dim]:
            raise ValueError(f'{key}: expected shape [{dim}]')
        covered = []
        for spec in modality[group].values():
            start, end = spec['start'], spec['end']
            if not 0 <= start < end <= dim:
                raise ValueError(f'Invalid {group} slice: {spec}')
            covered.extend(range(start, end))
        if sorted(covered) != list(range(dim)):
            raise ValueError(f'{group} modality must cover every dimension exactly once')
    for spec in modality['video'].values():
        key = spec['original_key']
        if features.get(key, {}).get('dtype') not in {'video', 'image'}:
            raise ValueError(f'Missing image feature: {key}')
        if features[key]['dtype'] == 'video' and not (source / 'videos').is_dir():
            raise ValueError('Missing videos directory')
    for required in ('meta/tasks.jsonl', 'meta/episodes.jsonl', 'data'):
        if not (source / required).exists():
            raise FileNotFoundError(source / required)
    if not any((source / 'meta' / f).is_file() for f in ('stats.json', 'episodes_stats.jsonl')):
        raise ValueError('Missing normalization statistics')
    if min(int(info.get('total_episodes', 0)), int(info.get('total_frames', 0))) < 1:
        raise ValueError('Dataset is empty')
    return info


def prepare(source, target, benchmark, modality, provenance):
    if Path(target).is_symlink():
        raise FileExistsError(f'Refusing to replace a dataset symlink: {target}')
    source, target = Path(source).resolve(strict=True), Path(target).resolve()
    info = validate_dataset(source, benchmark, modality)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'Refusing to overwrite {target}')
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Source and output must be separate non-nested directories')
    target.mkdir(parents=True)
    shutil.copytree(source / 'meta', target / 'meta')
    for name in ('data', 'videos', 'images'):
        if (source / name).exists():
            (target / name).symlink_to(source / name, target_is_directory=True)
    (target / 'meta/modality.json').write_text(json.dumps(modality, indent=2) + '\n')
    record = dict(provenance, source=str(source), benchmark=benchmark,
                  codebase_version=info['codebase_version'], method='metadata-copy/data-symlink')
    (target / 'meta/evomoe_source.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Prepared {target}; source unchanged at {source}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--benchmark', choices=('libero', 'robocasa365'), required=True)
    parser.add_argument('--modality', type=Path, required=True)
    parser.add_argument('--source-kind', choices=('libero', 'human', 'mimicgen'), required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if (args.benchmark == 'libero') != (args.source_kind == 'libero'):
        parser.error('LIBERO requires source-kind=libero; RoboCasa365 requires human or mimicgen')
    modality = json.loads(args.modality.read_text())
    if args.validate_only:
        info = validate_dataset(args.source, args.benchmark, modality)
        print(json.dumps({k: info.get(k) for k in ('codebase_version', 'total_episodes', 'total_frames', 'fps')}))
    elif args.target is None:
        parser.error('--target is required unless --validate-only')
    else:
        prepare(args.source, args.target, args.benchmark, modality, {'source_kind': args.source_kind})

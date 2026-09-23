"""Download requested assets with recorded revisions; never move/delete source data."""
import argparse
import base64
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def within(root, name):
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside asset directory: {name}')
    return result

def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.partial')
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request, timeout=180) as response:
        append = offset and response.status == 206
        if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise RuntimeError('Unexpected Content-Range; preserving partial transfer')
        with part.open('ab' if append else 'wb') as out:
            shutil.copyfileobj(response, out, 8*1024*1024)
    part.replace(path)

def hf(repo, target, patterns=None, revision=None, repo_type='model', ignore=None):
    from huggingface_hub import HfApi, snapshot_download
    lock = target / 'download_manifest.json'
    previous = json.loads(lock.read_text()) if lock.exists() else None
    if previous and (previous['repo_id'] != repo or previous['repo_type'] != repo_type):
        raise ValueError(f'Asset directory belongs to a different repository: {target}')
    if previous and revision and previous['revision'] != revision:
        raise ValueError(f'Existing asset revision differs: {target}')
    revision = previous['revision'] if previous else revision
    api = HfApi()
    info = api.repo_info(repo, repo_type=repo_type, revision=revision, files_metadata=True)
    files = [f for f in info.siblings
             if (not patterns or any(fnmatch.fnmatch(f.rfilename, p) for p in patterns))
             and not any(fnmatch.fnmatch(f.rfilename, p) for p in (ignore or []))]
    if not files:
        raise ValueError(f'No requested files found in {repo}: {patterns}')
    manifest = dict(repo_id=repo, repo_type=repo_type, revision=info.sha,
                    files=[dict(path=f.rfilename, bytes=f.size, sha256=f.lfs.sha256 if f.lfs else None) for f in files])
    save(lock, manifest)
    snapshot_download(repo_id=repo, repo_type=repo_type, revision=info.sha,
                      local_dir=target, allow_patterns=patterns, ignore_patterns=ignore)
    for file in manifest['files']:
        path = within(target, file['path'])
        if not path.is_file() or file['bytes'] is not None and path.stat().st_size != file['bytes']:
            raise RuntimeError(f'Incomplete download: {path}')
        if file['sha256'] and digest(path) != file['sha256']:
            raise RuntimeError(f'Checksum mismatch: {path}; remove only this corrupt file and retry.')
    save(target / 'download_verified.json', dict(status='verified', **manifest))

def gcs(name, target):
    import google_crc32c
    prefix = f'checkpoints/{name}/'
    lock = target / 'download_manifest.json'
    if lock.exists():
        manifest = json.loads(lock.read_text())
        if manifest['prefix'] != prefix:
            raise ValueError(f'Checkpoint directory belongs to a different GCS prefix: {target}')
    else:
        items, token = [], None
        while True:
            query = dict(prefix=prefix)
            if token:
                query['pageToken'] = token
            with urllib.request.urlopen('https://storage.googleapis.com/storage/v1/b/openpi-assets/o?' + urllib.parse.urlencode(query), timeout=90) as response:
                page = json.load(response)
            items += [x for x in page.get('items', []) if not x['name'].endswith('/')]
            token = page.get('nextPageToken')
            if not token:
                break
        if not items:
            raise ValueError(f'Empty GCS checkpoint: {prefix}')
        manifest = dict(prefix=prefix, objects=items)
        save(lock, manifest)
    for item in manifest['objects']:
        path = within(target, item['name'][len(prefix):])
        def verified():
            if not path.is_file() or path.stat().st_size != int(item['size']):
                return False
            h = google_crc32c.Checksum()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(8*1024*1024), b''):
                    h.update(chunk)
            return base64.b64encode(h.digest()).decode() == item['crc32c']
        if not verified():
            url = 'https://storage.googleapis.com/openpi-assets/' + urllib.parse.quote(item['name'], safe='/')
            fetch(url + '?generation=' + item['generation'], path)
        if not verified():
            raise RuntimeError(f'GCS checksum mismatch: {path}')
        print('VERIFIED', path, flush=True)
    save(target / 'download_verified.json', dict(status='verified', **manifest))

def checkpoints(a):
    root = a.root / 'checkpoints'
    benches = a.benchmarks.split(',')
    if a.kind in ('base','both'):
        if a.model == 'pi05':
            gcs('pi05_base', root / 'cl_base/pi05_base')
        elif a.model == 'xvla':
            hf('2toINF/X-VLA-Pt', root / 'cl_base/xvla_pt')
        else:
            hf('nvidia/GR00T-N1.5-3B', root / 'cl_base/gr00t_n15_3b')
    if a.kind in ('eval','both'):
        if 'libero' in benches:
            if a.model == 'pi05':
                gcs('pi05_libero', root / 'pi05-libero')
            elif a.model == 'xvla':
                hf('2toINF/X-VLA-Libero', root / 'xvla-libero')
            else:
                hf('twanghcmut/GR00T-N1.5-LIBERO-4suite-combined', root / 'gr00t-libero-combined',
                   revision='aee73136d034537f68c4e34d2619d3d357c7a2b1')
        if 'robocasa' in benches and a.model in ('pi05','gr00t'):
            prefix = ('pi05_pretrain_human300/multitask_learning/75000/' if a.model == 'pi05'
                      else 'gr00t_n1-5/multitask_learning/checkpoint-120000/')
            hf('robocasa/robocasa365_checkpoints', root / 'robocasa', patterns=[prefix+'*'],
               revision='c484448aba1a9b60a04c9b0ca117241518ea69f3', ignore=[prefix+'train_state/*'])

def libero(a):
    from libero.libero import benchmark
    from huggingface_hub import HfApi
    repo = 'yifengzhu-hf/LIBERO-datasets'
    raw = a.root / 'datasets/libero-source'
    lock = raw / 'download_manifest.json'
    revision = json.loads(lock.read_text())['revision'] if lock.exists() else None
    info = HfApi().repo_info(repo, repo_type='dataset', revision=revision)
    by_name = {}
    for f in info.siblings:
        by_name.setdefault(Path(f.rfilename).name, []).append(f.rfilename)
    mapping = []
    for suite in ('libero_spatial','libero_object','libero_goal','libero_10'):
        tasks = benchmark.get_benchmark_dict()[suite](task_order_index=0)
        for i in range(tasks.n_tasks):
            name = tasks.get_task(i).name + '_demo.hdf5'
            candidates = by_name.get(name, [])
            preferred = [p for p in candidates if p.startswith(suite+'/')]
            if len(preferred or candidates) != 1:
                raise ValueError(f'Ambiguous/missing dataset file: {suite}/{name}: {candidates}')
            mapping.append((suite, name, (preferred or candidates)[0]))
    hf(repo, raw, patterns=[row[2] for row in mapping], revision=info.sha, repo_type='dataset')
    for suite, name, source in mapping:
        target = a.root / 'datasets/libero' / suite / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original = raw / source
        if target.exists() or target.is_symlink():
            if target.resolve() != original.resolve():
                raise ValueError(f'Refusing to replace existing dataset: {target}')
        else:
            target.symlink_to(original)
    save(a.root / 'datasets/libero/layout.json', dict(repo=repo, revision=info.sha, files=mapping))

def unpack(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    # Only regular files and directories; reject links and traversal in downloaded archives.
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                within(destination, member.filename)
                if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError('Archive contains a symbolic link')
            z.extractall(destination)
    else:
        with tarfile.open(archive) as tar:
            for member in tar:
                path = within(destination, member.name)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, path.open('wb') as out:
                        shutil.copyfileobj(src, out)
                else:
                    raise ValueError('Archive contains a link or special file')

def kitchen(a):
    from robocasa.scripts.download_kitchen_assets import DOWNLOAD_ASSET_REGISTRY
    for name, entry in DOWNLOAD_ASSET_REGISTRY.items():
        folder = Path(entry['folder'])
        if not folder.resolve().is_relative_to((a.root / 'upstreams/robocasa').resolve()):
            raise ValueError('RoboCasa imported from outside the selected deployment')
        marker = a.root / 'config' / f'kitchen-{name}.json'
        if marker.exists() and folder.exists():
            continue
        archive = a.root / 'downloads' / f'kitchen-{name}.zip'
        if not archive.exists():
            fetch(entry['url'], archive)
        unpack(archive, folder.parent)
        if not folder.exists():
            raise RuntimeError(f'Asset archive did not create {folder}')
        save(marker, dict(source=entry['url'], sha256=digest(archive), destination=str(folder)))

def robocasa(a):
    from robocasa.scripts.download_datasets import BOX_LINKS_DS, _get_direct_download_url
    from robocasa.utils.dataset_registry_utils import get_ds_meta
    base = a.root / 'datasets/robocasa/v1.0'
    for task in a.tasks.split(','):
        meta = get_ds_meta(task=task, source='mg' if a.source == 'mimicgen' else a.source, split=a.split)
        if not meta or not meta.get('path'):
            raise ValueError(f'No official dataset: {task}/{a.split}/{a.source}')
        destination = Path(meta['path'])
        rel = destination.relative_to(base)
        key = str(rel.parent / (rel.name + '.tar'))
        marker = destination / '.xpl-download.json'
        if marker.exists():
            continue
        if destination.exists():
            raise ValueError(f'Unverified dataset already exists: {destination}; use a fresh root or validate it manually.')
        if key not in BOX_LINKS_DS:
            raise ValueError(f'No official download URL for {key}')
        archive = a.root / 'downloads' / (task + '-' + a.split + '-' + a.source + '.tar')
        url = _get_direct_download_url(BOX_LINKS_DS[key], ext='tar')
        if not archive.exists():
            fetch(url, archive)
        staging = destination.parent / (destination.name + '.xpl-extract')
        unpack(archive, staging)
        extracted = staging / destination.name
        if not (extracted / 'meta/info.json').exists():
            raise RuntimeError(f'Missing LeRobot metadata after extraction: {destination}')
        extracted.replace(destination)
        save(marker, dict(source=url, sha256=digest(archive), task=task, split=a.split, kind=a.source))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('checkpoints','libero','robocasa','kitchen'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', choices=('pi05','xvla','gr00t'))
    p.add_argument('--benchmarks', default='libero')
    p.add_argument('--kind', choices=('base','eval','both'), default='both')
    p.add_argument('--tasks', default='OpenDrawer')
    p.add_argument('--split', default='pretrain', choices=('pretrain','target'))
    p.add_argument('--source', default='human', choices=('human','mimicgen'))
    a = p.parse_args()
    a.root = a.root.resolve()
    globals()[a.mode](a)

if __name__ == '__main__':
    main()

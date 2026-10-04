"""Machine configuration and portable artifact paths; no GPU/library imports."""
import os
from pathlib import Path


def configured_path(name, default=None):
    value = os.environ.get(name, default)
    if not value:
        raise RuntimeError(f"Set {name}; source labserver_env.sh or bcc_env.sh first")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute: {value}")
    # Do not resolve executable symlinks: that would bypass virtual environments.
    return path


CODE = Path(__file__).resolve().parent
REPO = configured_path('V1_REPO', str(CODE.parents[1]))
ROOT = configured_path('V1_ROOT', str(REPO.parent))
ASSETS = configured_path('V1_ASSETS')
RUN = configured_path('V1_RUN')
DATA = configured_path('V1_DATA')
BASE = configured_path('V1_BASE')
OPENPI = configured_path('V1_OPENPI')


def relative_path(path, root):
    """Serialize a path under root; reject traversal/symlink escape."""
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def under(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Expected a contained relative path: {relative}')
    path = Path(root) / relative
    relative_path(path, root)
    return path


def resolve_checkpoint(stream, value, task):
    """Read new relative paths or rebase legacy absolute records to this stream.

    Never use the legacy location, even when it still exists. A relocated run
    must not silently load another machine/run's checkpoint.
    """
    stream, value = Path(stream), Path(value)
    if value.is_absolute():
        marker = (stream.parent.name, stream.name, f'task{task:02d}', 'checkpoints')
        hits = [i for i in range(len(value.parts) - 3)
                if value.parts[i:i + 4] == marker]
        if len(hits) != 1:
            raise ValueError(f'Cannot identify stream/task in legacy checkpoint: {value}')
        value = Path(*value.parts[hits[0] + 2:])
    if value.parts[:2] != (f'task{task:02d}', 'checkpoints'):
        raise ValueError(f'Checkpoint does not belong to task {task}: {value}')
    path = under(stream, value)
    if not (path / 'params').is_dir() or not (path / 'train_state').is_dir():
        raise FileNotFoundError(f'Incomplete or missing checkpoint at relocated path: {path}')
    return path


def resolve_data_path(value):
    """Resolve Parquet paths against V1_DATA, including legacy manifests."""
    value = Path(value)
    if value.is_absolute():
        try:
            value = value.relative_to(DATA)
        except ValueError:
            legacy = os.environ.get('V1_LEGACY_DATA_ROOT')
            if legacy:
                value = value.relative_to(Path(legacy))
            else:
                # Original physical-intelligence/libero layout: data/chunk-NNN/*.parquet.
                parts = value.parts
                hits = [i for i in range(len(parts) - 2)
                        if parts[i] == 'data' and parts[i + 1].startswith('chunk-')]
                if len(hits) != 1 or value.suffix != '.parquet':
                    raise ValueError('Unknown legacy data layout; set V1_LEGACY_DATA_ROOT: ' + str(value))
                value = Path(*parts[hits[0]:])
    path = under(DATA, value)
    if not path.is_file():
        raise FileNotFoundError(f'Missing data under V1_DATA: {path}')
    return path

"""CPU-only deployment path check; imports no model or simulator."""
import json
import os
from pathlib import Path
from paths import REPO, DATA, BASE, OPENPI, RUN, configured_path


def main():
    required = {'repository': REPO, 'data metadata': DATA / 'meta/info.json',
                'base parameters': BASE / 'params', 'openpi training': OPENPI / 'scripts/train.py',
                'training Python': configured_path('V1_PI_PY'),
                'simulator Python': configured_path('V1_SIM_PY'),
                'LIBERO config': configured_path('V1_LIBERO_CONFIG')}
    for name in ('V1_PI_PATHS', 'V1_SIM_PATHS', 'V1_WS_PATHS'):
        for i, path in enumerate(os.environ.get(name, '').split(os.pathsep)):
            if path:
                required[f'{name}[{i}]'] = Path(path)
    missing = []
    for name, path in required.items():
        ok = path.exists()
        print(f'{"OK" if ok else "MISSING"} {name}: {path}')
        if not ok: missing.append(name)
    print(json.dumps({'run': str(RUN), 'tmp': os.environ.get('TMPDIR'),
                      'hf_home': os.environ.get('HF_HOME'), 'missing': missing}, indent=2))
    raise SystemExit(bool(missing))


if __name__ == '__main__':
    main()

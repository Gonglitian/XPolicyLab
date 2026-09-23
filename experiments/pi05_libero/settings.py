"""Portable paths and queue selection, set by scripts/deploy/run_experiments.sh."""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = json.loads(Path(os.environ['XPL_DEPLOY_CONFIG']).read_text())
ROOT = Path(CONFIG['root'])
ASSETS = ROOT
RUN = Path(os.environ['XPL_RUN_DIR']).resolve()
OPENPI = ROOT / 'upstreams/openpi'
PI_PY = ROOT / 'envs/pi05/bin/python'
SIM_PY = ROOT / 'envs/libero/bin/python'
ALL_SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
SUITES = os.environ.get('XPL_SUITES', ','.join(ALL_SUITES)).split(',')
METHODS = os.environ.get('XPL_METHODS', 'sf,er').split(',')
GPUS = os.environ.get('XPL_GPUS', '0,1,2,3').split(',')
PORT = int(os.environ.get('XPL_PORT_BASE', '6321'))
if not SUITES or len(set(SUITES)) != len(SUITES) or not set(SUITES) <= set(ALL_SUITES):
    raise ValueError('Invalid or duplicate suite selection')
if not METHODS or len(set(METHODS)) != len(METHODS) or not set(METHODS) <= {'sf', 'er'}:
    raise ValueError('Invalid or duplicate method selection')
if len(GPUS) != 4 or len(set(GPUS)) != 4 or not all(g.isdigit() for g in GPUS):
    raise ValueError('Exactly four distinct physical GPU indices are required')
if not 1024 <= PORT <= 65532:
    raise ValueError('Port base must be in [1024, 65532]')

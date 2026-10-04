"""Run inside the assigned Slurm GPU environment: envrun.py sim|pi -- CMD..."""
import os
import sys
from lane import environment, PI_PY, SIM_PY
from slurm_runtime import require_slurm
require_slurm()
kind = sys.argv[1]
if kind not in ('sim', 'pi') or sys.argv[2] != '--':
    raise SystemExit('Usage: envrun.py sim|pi -- CMD...')
cmd = sys.argv[3:]
if cmd[0] == 'PY': cmd[0] = str(SIM_PY if kind == 'sim' else PI_PY)
os.execvpe(cmd[0], cmd, environment(sim=kind == 'sim'))

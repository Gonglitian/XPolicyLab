"""Run a command inside the lane environment: envrun.py GPU sim|pi -- CMD..."""
import os, sys
from lane import environment, PI_PY, SIM_PY
gpu, kind = sys.argv[1], sys.argv[2]
assert sys.argv[3] == '--'
cmd = sys.argv[4:]
cmd[0] = str(SIM_PY if kind.startswith('sim') else PI_PY) if cmd[0] == 'PY' else cmd[0]
env = environment(gpu, sim=kind.startswith('sim'))
if kind == 'sim-usersite':
    env.pop('PYTHONNOUSERSITE', None)  # pyarrow for the sim env lives in ~/.local only
os.execvpe(cmd[0], cmd, env)

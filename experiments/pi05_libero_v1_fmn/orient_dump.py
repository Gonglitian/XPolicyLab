"""(openpi env) Dump the first frame of the first 10 episodes of task 0 per suite for the orientation check."""
import io, json
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from common import RUN, SUITES
manifest = json.loads((RUN / 'manifest.json').read_text())
out = {}
for suite in SUITES:
    frames = []
    for path in manifest[suite][0]['episode_files'][:10]:
        t = pq.read_table(path, columns=['image'])
        frames.append(np.asarray(Image.open(io.BytesIO(t.column('image')[0].as_py()['bytes'])).convert('RGB')))
    out[suite] = np.stack(frames)
np.savez_compressed(RUN / 'orient_frames.npz', **out)
print('DUMPED', {k: v.shape for k, v in out.items()})

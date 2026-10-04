"""(simulator env) Official LIBERO task order, language and the 50 fixed initial states per task."""
import hashlib, json, sys
import numpy as np
from libero.libero import benchmark
out = {}
for suite_name in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
    suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
    tasks = []
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        initial = np.asarray(suite.get_task_init_states(i))
        assert len(initial) >= 50, (suite_name, i, initial.shape)
        tasks.append(dict(task_id=i, task_name=task.name, instruction=str(task.language).strip(),
                          initial_states_sha256=hashlib.sha256(initial[:50].tobytes()).hexdigest()))
    assert len(tasks) == 10
    out[suite_name] = tasks
open(sys.argv[1], 'w').write(json.dumps(out, indent=2) + '\n')
print('TASKS_WRITTEN', {s: len(t) for s, t in out.items()})

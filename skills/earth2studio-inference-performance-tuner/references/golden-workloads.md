# Golden inference workloads

Use smoke shapes only for execution validation and representative shapes for conclusions. Prefer cached/local inputs and record cache state.

| Workload | Repository seed | Coverage |
|---|---|---|
| deterministic forecast | `examples/01_getting_started/01_deterministic_workflow.py` | fetch, mapping, forecast step, write |
| ensemble and IO | `examples/07_misc/03_io_performance.py` | perturbation, batching, memory, async output |
| generative downscaling | `examples/03_downscaling/01_corrdiff_inference.py` | sampler, sample throughput, kernels |
| evaluation recipe | `recipes/eval/main.py` | work items, regrid, output, resume, ranks |
| data assimilation | `examples/05_data_assimilation/01_stormcast_sda.py` | observations and assimilation |
| seasonal rollout | `examples/06_seasonal/02_dlesym_example.py` | long rollout and memory |

Validate the first four before expanding to assimilation and seasonal cases.

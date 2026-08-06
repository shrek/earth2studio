# Conditional compilation comparison

Use only when the model exposes a supported compile path. Hold source, weights, config, data/order, output, seed, precision, topology, horizon, and batch/member/sample counts invariant. Record compiled modules/methods, backend, mode, fullgraph, dynamic settings, cache state, and model-specific controls.

Warm every intended graph before steady timing and exclude compile and first-step costs from default benchmark samples. Measure those startup costs separately only when the user explicitly requests them. Run bounded non-benchmark graph-break/recompile diagnostics. Report source, reason/count, guards, cache warnings, backend failures, and eager fallbacks. Never infer a benefit from diagnostic wall time.

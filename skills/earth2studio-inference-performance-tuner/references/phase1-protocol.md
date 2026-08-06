# Phase-1 inference profiling protocol

## Configuration contract

Resolve and display baseline/candidate commands, workdir, revision/local changes, model/config, data identity/cache state, initialization times, forecast steps, variables, batch/member/sample sizes, seed, precision, compile settings, topology, source settings, IO/compression/chunking, checkpoint settings, whether startup measurement is explicitly enabled, budgets, correctness check, goal, and artifact output. Bind confirmation to the SHA-256 of `test-config.json`.

## Artifact contract

Store configuration at the root, raw samples under `baseline/`, traces under `traces/`, normalized evidence under `hta/`, compile diagnostics under `compile/`, and NCU artifacts under `ncu/`. Required ready-state artifacts are:

- `test-config.json`, `test-config.md`, `config-confirmation.json`;
- `correctness.json` and `baseline/summary.json` with warm end-to-end and steady-state results; startup results are optional;
- annotation-health JSON for every analyzed trace;
- `hta/pipeline.json` and `hta/pipeline.svg`;
- `hta/dominant-kernels.json`, `.csv`, `.md`, and `.svg`;
- `phase-source-map.json`, `source-analysis.json`, `findings.json`, and `report.md`;
- `ncu/decision.json` plus evidence or a skip reason.

## Evidence and report contract

Classify bottlenecks as input/data, preprocessing/coordinates/regrid, model compute, sampler, launch/compile, memory movement, output, checkpoint, or distributed. Link each finding to metrics, trace ranges/kernels, exact source, and one isolated experiment. Present measurements first, diagrams and kernel table second, interpretation third, and hypotheses last. Use `recommendation_only` status.

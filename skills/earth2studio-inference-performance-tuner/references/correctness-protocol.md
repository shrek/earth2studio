# Inference correctness protocol

Record the command, tolerances, seed policy, reference artifact, and result. Require identical coordinate keys/order/values, shapes, variables, lead times, output count, and storage schema; no new NaN/Inf values; deterministic values within declared tolerances; equivalent checkpoint/resume completion when in scope; fixed RNG reset behavior for seeded comparisons; and matching sampling schedule/count for generative comparisons.

Bitwise equality is not generally required across compiler or kernel implementations. For stochastic algorithms, use matched seeds when possible and declare field-error plus distribution summaries. Never weaken tolerances after seeing a candidate without reconfirming the test configuration.

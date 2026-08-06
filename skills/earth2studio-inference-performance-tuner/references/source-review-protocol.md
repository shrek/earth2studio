# Phase-to-source review protocol

Create `phase-source-map.json` with one record per canonical phase: status, trace range, file, symbol, line bounds, configuration controls, and evidence. Use `not_applicable` with a reason when absent.

Create `source-analysis.json` records with ID, phase, exact path/symbol/lines, measured evidence, code observation, mechanism, recommendation or `no_change_reason`, isolated experiment, correctness/performance checks, semantic risk, confidence, and `recommendation_only`. Review depth in proportion to critical-path contribution and link every finding to source-analysis IDs. Source inspection alone is insufficient.

# Evaluation

## Inputs and resource boundaries

The selector accepts local UTF-8 JSONL with exactly `record_id`, `group_id`, `context`, `goal`, `split`, and `access`. Use `goal_only` when downstream questions are withheld, or `question_conditioned` when the question is itself the selector goal. Use stable source-group IDs across records and keep each group within one split. Keep answers, Gold spans, and scoring labels outside this file.

Use material you are authorized to process. This repository downloads the pinned BGE-small and DeBERTa-small ONNX assets only. For KEEP-Bench, follow the [fixed dataset card](https://huggingface.co/datasets/uclanlp/KEEP-Bench/blob/91c37d5470f81ab7655b4893e7ba10536d5c952f/README.md). Its access prompt limits research use and prohibits using KEEP as training data for models evaluated on the benchmark. Keep KEEP inputs evaluation-only. For GACC-Bench, verify terms for the chosen sources before training or distribution. The [pinned GACC card](https://huggingface.co/datasets/Regimes/GACC-Bench/blob/3eb576c6984d3370fa3f3b1083bee7561588194f/README.md) lacks a dataset-wide license field.

For an authorized local KEEP subset, `python -m scripts.import_keep_eval --input local/test.jsonl --subset conditional --output build/keep-eval` produces question-conditioned Final inputs and a separate Reader question file. It omits answers and evidence annotations. Keep the source and resulting files local.

The selection CLI requires a local Wk8 state dictionary. Safe, Swap, and Guarded also require a CleanGate state dictionary and JSON with `temperatures`, `biases`, `keep_T`, `keep_bias`, `tau_select`, `tau_bind`, and `tau_bind_learned`. Supply frozen values fitted only on authorized calibration data. `tau_bind_learned` may be null. The example in `scripts/demo.py` uses fixture settings solely to exercise the methods. The published wrapper calls the formal selection, scoring, and rendering functions on newly extracted local caches. It is an entry point for new experiments.

Budget ratios are 5%, 10%, 15%, and 20%. The hard cap is the floor of the ratio times the source's `cl100k_base` token count. Both the serialized output and its flat Reader form are checked, including any labels and separators. A feasible timeout can yield a different extract across solver versions.

## Reader and statistics

Provide separate local question JSONL with `record_id`, `question_id`, and `question`. Seal selections before joining questions to goal-only exports. The payload tool deduplicates identical messages per generation setting and keeps a separate method-to-request map. Use `requirements-reader.txt` in a separate environment and the [reference Qwen revision](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28). The runner checks the FP16 deployment on two T4 devices, software, and model config. It records generated IDs and takes the primary 128-token result from the first 128 original IDs. This release tests payload construction on CPU. Full GPU inference requires compatible local weights and hardware.

```sh
python scripts/run_reader.py --payload build/reader/payload.json --model-dir models/reader --output build/predictions
python -m scripts.score qa examples/qa.jsonl
python -m scripts.score paired local/source_scores.jsonl --method Guarded --reference Wk8
```

Include every planned evaluation arm, including unresolved answers. The QA file uses a question-level `record_id`, `group_id`, `method`, `ratio`, `prompt`, `cap`, and the explicit labels shown in the synthetic fixture. Strict requires every requested answer component and required qualification. Scope-Verified requires both source scope and a correct target value. The lower bound counts confirmed events. The screening upper bound adds unresolved screening positives, and the unrestricted upper bound adds all unresolved arms. These are fixed-cohort bounds. Source completeness can be computed with `rc_gacc.scope_retention.scope_span_mask` from separate source annotations.

The paired source-score format uses `record_id`, `group_id`, `method`, `seed`, and `value`. Supply one budget and metric per call. The script averages seeds within record, computes paired differences within source group, and uses 2,000 group bootstrap draws at seed 42 for a pointwise percentile interval.

## Independent evaluation after retraining

1. Train from random initialization on original synthetic data or sources with verified training rights. Track source-group provenance and exclude KEEP-derived texts, annotations, teachers, caches and weights. Screen overlap against held-out sources before any fit.
2. Fit parameter-matched MLP0 and Wk8 on the same corpus and offline tensors at seeds 42, 137 and 3407. Use the paper's six-epoch node recipe. Freeze the node scorer and fit the Edge Head on licensed or original typed C/Q supervision with the 16-epoch recipe. Learn selection and binding settings from a disjoint calibration split from the authorized corpus.
3. Freeze parameters, thresholds, cost policies, baselines and evaluation endpoints before opening test labels. Use independent goal-only evaluation and evaluation-only KEEP tracks under the provider terms. Keep held-out synthetic C/Q and natural Q separate. Use the same information, Reader prompts, caps and budget accounting for matched comparisons.
4. Regenerate all model-dependent tables and figures from the newly sealed exports and scoring. Report source retention, typed pair integrity, Strict, Scope-Verified, unresolved intervals and source-group uncertainty separately. Compare matched baselines using identical cohorts. Report any failed or unavailable arms with the corresponding denominator.

This protocol requires a cleared training corpus, audited supervision and fresh runs. The code release establishes implementation and fixture behavior. Scientific performance claims require the new independent results.

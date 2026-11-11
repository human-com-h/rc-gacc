# RC-GACC

RC-GACC selects goal-relevant evidence under hard token budgets. Wk8 scores candidate evidence and goal coverage, the Edge Head scores typed relations, and the MILP selects a budgeted extract. Safe retains the Wk8 parent evidence, Swap can replace selected units under objective and coverage floors, and Guarded chooses Swap when it preserves a credited pair that Safe missed.

This is the reference implementation for *Preserving Typed Evidence Relations under Hard Context Budgets*. Release 0.1.0 verifies the implementation on synthetic inputs. The code release includes no paper results, benchmark instances, trained checkpoints, or fitted calibration values.

## Quick start

Use Python 3.12 and run the following from the repository root in a virtual environment. The first install selects the CPU wheel on Linux and Windows.

```sh
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.demo
python -m scripts.score qa examples/qa.jsonl
```

The demo runs Wk8, Safe, Swap, and Guarded using deterministic fixture parameters. Each output includes the original token budget and the billed extract length. On first use, the tokenizer obtains its vocabulary from its provider.

## Local corpus and evaluation

```sh
python -m scripts.fetch_encoders
python -m scripts.prepare --input examples/selector.jsonl --output build/cache
```

The feature extractor verifies the pinned ONNX assets, separates the selector view from answer annotations, and writes local Parquet caches. With weights and calibration fitted on authorized inputs, use the following entry point.

```sh
python -m scripts.select_context --index build/cache/index.jsonl --wk8 local/Wk8.pt --edge-head local/EdgeHead.pt --calibration local/calibration.json --method Guarded --ratio 0.10 --output build/guarded.jsonl
python -m scripts.reader_payload --extracts build/guarded.jsonl --questions examples/questions.jsonl --output build/reader
```

The workpoint implementation is in `rc_gacc/`. The [evaluation guide](docs/evaluation.md) documents local formats, fitted resource requirements, scoring and the proposed independent evaluation. The [maintenance guide](docs/maintenance.md) records versioning and release checks. Use [CITATION.cff](CITATION.cff) to cite the software. The MIT license covers project code and documentation. Models and datasets retain their upstream terms.

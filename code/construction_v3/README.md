# FinGraphQA Refactor v3

This package replaces the construction/evaluation layer with a cleaner pipeline:

1. `graph_builder.py`
   - Source graph: page/text/table/row/column/cell
   - Financial fact graph: metric/period/quantity/unit/scale/currency
   - Question-conditioned candidate operands
   - Stronger table-text bridge edges using table-neighborhood + lexical/metric/period overlap

2. `evidence_set_ranker.py`
   - Converts candidate facts into source-level rankings
   - Adds operation-aware source bonuses
   - Applies controlled table-text bridge expansion
   - Provides relaxed coverage helpers

3. `evaluate_graph_retrieval.py`
   - Reports strict and relaxed recall
   - Reports modality-level and formula-cue-level metrics
   - Saves strict/mixed/relaxed missed cases
   - Shows expansion diagnostics

## Install

From your activated virtual environment:

```bash
python -m pip install networkx matplotlib pandas numpy
```

## Apply

Copy all `.py` files into:

```bash
FinGraphQA/code/construction_update/
```

## Run

```bash
cd FinGraphQA/code/construction_update
python check_graph.py --split train --idx 0
python evaluate_graph_retrieval.py --split train --limit 200 --verbose_missed --save_missed
```

To compare without bridge expansion:

```bash
python evaluate_graph_retrieval.py --split train --limit 200 --disable_bridge_expansion
```

## Notes

Bridge expansion is intentionally stronger than the previous version because mixed table-text evidence was the main failure mode. Watch these metrics:

- `mixed StrictAll@5 / StrictAll@10`
- `mixed RelaxedAll@5 / RelaxedAll@10`
- `table_only StrictAll@5`
- `examples with expansion`

If table-only strict recall drops too much, run with `--disable_bridge_expansion` or raise the bridge threshold inside `evidence_set_ranker.py`.

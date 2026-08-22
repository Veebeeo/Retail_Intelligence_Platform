# scripts/

Thin wrappers kept so the commands in the original README keep working. The
implementations live in `src/retail_intel/`, where they are importable and
testable; these files only forward to them.

| Legacy command | Equivalent |
| --- | --- |
| `python -m scripts.ingest_data` | `python -m retail_intel.data.ingest` |
| `python -m scripts.ingest_features` | `python -m retail_intel.data.features` |
| `python -m scripts.train_models` | `python -m retail_intel.forecasting.train` |
| `python -m scripts.cluster_customers` | `python -m retail_intel.segmentation.pipeline` |

Prefer `make pipeline`, which runs the whole chain in dependency order.

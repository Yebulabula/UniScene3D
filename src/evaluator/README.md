`evaluator/scene_classification`, `evaluator/scene_retrieval`, and `evaluator/view_retrieval` are the runnable benchmark packages.

- `scene_classification/`: zero-shot and few-shot room-type classification Python modules.
- `scene_retrieval/`: text-to-scene retrieval Python modules.
- `view_retrieval/`: grounded view retrieval Python modules.
- `common/`: shared model and path helpers.
- `utils/`: benchmark-specific utilities.

`dataset/classification/` stores room-type classification assets.
`dataset/retrieval/` stores retrieval JSONL assets.
`dataset/refer/` stores refer JSONL assets used by pretraining.

Shell launchers for SpatialBench now live directly under:

- `scripts/scene_classification/`
- `scripts/scene_retrieval/`
- `scripts/view_retrieval/`

They share `scripts/spatial_bench_common.sh`.

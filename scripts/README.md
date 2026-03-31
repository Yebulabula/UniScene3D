Shell entry points are organized by task:

- `pretraining/`: pretraining launchers, including Slurm variants.
- `3dvqa/`: 3D VQA and reasoning fine-tuning launchers, including Slurm variants.
- `retrieval/`: retrieval fine-tuning launchers.
- `scene_classification/`: SpatialBench scene classification launchers.
- `scene_retrieval/`: SpatialBench scene retrieval launchers.
- `view_retrieval/`: SpatialBench view retrieval launchers.
- `spatial_bench_common.sh`: shared shell helper for retrieval/classification launchers.
- Benchmark Python modules live under `evaluator/scene_classification`, `evaluator/scene_retrieval`, and `evaluator/view_retrieval`.
- `job/`: captured Slurm stdout and stderr logs.

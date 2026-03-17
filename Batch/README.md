# Batch SHARP

This folder contains a separate high-throughput batch runner for SHARP.

## Usage

From the repo root:

```cmd
.\.venv\Scripts\python.exe Batch\batch_predict.py ^
  --input-dir .\Input ^
  --output-dir .\BatchOutput ^
  --checkpoint-path .\sharp_2572gikvuh.pt ^
  --focal-35mm-mm 35 ^
  --device cuda
```

## Behavior

- Walks the input directory recursively for supported image files.
- Uses one shared 35mm-equivalent focal length for the whole run.
- Writes a flat output folder of `.ply` files.
- Fails before starting if two inputs would generate the same output name.
- Keeps one GPU worker producing exact intermediate results while CPU workers run the same covariance decomposition and `.ply` serialization concurrently.
- Continues past per-image runtime failures and writes `batch_summary.json` with results.

## Performance knobs

- `--postprocess-workers` sets the total CPU worker budget used to derive exact finalize and save workers.
  By default, the runner derives a more save-friendly split of roughly two-thirds SVD/finalize and one-third save workers.
- `--svd-workers` explicitly sets exact CPU covariance/finalize workers.
- `--save-workers` explicitly sets PLY packing/write workers.
- `--write-workers` is retained as a backward-compatible alias for the legacy total-worker flag.
- `--profile` adds per-image timing and queue statistics to the summary JSON.
- `--hardware-profile` records lightweight CPU, RAM, and GPU utilization samples during the run.
- `--hardware-sample-interval` controls how often hardware samples are captured. Default: `0.5` seconds.

## Telemetry output

- `batch_summary.json` includes hardware rollups and, for shorter runs, inline hardware samples.
- `batch_summary.json` also includes separate `cpu_svd_seconds`, `cpu_finalize_seconds`, `ply_tensor_export_seconds`, `ply_vertex_fill_seconds`, `ply_metadata_pack_seconds`, `ply_pack_seconds`, and `ply_write_seconds` timing buckets.
- `batch_hardware_samples.jsonl` always contains the full hardware sample timeline.
- GPU telemetry uses `nvidia-smi` when available. CPU and RAM telemetry uses `psutil` when installed, and missing metrics are recorded as `null` rather than failing the batch run.

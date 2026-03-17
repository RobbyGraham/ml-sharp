"""High-throughput batch inference runner for SHARP."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import statistics
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency at runtime
    psutil = None

from sharp.cli.predict import (
    prepare_image_for_prediction,
    prepare_prediction_postprocess_inputs,
    predict_image_ndc,
)
from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    PreparedGaussians3D,
    finalize_prepared_gaussians_profiled,
    move_prepared_gaussians_to_cpu,
    save_ply_profiled,
)

LOGGER = logging.getLogger(__name__)

SUMMARY_FILENAME = "batch_summary.json"
HARDWARE_SAMPLES_FILENAME = "batch_hardware_samples.jsonl"
LOG_FILENAME = "batch_predict.log"
MAX_INLINE_HARDWARE_SAMPLES = 512
LOW_GPU_UTILIZATION_THRESHOLD = 30.0


@dataclass(frozen=True)
class ImageJob:
    """Describes one input/output mapping for batch inference."""

    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class DecodedImage:
    """In-memory representation of a decoded image."""

    job: ImageJob
    image: np.ndarray
    focal_length_px: float
    image_shape: tuple[int, int]
    decode_seconds: float
    decoded_at: float


@dataclass(frozen=True)
class PreparedCpuJob:
    """Payload prepared on GPU and ready for CPU finalize work."""

    job: ImageJob
    focal_length_px: float
    image_shape: tuple[int, int]
    prepared_gaussians: PreparedGaussians3D
    decode_seconds: float
    decode_queue_wait_seconds: float
    gpu_preprocess_seconds: float
    gpu_forward_seconds: float
    gpu_prepare_seconds: float
    gpu_to_cpu_transfer_seconds: float
    submitted_at: float


@dataclass(frozen=True)
class FinalizedCpuJob:
    """Payload finalized on CPU and ready for PLY serialization."""

    job: ImageJob
    focal_length_px: float
    image_shape: tuple[int, int]
    gaussians_cpu: Gaussians3D
    decode_seconds: float
    decode_queue_wait_seconds: float
    gpu_preprocess_seconds: float
    gpu_forward_seconds: float
    gpu_prepare_seconds: float
    gpu_to_cpu_transfer_seconds: float
    postprocess_queue_wait_seconds: float
    cpu_svd_seconds: float
    cpu_finalize_seconds: float
    cpu_postprocess_seconds: float
    save_submitted_at: float


@dataclass(frozen=True)
class CompletedJob:
    """Result for one completed image pipeline."""

    job: ImageJob
    decode_seconds: float
    decode_queue_wait_seconds: float
    gpu_preprocess_seconds: float
    gpu_forward_seconds: float
    gpu_prepare_seconds: float
    gpu_to_cpu_transfer_seconds: float
    postprocess_queue_wait_seconds: float
    save_queue_wait_seconds: float
    cpu_svd_seconds: float
    cpu_finalize_seconds: float
    cpu_postprocess_seconds: float
    ply_tensor_export_seconds: float
    ply_vertex_fill_seconds: float
    ply_metadata_pack_seconds: float
    ply_pack_seconds: float
    ply_write_seconds: float
    save_seconds: float
    completed_at: float


class PipelineStageError(RuntimeError):
    """Wraps an exception with pipeline stage metadata."""

    def __init__(self, stage: str, job: ImageJob, error: Exception):
        super().__init__(f"{stage} failed for {job.input_path}: {error}")
        self.stage = stage
        self.job = job
        self.error = error


@dataclass
class PipelineCounters:
    """Thread-safe counters for pipeline backlog and progress telemetry."""

    pending_decode_tasks: int = 0
    pending_svd_tasks: int = 0
    pending_save_tasks: int = 0
    completed_images: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def set_pending_decode_tasks(self, count: int) -> None:
        with self._lock:
            self.pending_decode_tasks = count

    def set_pending_svd_tasks(self, count: int) -> None:
        with self._lock:
            self.pending_svd_tasks = count

    def set_pending_save_tasks(self, count: int) -> None:
        with self._lock:
            self.pending_save_tasks = count

    def increment_completed_images(self) -> None:
        with self._lock:
            self.completed_images += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "pending_decode_tasks": self.pending_decode_tasks,
                "pending_postprocess_tasks": self.pending_svd_tasks + self.pending_save_tasks,
                "completed_images": self.completed_images,
            }


@dataclass(frozen=True)
class HardwareSample:
    """One point-in-time system utilization sample."""

    timestamp: str
    seconds_since_start: float
    gpu_util_percent: float | None
    gpu_memory_used_mb: float | None
    gpu_memory_total_mb: float | None
    gpu_memory_util_percent: float | None
    gpu_power_watts: float | None
    gpu_temperature_c: float | None
    cpu_util_percent: float | None
    ram_used_mb: float | None
    ram_total_mb: float | None
    ram_util_percent: float | None
    pending_decode_tasks: int
    pending_postprocess_tasks: int
    completed_images: int


class HardwareSampler:
    """Background sampler for lightweight system telemetry."""

    def __init__(
        self,
        *,
        sample_interval_seconds: float,
        counters: PipelineCounters,
        device: torch.device,
        started_at_perf: float,
    ) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self.counters = counters
        self.device = device
        self.started_at_perf = started_at_perf
        self.samples: list[HardwareSample] = []
        self._samples_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvidia_smi_path = shutil.which("nvidia-smi")
        self._gpu_query_failed = False

    def start(self) -> None:
        if self._thread is not None:
            return
        if psutil is not None:
            psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, name="batch-hardware-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def collected_samples(self) -> list[dict[str, Any]]:
        with self._samples_lock:
            return [sample.__dict__.copy() for sample in self.samples]

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._record_sample()
            if self._stop_event.wait(self.sample_interval_seconds):
                break

    def _record_sample(self) -> None:
        counters = self.counters.snapshot()
        gpu_metrics = self._collect_gpu_metrics()
        cpu_metrics = self._collect_cpu_and_ram_metrics()
        sample = HardwareSample(
            timestamp=iso_timestamp(),
            seconds_since_start=max(0.0, time.perf_counter() - self.started_at_perf),
            gpu_util_percent=gpu_metrics["gpu_util_percent"],
            gpu_memory_used_mb=gpu_metrics["gpu_memory_used_mb"],
            gpu_memory_total_mb=gpu_metrics["gpu_memory_total_mb"],
            gpu_memory_util_percent=gpu_metrics["gpu_memory_util_percent"],
            gpu_power_watts=gpu_metrics["gpu_power_watts"],
            gpu_temperature_c=gpu_metrics["gpu_temperature_c"],
            cpu_util_percent=cpu_metrics["cpu_util_percent"],
            ram_used_mb=cpu_metrics["ram_used_mb"],
            ram_total_mb=cpu_metrics["ram_total_mb"],
            ram_util_percent=cpu_metrics["ram_util_percent"],
            pending_decode_tasks=counters["pending_decode_tasks"],
            pending_postprocess_tasks=counters["pending_postprocess_tasks"],
            completed_images=counters["completed_images"],
        )
        with self._samples_lock:
            self.samples.append(sample)

    def _collect_cpu_and_ram_metrics(self) -> dict[str, float | None]:
        if psutil is None:
            return {
                "cpu_util_percent": None,
                "ram_used_mb": None,
                "ram_total_mb": None,
                "ram_util_percent": None,
            }

        virtual_memory = psutil.virtual_memory()
        return {
            "cpu_util_percent": float(psutil.cpu_percent(interval=None)),
            "ram_used_mb": float(virtual_memory.used) / (1024 * 1024),
            "ram_total_mb": float(virtual_memory.total) / (1024 * 1024),
            "ram_util_percent": float(virtual_memory.percent),
        }

    def _collect_gpu_metrics(self) -> dict[str, float | None]:
        empty_metrics = {
            "gpu_util_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_memory_util_percent": None,
            "gpu_power_watts": None,
            "gpu_temperature_c": None,
        }
        if self.device.type != "cuda" or self._nvidia_smi_path is None or self._gpu_query_failed:
            return empty_metrics

        gpu_index = self.device.index if self.device.index is not None else 0
        command = [
            self._nvidia_smi_path,
            f"--id={gpu_index}",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=True,
                text=True,
                timeout=max(2.0, min(5.0, self.sample_interval_seconds * 2.0)),
            )
        except subprocess.TimeoutExpired:
            return empty_metrics
        except (OSError, subprocess.CalledProcessError):
            self._gpu_query_failed = True
            return empty_metrics

        line = result.stdout.strip().splitlines()
        if not line:
            self._gpu_query_failed = True
            return empty_metrics

        values = [value.strip() for value in line[0].split(",")]
        if len(values) != 5:
            self._gpu_query_failed = True
            return empty_metrics

        memory_used = safe_float(values[1])
        memory_total = safe_float(values[2])
        if memory_used is not None and memory_total not in (None, 0.0):
            memory_util = (memory_used / memory_total) * 100.0
        else:
            memory_util = None

        return {
            "gpu_util_percent": safe_float(values[0]),
            "gpu_memory_used_mb": memory_used,
            "gpu_memory_total_mb": memory_total,
            "gpu_memory_util_percent": memory_util,
            "gpu_power_watts": safe_float(values[3]),
            "gpu_temperature_c": safe_float(values[4]),
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SHARP on a directory of images with a shared focal length.",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of input images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where .ply outputs and run metadata will be written.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to a local SHARP checkpoint (.pt).",
    )
    parser.add_argument(
        "--focal-35mm-mm",
        type=float,
        required=True,
        help="Shared 35mm-equivalent focal length in millimeters.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on. ['cuda', 'cpu', 'mps', 'default']",
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=None,
        help="Number of worker threads for image decode and preprocessing.",
    )
    parser.add_argument(
        "--postprocess-workers",
        "--write-workers",
        dest="postprocess_workers",
        type=int,
        default=None,
        help="Total worker budget used to derive exact CPU finalize and save workers.",
    )
    parser.add_argument(
        "--svd-workers",
        type=int,
        default=None,
        help="Number of worker threads for exact CPU covariance decomposition/finalize work.",
    )
    parser.add_argument(
        "--save-workers",
        type=int,
        default=None,
        help="Number of worker threads for CPU PLY packing and file writes.",
    )
    parser.add_argument(
        "--queue-depth",
        type=int,
        default=None,
        help="Max decoded images kept ready ahead of the GPU worker.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Include per-image timing and queue profiling in batch_summary.json.",
    )
    parser.add_argument(
        "--hardware-profile",
        action="store_true",
        help="Record lightweight CPU, RAM, and GPU utilization samples during the run.",
    )
    parser.add_argument(
        "--hardware-sample-interval",
        type=float,
        default=0.5,
        help="Seconds between lightweight hardware utilization samples.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    if args.focal_35mm_mm <= 0:
        parser.error("--focal-35mm-mm must be positive.")

    if args.decode_workers is None:
        args.decode_workers = default_decode_workers()
    elif args.decode_workers <= 0:
        parser.error("--decode-workers must be positive.")

    if args.postprocess_workers is None:
        args.postprocess_workers = default_postprocess_workers()
    elif args.postprocess_workers <= 0:
        parser.error("--postprocess-workers must be positive.")

    if args.svd_workers is not None and args.svd_workers <= 0:
        parser.error("--svd-workers must be positive.")

    if args.save_workers is not None and args.save_workers <= 0:
        parser.error("--save-workers must be positive.")

    if args.queue_depth is None:
        args.queue_depth = max(2, args.decode_workers * 2)
    elif args.queue_depth <= 0:
        parser.error("--queue-depth must be positive.")

    if args.hardware_sample_interval <= 0:
        parser.error("--hardware-sample-interval must be positive.")

    if args.hardware_profile:
        args.profile = True

    args.svd_workers, args.save_workers = resolve_cpu_stage_workers(
        args.postprocess_workers,
        args.svd_workers,
        args.save_workers,
    )
    args.postprocess_workers = args.svd_workers + args.save_workers

    return args


def default_decode_workers() -> int:
    """Return the default number of decode workers."""
    cpu_count = os.cpu_count() or 8
    return min(16, max(4, cpu_count // 2))


def default_postprocess_workers() -> int:
    """Return the default number of exact CPU postprocess workers."""
    cpu_count = os.cpu_count() or 8
    return min(8, max(2, cpu_count // 4))


def derive_cpu_stage_workers(total_workers: int) -> tuple[int, int]:
    """Derive SVD/save worker counts from a shared CPU worker budget."""
    if total_workers <= 2:
        return 1, 1

    save_workers = max(2, (total_workers + 2) // 3)
    save_workers = min(save_workers, total_workers - 1)
    svd_workers = total_workers - save_workers
    return svd_workers, save_workers


def resolve_cpu_stage_workers(
    total_workers: int,
    svd_workers: int | None,
    save_workers: int | None,
) -> tuple[int, int]:
    """Resolve CPU worker counts with explicit overrides when provided."""
    if svd_workers is None and save_workers is None:
        return derive_cpu_stage_workers(total_workers)

    if svd_workers is None:
        svd_workers = max(1, total_workers - save_workers)  # type: ignore[operator]
    if save_workers is None:
        save_workers = max(1, total_workers - svd_workers)
    return svd_workers, save_workers


def default_svd_workers() -> int:
    """Return the default number of SVD/finalize workers."""
    return derive_cpu_stage_workers(default_postprocess_workers())[0]


def default_save_workers() -> int:
    """Return the default number of save workers."""
    return derive_cpu_stage_workers(default_postprocess_workers())[1]


def default_write_workers() -> int:
    """Backward-compatible alias for the legacy worker name."""
    return default_save_workers()


def discover_image_paths(input_dir: Path) -> list[Path]:
    """Find supported image files recursively under input_dir."""
    supported_extensions = {ext.lower() for ext in io.get_supported_image_extensions()}
    image_paths = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_extensions
    ]
    image_paths.sort(key=lambda path: str(path.relative_to(input_dir)).lower())
    return image_paths


def build_image_jobs(input_dir: Path, output_dir: Path) -> list[ImageJob]:
    """Create input/output jobs and validate the flat output mapping."""
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    image_paths = discover_image_paths(input_dir)
    if not image_paths:
        raise ValueError(f"No supported images found under {input_dir}")

    jobs: list[ImageJob] = []
    stems_to_path: dict[str, Path] = {}

    for image_path in image_paths:
        stem_key = image_path.stem.lower()
        if stem_key in stems_to_path:
            first_path = stems_to_path[stem_key]
            raise ValueError(
                "Duplicate output filename detected for flat output mode: "
                f"{first_path} and {image_path} would both map to {image_path.stem}.ply"
            )
        stems_to_path[stem_key] = image_path
        jobs.append(ImageJob(input_path=image_path, output_path=output_dir / f"{image_path.stem}.ply"))

    return jobs


def resolve_device(device_name: str) -> torch.device:
    """Resolve and validate the requested torch device."""
    if device_name == "default":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "mps") and torch.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    if device.type == "mps" and not (hasattr(torch, "mps") and torch.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available.")
    return device


def synchronize_device(device: torch.device) -> None:
    """Synchronize the current accelerator device, if any."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def load_predictor(checkpoint_path: Path, device: torch.device):
    """Load the SHARP model checkpoint onto the requested device."""
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")

    LOGGER.info("Loading checkpoint from %s", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    return predictor


def decode_image(job: ImageJob, focal_length_35mm_mm: float) -> DecodedImage:
    """Decode one image and compute the per-image focal length in pixels."""
    start_time = time.perf_counter()
    image, _, focal_length_px = io.load_rgb(
        job.input_path,
        focal_length_35mm_mm=focal_length_35mm_mm,
    )
    height, width = image.shape[:2]
    return DecodedImage(
        job=job,
        image=image,
        focal_length_px=focal_length_px,
        image_shape=(height, width),
        decode_seconds=time.perf_counter() - start_time,
        decoded_at=time.perf_counter(),
    )


def run_gpu_stage(
    predictor,
    decoded_image: DecodedImage,
    device: torch.device,
) -> PreparedCpuJob:
    """Run the exact GPU stages and hand off a CPU-ready payload."""
    decode_queue_wait_seconds = max(0.0, time.perf_counter() - decoded_image.decoded_at)

    if device.type == "cuda":
        preprocess_start = torch.cuda.Event(enable_timing=True)
        preprocess_end = torch.cuda.Event(enable_timing=True)
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        prepare_start = torch.cuda.Event(enable_timing=True)
        prepare_end = torch.cuda.Event(enable_timing=True)

        preprocess_start.record()
        prepared_inputs = prepare_image_for_prediction(
            decoded_image.image,
            decoded_image.focal_length_px,
            device,
        )
        preprocess_end.record()

        forward_start.record()
        gaussians_ndc = predict_image_ndc(predictor, prepared_inputs)
        forward_end.record()

        prepare_start.record()
        prepared_gaussians = prepare_prediction_postprocess_inputs(gaussians_ndc, prepared_inputs)
        prepare_end.record()

        transfer_start = time.perf_counter()
        prepared_gaussians_cpu = move_prepared_gaussians_to_cpu(
            prepared_gaussians,
            output_dtype=torch.float32,
        )
        torch.cuda.synchronize(device)
        gpu_to_cpu_transfer_seconds = time.perf_counter() - transfer_start

        gpu_preprocess_seconds = preprocess_start.elapsed_time(preprocess_end) / 1000.0
        gpu_forward_seconds = forward_start.elapsed_time(forward_end) / 1000.0
        gpu_prepare_seconds = prepare_start.elapsed_time(prepare_end) / 1000.0
    else:
        stage_start = time.perf_counter()
        prepared_inputs = prepare_image_for_prediction(
            decoded_image.image,
            decoded_image.focal_length_px,
            device,
        )
        synchronize_device(device)
        gpu_preprocess_seconds = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        gaussians_ndc = predict_image_ndc(predictor, prepared_inputs)
        synchronize_device(device)
        gpu_forward_seconds = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        prepared_gaussians = prepare_prediction_postprocess_inputs(gaussians_ndc, prepared_inputs)
        synchronize_device(device)
        gpu_prepare_seconds = time.perf_counter() - stage_start

        transfer_start = time.perf_counter()
        prepared_gaussians_cpu = move_prepared_gaussians_to_cpu(
            prepared_gaussians,
            output_dtype=torch.float32,
        )
        gpu_to_cpu_transfer_seconds = time.perf_counter() - transfer_start

    return PreparedCpuJob(
        job=decoded_image.job,
        focal_length_px=decoded_image.focal_length_px,
        image_shape=decoded_image.image_shape,
        prepared_gaussians=prepared_gaussians_cpu,
        decode_seconds=decoded_image.decode_seconds,
        decode_queue_wait_seconds=decode_queue_wait_seconds,
        gpu_preprocess_seconds=gpu_preprocess_seconds,
        gpu_forward_seconds=gpu_forward_seconds,
        gpu_prepare_seconds=gpu_prepare_seconds,
        gpu_to_cpu_transfer_seconds=gpu_to_cpu_transfer_seconds,
        submitted_at=time.perf_counter(),
    )


def finalize_cpu_job(prepared_job: PreparedCpuJob) -> FinalizedCpuJob:
    """Run exact CPU finalize work and prepare a save job."""
    worker_start = time.perf_counter()
    postprocess_queue_wait_seconds = max(0.0, worker_start - prepared_job.submitted_at)

    try:
        finalize_result = finalize_prepared_gaussians_profiled(
            prepared_job.prepared_gaussians,
            output_device=torch.device("cpu"),
            output_dtype=torch.float32,
        )
    except Exception as exc:  # pragma: no cover - exercised via error handling
        raise PipelineStageError("postprocess", prepared_job.job, exc) from exc

    cpu_finalize_seconds = finalize_result.cpu_finalize_seconds
    cpu_postprocess_seconds = finalize_result.cpu_svd_seconds + cpu_finalize_seconds

    return FinalizedCpuJob(
        job=prepared_job.job,
        focal_length_px=prepared_job.focal_length_px,
        image_shape=prepared_job.image_shape,
        gaussians_cpu=finalize_result.gaussians,
        decode_seconds=prepared_job.decode_seconds,
        decode_queue_wait_seconds=prepared_job.decode_queue_wait_seconds,
        gpu_preprocess_seconds=prepared_job.gpu_preprocess_seconds,
        gpu_forward_seconds=prepared_job.gpu_forward_seconds,
        gpu_prepare_seconds=prepared_job.gpu_prepare_seconds,
        gpu_to_cpu_transfer_seconds=prepared_job.gpu_to_cpu_transfer_seconds,
        postprocess_queue_wait_seconds=postprocess_queue_wait_seconds,
        cpu_svd_seconds=finalize_result.cpu_svd_seconds,
        cpu_finalize_seconds=cpu_finalize_seconds,
        cpu_postprocess_seconds=cpu_postprocess_seconds,
        save_submitted_at=time.perf_counter(),
    )


def save_finalized_job(finalized_job: FinalizedCpuJob) -> CompletedJob:
    """Pack and write the final PLY from CPU-resident Gaussians."""
    worker_start = time.perf_counter()
    save_queue_wait_seconds = max(0.0, worker_start - finalized_job.save_submitted_at)

    try:
        save_result = save_ply_profiled(
            finalized_job.gaussians_cpu,
            finalized_job.focal_length_px,
            finalized_job.image_shape,
            finalized_job.job.output_path,
        )
    except Exception as exc:  # pragma: no cover - exercised via error handling
        raise PipelineStageError("save", finalized_job.job, exc) from exc

    save_seconds = save_result.ply_pack_seconds + save_result.ply_write_seconds
    return CompletedJob(
        job=finalized_job.job,
        decode_seconds=finalized_job.decode_seconds,
        decode_queue_wait_seconds=finalized_job.decode_queue_wait_seconds,
        gpu_preprocess_seconds=finalized_job.gpu_preprocess_seconds,
        gpu_forward_seconds=finalized_job.gpu_forward_seconds,
        gpu_prepare_seconds=finalized_job.gpu_prepare_seconds,
        gpu_to_cpu_transfer_seconds=finalized_job.gpu_to_cpu_transfer_seconds,
        postprocess_queue_wait_seconds=finalized_job.postprocess_queue_wait_seconds,
        save_queue_wait_seconds=save_queue_wait_seconds,
        cpu_svd_seconds=finalized_job.cpu_svd_seconds,
        cpu_finalize_seconds=finalized_job.cpu_finalize_seconds,
        cpu_postprocess_seconds=finalized_job.cpu_postprocess_seconds,
        ply_tensor_export_seconds=save_result.ply_tensor_export_seconds,
        ply_vertex_fill_seconds=save_result.ply_vertex_fill_seconds,
        ply_metadata_pack_seconds=save_result.ply_metadata_pack_seconds,
        ply_pack_seconds=save_result.ply_pack_seconds,
        ply_write_seconds=save_result.ply_write_seconds,
        save_seconds=save_seconds,
        completed_at=time.perf_counter(),
    )


def iso_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def compute_throughput_after_warmup(completion_times: list[float]) -> float:
    """Estimate steady-state throughput after the first completed image."""
    if len(completion_times) < 2:
        return 0.0

    steady_state_seconds = completion_times[-1] - completion_times[0]
    if steady_state_seconds <= 0:
        return 0.0
    return (len(completion_times) - 1) / steady_state_seconds


def safe_float(value: str) -> float | None:
    """Parse a float-like value from telemetry output."""
    text = value.strip()
    if not text or text.lower() in {"n/a", "[not supported]", "not supported"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def percentile(values: list[float], pct: float) -> float | None:
    """Return a simple percentile for a non-empty numeric list."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_hardware_samples(
    hardware_samples: list[dict[str, Any]],
    sample_interval_seconds: float,
) -> dict[str, float | None]:
    """Aggregate time-series telemetry into summary statistics."""
    gpu_util_values = [
        sample["gpu_util_percent"]
        for sample in hardware_samples
        if sample["gpu_util_percent"] is not None
    ]
    gpu_memory_used_values = [
        sample["gpu_memory_used_mb"]
        for sample in hardware_samples
        if sample["gpu_memory_used_mb"] is not None
    ]
    cpu_util_values = [
        sample["cpu_util_percent"]
        for sample in hardware_samples
        if sample["cpu_util_percent"] is not None
    ]
    ram_util_values = [
        sample["ram_util_percent"]
        for sample in hardware_samples
        if sample["ram_util_percent"] is not None
    ]
    low_gpu_with_backlog_samples = sum(
        1
        for sample in hardware_samples
        if sample["gpu_util_percent"] is not None
        and sample["gpu_util_percent"] < LOW_GPU_UTILIZATION_THRESHOLD
        and sample["pending_postprocess_tasks"] > 0
    )
    return {
        "gpu_util_avg_percent": statistics.fmean(gpu_util_values) if gpu_util_values else None,
        "gpu_util_p95_percent": percentile(gpu_util_values, 0.95),
        "gpu_util_max_percent": max(gpu_util_values) if gpu_util_values else None,
        "gpu_memory_used_avg_mb": (
            statistics.fmean(gpu_memory_used_values) if gpu_memory_used_values else None
        ),
        "gpu_memory_used_max_mb": max(gpu_memory_used_values) if gpu_memory_used_values else None,
        "cpu_util_avg_percent": statistics.fmean(cpu_util_values) if cpu_util_values else None,
        "cpu_util_p95_percent": percentile(cpu_util_values, 0.95),
        "ram_util_avg_percent": statistics.fmean(ram_util_values) if ram_util_values else None,
        "ram_util_max_percent": max(ram_util_values) if ram_util_values else None,
        "seconds_gpu_below_threshold_with_postprocess_backlog": (
            low_gpu_with_backlog_samples * sample_interval_seconds
        ),
    }


def write_hardware_samples(output_dir: Path, hardware_samples: list[dict[str, Any]]) -> Path:
    """Write hardware samples as newline-delimited JSON."""
    samples_path = output_dir / HARDWARE_SAMPLES_FILENAME
    lines = [json.dumps(sample, sort_keys=True) for sample in hardware_samples]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    samples_path.write_text(payload, encoding="utf-8")
    return samples_path


def write_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Write the batch summary JSON to disk."""
    summary_path = output_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Run the batch inference pipeline."""
    started_at = iso_timestamp()
    wall_start = time.perf_counter()

    jobs = build_image_jobs(args.input_dir, args.output_dir)
    device = resolve_device(args.device)
    predictor = load_predictor(args.checkpoint_path, device)

    LOGGER.info("Discovered %d images under %s", len(jobs), args.input_dir)
    LOGGER.info("Using device %s", device)
    LOGGER.info(
        "Decode workers=%d, svd workers=%d, save workers=%d, queue depth=%d",
        args.decode_workers,
        args.svd_workers,
        args.save_workers,
        args.queue_depth,
    )

    counters = PipelineCounters()
    failures: list[dict[str, str]] = []
    success_outputs: list[str] = []
    completion_times: list[float] = []
    profile_rows: list[dict[str, Any]] = []
    timings = {
        "decode_seconds": 0.0,
        "decode_queue_wait_seconds": 0.0,
        "gpu_preprocess_seconds": 0.0,
        "gpu_forward_seconds": 0.0,
        "gpu_prepare_seconds": 0.0,
        "gpu_to_cpu_transfer_seconds": 0.0,
        "cpu_svd_seconds": 0.0,
        "cpu_finalize_seconds": 0.0,
        "cpu_postprocess_seconds": 0.0,
        "postprocess_queue_wait_seconds": 0.0,
        "save_queue_wait_seconds": 0.0,
        "ply_tensor_export_seconds": 0.0,
        "ply_vertex_fill_seconds": 0.0,
        "ply_metadata_pack_seconds": 0.0,
        "ply_pack_seconds": 0.0,
        "ply_write_seconds": 0.0,
        "save_seconds": 0.0,
        "inference_seconds": 0.0,
        "wall_seconds": 0.0,
    }
    queue_stats = {
        "max_pending_decode_tasks": 0,
        "max_pending_svd_tasks": 0,
        "max_pending_save_tasks": 0,
        "max_pending_postprocess_tasks": 0,
        "max_decode_queue_wait_seconds": 0.0,
        "max_postprocess_queue_wait_seconds": 0.0,
        "max_save_queue_wait_seconds": 0.0,
    }
    hardware_sampler: HardwareSampler | None = None
    if args.hardware_profile:
        hardware_sampler = HardwareSampler(
            sample_interval_seconds=args.hardware_sample_interval,
            counters=counters,
            device=device,
            started_at_perf=wall_start,
        )
        hardware_sampler.start()

    pending_svd: dict[Future[FinalizedCpuJob], PreparedCpuJob] = {}
    pending_save: dict[Future[CompletedJob], FinalizedCpuJob] = {}
    max_pending_svd = max(args.svd_workers * 2, 4)
    max_pending_save = max(args.save_workers * 2, 4)

    try:
        with ThreadPoolExecutor(max_workers=args.decode_workers) as decode_executor, ThreadPoolExecutor(
            max_workers=args.svd_workers
        ) as svd_executor, ThreadPoolExecutor(max_workers=args.save_workers) as save_executor:
            job_iterator = iter(jobs)
            decode_futures: dict[Future[DecodedImage], ImageJob] = {}

            def update_postprocess_queue_stats() -> None:
                total_pending = len(pending_svd) + len(pending_save)
                queue_stats["max_pending_postprocess_tasks"] = max(
                    queue_stats["max_pending_postprocess_tasks"],
                    total_pending,
                )

            def submit_decode_jobs() -> None:
                while len(decode_futures) < args.queue_depth:
                    try:
                        job = next(job_iterator)
                    except StopIteration:
                        break
                    future = decode_executor.submit(decode_image, job, args.focal_35mm_mm)
                    decode_futures[future] = job
                    counters.set_pending_decode_tasks(len(decode_futures))
                    queue_stats["max_pending_decode_tasks"] = max(
                        queue_stats["max_pending_decode_tasks"],
                        len(decode_futures),
                    )

            def collect_save_futures(*, block: bool) -> None:
                if not pending_save:
                    return

                done, _ = wait(
                    pending_save.keys(),
                    timeout=None if block else 0,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    finalized_job = pending_save.pop(future)
                    counters.set_pending_save_tasks(len(pending_save))
                    try:
                        completed_job = future.result()
                    except PipelineStageError as exc:
                        failures.append(
                            {
                                "input_path": str(exc.job.input_path),
                                "stage": exc.stage,
                                "error": str(exc.error),
                            }
                        )
                        LOGGER.exception("%s failed for %s", exc.stage.capitalize(), exc.job.input_path)
                        continue
                    except Exception as exc:
                        failures.append(
                            {
                                "input_path": str(finalized_job.job.input_path),
                                "stage": "save",
                                "error": str(exc),
                            }
                        )
                        LOGGER.exception(
                            "Save pipeline failed for %s",
                            finalized_job.job.input_path,
                        )
                        continue

                    timings["decode_seconds"] += completed_job.decode_seconds
                    timings["decode_queue_wait_seconds"] += completed_job.decode_queue_wait_seconds
                    timings["gpu_preprocess_seconds"] += completed_job.gpu_preprocess_seconds
                    timings["gpu_forward_seconds"] += completed_job.gpu_forward_seconds
                    timings["gpu_prepare_seconds"] += completed_job.gpu_prepare_seconds
                    timings["gpu_to_cpu_transfer_seconds"] += (
                        completed_job.gpu_to_cpu_transfer_seconds
                    )
                    timings["cpu_svd_seconds"] += completed_job.cpu_svd_seconds
                    timings["cpu_finalize_seconds"] += completed_job.cpu_finalize_seconds
                    timings["postprocess_queue_wait_seconds"] += (
                        completed_job.postprocess_queue_wait_seconds
                    )
                    timings["save_queue_wait_seconds"] += completed_job.save_queue_wait_seconds
                    timings["cpu_postprocess_seconds"] += completed_job.cpu_postprocess_seconds
                    timings["ply_tensor_export_seconds"] += completed_job.ply_tensor_export_seconds
                    timings["ply_vertex_fill_seconds"] += completed_job.ply_vertex_fill_seconds
                    timings["ply_metadata_pack_seconds"] += completed_job.ply_metadata_pack_seconds
                    timings["ply_pack_seconds"] += completed_job.ply_pack_seconds
                    timings["ply_write_seconds"] += completed_job.ply_write_seconds
                    timings["save_seconds"] += completed_job.save_seconds
                    queue_stats["max_decode_queue_wait_seconds"] = max(
                        queue_stats["max_decode_queue_wait_seconds"],
                        completed_job.decode_queue_wait_seconds,
                    )
                    queue_stats["max_postprocess_queue_wait_seconds"] = max(
                        queue_stats["max_postprocess_queue_wait_seconds"],
                        completed_job.postprocess_queue_wait_seconds,
                    )
                    queue_stats["max_save_queue_wait_seconds"] = max(
                        queue_stats["max_save_queue_wait_seconds"],
                        completed_job.save_queue_wait_seconds,
                    )
                    success_outputs.append(str(completed_job.job.output_path))
                    completion_times.append(completed_job.completed_at)
                    counters.increment_completed_images()

                    if args.profile:
                        profile_rows.append(
                            {
                                "input_path": str(completed_job.job.input_path),
                                "output_path": str(completed_job.job.output_path),
                                "decode_seconds": completed_job.decode_seconds,
                                "decode_queue_wait_seconds": completed_job.decode_queue_wait_seconds,
                                "gpu_preprocess_seconds": completed_job.gpu_preprocess_seconds,
                                "gpu_forward_seconds": completed_job.gpu_forward_seconds,
                                "gpu_prepare_seconds": completed_job.gpu_prepare_seconds,
                                "gpu_to_cpu_transfer_seconds": (
                                    completed_job.gpu_to_cpu_transfer_seconds
                                ),
                                "cpu_svd_seconds": completed_job.cpu_svd_seconds,
                                "cpu_finalize_seconds": completed_job.cpu_finalize_seconds,
                                "postprocess_queue_wait_seconds": (
                                    completed_job.postprocess_queue_wait_seconds
                                ),
                                "save_queue_wait_seconds": completed_job.save_queue_wait_seconds,
                                "cpu_postprocess_seconds": completed_job.cpu_postprocess_seconds,
                                "ply_tensor_export_seconds": (
                                    completed_job.ply_tensor_export_seconds
                                ),
                                "ply_vertex_fill_seconds": completed_job.ply_vertex_fill_seconds,
                                "ply_metadata_pack_seconds": (
                                    completed_job.ply_metadata_pack_seconds
                                ),
                                "ply_pack_seconds": completed_job.ply_pack_seconds,
                                "ply_write_seconds": completed_job.ply_write_seconds,
                                "save_seconds": completed_job.save_seconds,
                            }
                        )

                    LOGGER.info("Saved %s", completed_job.job.output_path)
                    update_postprocess_queue_stats()

            def collect_svd_futures(*, block: bool) -> None:
                if not pending_svd:
                    return

                done, _ = wait(
                    pending_svd.keys(),
                    timeout=None if block else 0,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    prepared_job = pending_svd.pop(future)
                    counters.set_pending_svd_tasks(len(pending_svd))
                    try:
                        finalized_job = future.result()
                    except PipelineStageError as exc:
                        failures.append(
                            {
                                "input_path": str(exc.job.input_path),
                                "stage": exc.stage,
                                "error": str(exc.error),
                            }
                        )
                        LOGGER.exception("%s failed for %s", exc.stage.capitalize(), exc.job.input_path)
                        continue
                    except Exception as exc:
                        failures.append(
                            {
                                "input_path": str(prepared_job.job.input_path),
                                "stage": "postprocess",
                                "error": str(exc),
                            }
                        )
                        LOGGER.exception(
                            "Postprocess pipeline failed for %s",
                            prepared_job.job.input_path,
                        )
                        continue

                    save_future = save_executor.submit(save_finalized_job, finalized_job)
                    pending_save[save_future] = finalized_job
                    counters.set_pending_save_tasks(len(pending_save))
                    queue_stats["max_pending_save_tasks"] = max(
                        queue_stats["max_pending_save_tasks"],
                        len(pending_save),
                    )
                    update_postprocess_queue_stats()

                    if len(pending_save) >= max_pending_save:
                        collect_save_futures(block=True)

            submit_decode_jobs()

            while decode_futures:
                done, _ = wait(decode_futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    job = decode_futures.pop(future)
                    counters.set_pending_decode_tasks(len(decode_futures))
                    try:
                        decoded_image = future.result()
                    except Exception as exc:
                        failures.append(
                            {
                                "input_path": str(job.input_path),
                                "stage": "decode",
                                "error": str(exc),
                            }
                        )
                        LOGGER.exception("Image decode failed for %s", job.input_path)
                        continue

                    LOGGER.info("Running GPU stages for %s", decoded_image.job.input_path)
                    try:
                        prepared_job = run_gpu_stage(predictor, decoded_image, device)
                    except Exception as exc:
                        failures.append(
                            {
                                "input_path": str(decoded_image.job.input_path),
                                "stage": "gpu",
                                "error": str(exc),
                            }
                        )
                        LOGGER.exception("GPU stage failed for %s", decoded_image.job.input_path)
                        continue

                    svd_future = svd_executor.submit(finalize_cpu_job, prepared_job)
                    pending_svd[svd_future] = prepared_job
                    counters.set_pending_svd_tasks(len(pending_svd))
                    queue_stats["max_pending_svd_tasks"] = max(
                        queue_stats["max_pending_svd_tasks"],
                        len(pending_svd),
                    )
                    update_postprocess_queue_stats()

                    if len(pending_svd) >= max_pending_svd:
                        collect_svd_futures(block=True)

                submit_decode_jobs()
                collect_svd_futures(block=False)
                collect_save_futures(block=False)

            while pending_svd:
                collect_svd_futures(block=True)
                collect_save_futures(block=False)

            while pending_save:
                collect_save_futures(block=True)
    finally:
        if hardware_sampler is not None:
            hardware_sampler.stop()

    timings["inference_seconds"] = (
        timings["gpu_preprocess_seconds"]
        + timings["gpu_forward_seconds"]
        + timings["gpu_prepare_seconds"]
        + timings["gpu_to_cpu_transfer_seconds"]
        + timings["cpu_svd_seconds"]
        + timings["cpu_finalize_seconds"]
    )
    timings["wall_seconds"] = time.perf_counter() - wall_start
    throughput_after_warmup = compute_throughput_after_warmup(completion_times)
    hardware_samples: list[dict[str, Any]] = []
    hardware_rollups: dict[str, float | None] = {}
    hardware_samples_path: str | None = None
    if hardware_sampler is not None:
        hardware_samples = hardware_sampler.collected_samples()
        hardware_rollups = summarize_hardware_samples(
            hardware_samples,
            args.hardware_sample_interval,
        )
        hardware_samples_path = str(write_hardware_samples(args.output_dir, hardware_samples))
    summary: dict[str, Any] = {
        "started_at": started_at,
        "finished_at": iso_timestamp(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "checkpoint_path": str(args.checkpoint_path),
        "device": str(device),
        "focal_35mm_mm": args.focal_35mm_mm,
        "decode_workers": args.decode_workers,
        "svd_workers": args.svd_workers,
        "save_workers": args.save_workers,
        "postprocess_workers": args.postprocess_workers,
        "write_workers": args.save_workers,
        "queue_depth": args.queue_depth,
        "profile_enabled": args.profile,
        "hardware_profile_enabled": args.hardware_profile,
        "hardware_sample_interval_seconds": (
            args.hardware_sample_interval if args.hardware_profile else None
        ),
        "hardware_sample_count": len(hardware_samples),
        "hardware_samples_path": hardware_samples_path,
        "hardware_rollups": hardware_rollups,
        "total_images": len(jobs),
        "successful_images": len(success_outputs),
        "failed_images": len(failures),
        "success_outputs": sorted(success_outputs),
        "failures": failures,
        "timings": timings,
        "throughput_after_warmup_images_per_second": throughput_after_warmup,
        "profiling": {
            "enabled": args.profile,
            "queue_stats": queue_stats,
        },
    }
    if args.profile:
        summary["profiling"]["per_image"] = sorted(  # type: ignore[index]
            profile_rows,
            key=lambda row: row["input_path"],
        )
    if args.hardware_profile and len(hardware_samples) <= MAX_INLINE_HARDWARE_SAMPLES:
        summary["hardware_samples"] = hardware_samples
    return summary


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.checkpoint_path = args.checkpoint_path.resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / LOG_FILENAME
    logging_utils.configure(logging.DEBUG if args.verbose else logging.INFO, log_path=log_path)

    try:
        summary = run_batch(args)
    except Exception:
        LOGGER.exception("Batch inference failed before completion.")
        return 1

    summary_path = write_summary(args.output_dir, summary)
    LOGGER.info(
        "Batch complete: %d succeeded, %d failed. Summary written to %s",
        summary["successful_images"],
        summary["failed_images"],
        summary_path,
    )
    return 0 if summary["failed_images"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

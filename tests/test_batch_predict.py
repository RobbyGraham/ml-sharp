"""Unit tests for the batch SHARP runner."""

from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from Batch.batch_predict import (
    HARDWARE_SAMPLES_FILENAME,
    SUMMARY_FILENAME,
    build_image_jobs,
    compute_throughput_after_warmup,
    default_decode_workers,
    default_postprocess_workers,
    default_save_workers,
    default_svd_workers,
    default_write_workers,
    derive_cpu_stage_workers,
    discover_image_paths,
    parse_args,
    resolve_cpu_stage_workers,
    summarize_hardware_samples,
    write_hardware_samples,
    write_summary,
)
from sharp.utils.gaussians import (
    Gaussians3D,
    compose_covariance_matrices,
    decompose_covariance_matrices,
    finalize_prepared_gaussians,
    finalize_prepared_gaussians_profiled,
    move_prepared_gaussians_to_cpu,
    prepare_gaussians_for_decomposition,
    save_ply,
    save_ply_profiled,
)


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp-tests"


class BatchPredictTests(unittest.TestCase):
    """Coverage for batch preflight helpers."""

    @staticmethod
    @contextmanager
    def make_temp_dir():
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        path = TEST_TMP_ROOT / f"case-{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        try:
            yield str(path)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_discover_image_paths_recurses_and_sorts(self) -> None:
        with self.make_temp_dir() as temp_dir:
            root = Path(temp_dir)
            (root / "b").mkdir()
            (root / "a").mkdir()
            (root / "b" / "two.JPG").write_bytes(b"test")
            (root / "a" / "one.png").write_bytes(b"test")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            image_paths = discover_image_paths(root)

            self.assertEqual(
                [path.relative_to(root).as_posix() for path in image_paths],
                ["a/one.png", "b/two.JPG"],
            )

    def test_build_image_jobs_rejects_duplicate_output_stems(self) -> None:
        with self.make_temp_dir() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "Input"
            output_dir = root / "Output"
            (input_dir / "first").mkdir(parents=True)
            (input_dir / "second").mkdir(parents=True)
            (input_dir / "first" / "scene.jpg").write_bytes(b"test")
            (input_dir / "second" / "scene.png").write_bytes(b"test")

            with self.assertRaisesRegex(ValueError, "Duplicate output filename"):
                build_image_jobs(input_dir, output_dir)

    def test_build_image_jobs_creates_flat_output_paths(self) -> None:
        with self.make_temp_dir() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "Input"
            output_dir = root / "Output"
            (input_dir / "nested").mkdir(parents=True)
            (input_dir / "nested" / "scene01.jpeg").write_bytes(b"test")

            jobs = build_image_jobs(input_dir, output_dir)

            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].output_path, output_dir / "scene01.ply")

    def test_write_summary_persists_json(self) -> None:
        with self.make_temp_dir() as temp_dir:
            output_dir = Path(temp_dir)
            summary = {"total_images": 1, "successful_images": 1, "failed_images": 0}

            summary_path = write_summary(output_dir, summary)

            self.assertEqual(summary_path, output_dir / SUMMARY_FILENAME)
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, summary)

    def test_default_worker_counts_are_positive(self) -> None:
        self.assertGreaterEqual(default_decode_workers(), 4)
        self.assertGreaterEqual(default_postprocess_workers(), 2)
        self.assertGreaterEqual(default_svd_workers(), 1)
        self.assertGreaterEqual(default_save_workers(), 1)
        self.assertEqual(default_save_workers(), default_write_workers())

    def test_derive_cpu_stage_workers_prefers_svd_capacity(self) -> None:
        self.assertEqual(derive_cpu_stage_workers(8), (5, 3))
        self.assertEqual(derive_cpu_stage_workers(3), (1, 2))
        self.assertEqual(derive_cpu_stage_workers(2), (1, 1))

    def test_write_workers_alias_sets_postprocess_workers(self) -> None:
        argv = [
            "batch_predict.py",
            "--input-dir",
            "Input",
            "--output-dir",
            "Output",
            "--checkpoint-path",
            "sharp.pt",
            "--focal-35mm-mm",
            "35",
            "--write-workers",
            "7",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.postprocess_workers, 7)
        self.assertEqual((args.svd_workers, args.save_workers), (4, 3))

    def test_explicit_cpu_stage_workers_override_total_workers(self) -> None:
        argv = [
            "batch_predict.py",
            "--input-dir",
            "Input",
            "--output-dir",
            "Output",
            "--checkpoint-path",
            "sharp.pt",
            "--focal-35mm-mm",
            "35",
            "--postprocess-workers",
            "8",
            "--svd-workers",
            "4",
            "--save-workers",
            "3",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.svd_workers, 4)
        self.assertEqual(args.save_workers, 3)
        self.assertEqual(args.postprocess_workers, 7)

    def test_hardware_profile_implies_profile(self) -> None:
        argv = [
            "batch_predict.py",
            "--input-dir",
            "Input",
            "--output-dir",
            "Output",
            "--checkpoint-path",
            "sharp.pt",
            "--focal-35mm-mm",
            "35",
            "--hardware-profile",
            "--hardware-sample-interval",
            "0.25",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertTrue(args.hardware_profile)
        self.assertTrue(args.profile)
        self.assertEqual(args.hardware_sample_interval, 0.25)

    def test_compute_throughput_after_warmup(self) -> None:
        throughput = compute_throughput_after_warmup([10.0, 12.0, 14.0, 16.0])
        self.assertEqual(throughput, 0.5)

    def test_write_hardware_samples_persists_jsonl(self) -> None:
        with self.make_temp_dir() as temp_dir:
            output_dir = Path(temp_dir)
            samples = [
                {
                    "timestamp": "2026-03-17T15:00:00+00:00",
                    "seconds_since_start": 0.5,
                    "gpu_util_percent": None,
                    "gpu_memory_used_mb": None,
                    "gpu_memory_total_mb": None,
                    "gpu_memory_util_percent": None,
                    "gpu_power_watts": None,
                    "gpu_temperature_c": None,
                    "cpu_util_percent": 20.0,
                    "ram_used_mb": 4096.0,
                    "ram_total_mb": 8192.0,
                    "ram_util_percent": 50.0,
                    "pending_decode_tasks": 4,
                    "pending_postprocess_tasks": 2,
                    "completed_images": 1,
                }
            ]

            samples_path = write_hardware_samples(output_dir, samples)

            self.assertEqual(samples_path, output_dir / HARDWARE_SAMPLES_FILENAME)
            lines = samples_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), samples[0])

    def test_summarize_hardware_samples_handles_missing_gpu_metrics(self) -> None:
        samples = [
            {
                "gpu_util_percent": None,
                "gpu_memory_used_mb": None,
                "cpu_util_percent": 40.0,
                "ram_util_percent": 60.0,
                "pending_postprocess_tasks": 3,
            },
            {
                "gpu_util_percent": None,
                "gpu_memory_used_mb": None,
                "cpu_util_percent": 50.0,
                "ram_util_percent": 70.0,
                "pending_postprocess_tasks": 1,
            },
        ]

        rollups = summarize_hardware_samples(samples, 0.5)

        self.assertIsNone(rollups["gpu_util_avg_percent"])
        self.assertEqual(rollups["cpu_util_avg_percent"], 45.0)
        self.assertEqual(rollups["ram_util_max_percent"], 70.0)
        self.assertEqual(rollups["seconds_gpu_below_threshold_with_postprocess_backlog"], 0.0)

    def test_summarize_hardware_samples_tracks_low_gpu_with_backlog(self) -> None:
        samples = [
            {
                "gpu_util_percent": 10.0,
                "gpu_memory_used_mb": 1000.0,
                "cpu_util_percent": 60.0,
                "ram_util_percent": 55.0,
                "pending_postprocess_tasks": 4,
            },
            {
                "gpu_util_percent": 80.0,
                "gpu_memory_used_mb": 1100.0,
                "cpu_util_percent": 70.0,
                "ram_util_percent": 65.0,
                "pending_postprocess_tasks": 0,
            },
        ]

        rollups = summarize_hardware_samples(samples, 0.5)

        self.assertEqual(rollups["gpu_util_max_percent"], 80.0)
        self.assertEqual(rollups["gpu_memory_used_max_mb"], 1100.0)
        self.assertEqual(rollups["seconds_gpu_below_threshold_with_postprocess_backlog"], 0.5)

    def test_resolve_cpu_stage_workers_derives_missing_stage(self) -> None:
        self.assertEqual(resolve_cpu_stage_workers(8, None, None), (5, 3))
        self.assertEqual(resolve_cpu_stage_workers(8, 5, None), (5, 3))
        self.assertEqual(resolve_cpu_stage_workers(8, None, 3), (5, 3))

    def test_exact_postprocess_boundary_preserves_ply_bytes(self) -> None:
        torch.manual_seed(7)
        gaussians = Gaussians3D(
            mean_vectors=torch.randn(1, 8, 3, dtype=torch.float32),
            singular_values=torch.rand(1, 8, 3, dtype=torch.float32) + 0.25,
            quaternions=F.normalize(torch.randn(1, 8, 4, dtype=torch.float32), dim=-1),
            colors=torch.rand(1, 8, 3, dtype=torch.float32),
            opacities=torch.sigmoid(torch.randn(1, 8, dtype=torch.float32)),
        )
        transform = torch.tensor(
            [
                [1.2, 0.1, 0.0, 0.5],
                [0.0, 0.9, -0.2, -0.3],
                [0.1, 0.0, 1.1, 1.0],
            ],
            dtype=torch.float32,
        )

        transform_linear = transform[..., :3]
        transform_offset = transform[..., 3]
        mean_vectors = gaussians.mean_vectors @ transform_linear.T + transform_offset
        covariance_matrices = compose_covariance_matrices(
            gaussians.quaternions,
            gaussians.singular_values,
        )
        covariance_matrices = (
            transform_linear @ covariance_matrices @ transform_linear.transpose(-1, -2)
        )
        quaternions, singular_values = decompose_covariance_matrices(covariance_matrices)
        reference_gaussians = Gaussians3D(
            mean_vectors=mean_vectors,
            singular_values=singular_values,
            quaternions=quaternions,
            colors=gaussians.colors,
            opacities=gaussians.opacities,
        )

        prepared_gaussians = prepare_gaussians_for_decomposition(gaussians, transform)
        prepared_gaussians_cpu = move_prepared_gaussians_to_cpu(prepared_gaussians)
        staged_gaussians = finalize_prepared_gaussians(
            prepared_gaussians_cpu,
            output_device=torch.device("cpu"),
            output_dtype=torch.float32,
        )

        with self.make_temp_dir() as temp_dir:
            root = Path(temp_dir)
            reference_path = root / "reference.ply"
            staged_path = root / "staged.ply"
            save_ply(reference_gaussians, 512.0, (480, 640), reference_path)
            save_ply(staged_gaussians, 512.0, (480, 640), staged_path)

            self.assertEqual(reference_path.read_bytes(), staged_path.read_bytes())

    def test_profiled_exact_wrappers_preserve_ply_bytes(self) -> None:
        torch.manual_seed(11)
        gaussians = Gaussians3D(
            mean_vectors=torch.randn(1, 6, 3, dtype=torch.float32),
            singular_values=torch.rand(1, 6, 3, dtype=torch.float32) + 0.5,
            quaternions=F.normalize(torch.randn(1, 6, 4, dtype=torch.float32), dim=-1),
            colors=torch.rand(1, 6, 3, dtype=torch.float32),
            opacities=torch.sigmoid(torch.randn(1, 6, dtype=torch.float32)),
        )
        transform = torch.tensor(
            [
                [1.0, 0.2, 0.0, 0.3],
                [0.0, 0.8, -0.1, -0.4],
                [0.2, 0.0, 1.3, 0.7],
            ],
            dtype=torch.float32,
        )

        prepared_gaussians = prepare_gaussians_for_decomposition(gaussians, transform)
        prepared_gaussians_cpu = move_prepared_gaussians_to_cpu(prepared_gaussians)
        profiled_result = finalize_prepared_gaussians_profiled(
            prepared_gaussians_cpu,
            output_device=torch.device("cpu"),
            output_dtype=torch.float32,
        )

        self.assertGreaterEqual(profiled_result.cpu_svd_seconds, 0.0)
        self.assertGreaterEqual(profiled_result.cpu_finalize_seconds, 0.0)

        with self.make_temp_dir() as temp_dir:
            root = Path(temp_dir)
            plain_path = root / "plain.ply"
            profiled_path = root / "profiled.ply"
            save_ply(profiled_result.gaussians, 512.0, (480, 640), plain_path)
            save_result = save_ply_profiled(profiled_result.gaussians, 512.0, (480, 640), profiled_path)

            self.assertGreaterEqual(save_result.ply_tensor_export_seconds, 0.0)
            self.assertGreaterEqual(save_result.ply_vertex_fill_seconds, 0.0)
            self.assertGreaterEqual(save_result.ply_metadata_pack_seconds, 0.0)
            self.assertGreaterEqual(save_result.ply_pack_seconds, 0.0)
            self.assertGreaterEqual(save_result.ply_write_seconds, 0.0)
            self.assertAlmostEqual(
                save_result.ply_pack_seconds,
                save_result.ply_tensor_export_seconds
                + save_result.ply_vertex_fill_seconds
                + save_result.ply_metadata_pack_seconds,
            )
            self.assertEqual(plain_path.read_bytes(), profiled_path.read_bytes())


if __name__ == "__main__":
    unittest.main()

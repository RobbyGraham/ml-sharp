"""Contains `sharp predict` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    PreparedGaussians3D,
    SceneMetaData,
    finalize_prepared_gaussians,
    get_unprojection_matrix,
    move_prepared_gaussians_to_cpu,
    prepare_gaussians_for_decomposition,
    save_ply,
)

from .render import render_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
INTERNAL_SHAPE = (1536, 1536)


class PreparedPredictionInputs(NamedTuple):
    """Preprocessed tensors and metadata for one SHARP prediction."""

    image_resized_pt: torch.Tensor
    disparity_factor: torch.Tensor
    intrinsics_resized: torch.Tensor
    image_shape: tuple[int, int]
    internal_shape: tuple[int, int]
    focal_length_px: float


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to an image or containing a list of images.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the predicted Gaussians and renderings.",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to the .pt checkpoint. If not provided, downloads the default model automatically.",
    required=False,
)
@click.option(
    "--render/--no-render",
    "with_rendering",
    is_flag=True,
    default=False,
    help="Whether to render trajectory for checkpoint.",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    device: str,
    verbose: bool,
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    extensions = io.get_supported_image_extensions()

    image_paths = []
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    if len(image_paths) == 0:
        LOGGER.info("No valid images found. Input was %s.", input_path)
        return

    LOGGER.info("Processing %d valid image files.", len(image_paths))

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info("Using device %s", device)

    if with_rendering and device != "cuda":
        LOGGER.warning("Can only run rendering with gsplat on CUDA. Rendering is disabled.")
        with_rendering = False

    # Load or download checkpoint
    if checkpoint_path is None:
        LOGGER.info("No checkpoint provided. Downloading default model from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    output_path.mkdir(exist_ok=True, parents=True)

    for image_path in image_paths:
        LOGGER.info("Processing %s", image_path)
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]
        intrinsics = torch.tensor(
            [
                [f_px, 0, (width - 1) / 2.0, 0],
                [0, f_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )
        gaussians = predict_image(gaussian_predictor, image, f_px, torch.device(device))

        LOGGER.info("Saving 3DGS to %s", output_path)
        save_ply(gaussians, f_px, (height, width), output_path / f"{image_path.stem}.ply")

        if with_rendering:
            output_video_path = (output_path / image_path.stem).with_suffix(".mp4")
            LOGGER.info("Rendering trajectory to %s", output_video_path)

            metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")
            render_gaussians(gaussians, metadata, output_video_path)


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
) -> Gaussians3D:
    """Predict Gaussians from an image."""
    LOGGER.info("Running preprocessing.")
    prepared_inputs = prepare_image_for_prediction(image, f_px, device)

    # Predict Gaussians in the NDC space.
    LOGGER.info("Running inference.")
    gaussians_ndc = predict_image_ndc(predictor, prepared_inputs)

    LOGGER.info("Running postprocessing.")
    prepared_gaussians = prepare_prediction_postprocess_inputs(gaussians_ndc, prepared_inputs)
    return finalize_prepared_gaussians(
        prepared_gaussians,
        output_device=device,
        output_dtype=torch.float32,
    )


def prepare_image_for_prediction(
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    internal_shape: tuple[int, int] = INTERNAL_SHAPE,
) -> PreparedPredictionInputs:
    """Preprocess one image for SHARP prediction."""
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width], device=device, dtype=torch.float32)
    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )
    intrinsics = torch.tensor(
        [
            [f_px, 0, width / 2, 0],
            [0, f_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height
    return PreparedPredictionInputs(
        image_resized_pt=image_resized_pt,
        disparity_factor=disparity_factor,
        intrinsics_resized=intrinsics_resized,
        image_shape=(height, width),
        internal_shape=internal_shape,
        focal_length_px=f_px,
    )


def predict_image_ndc(
    predictor: RGBGaussianPredictor,
    prepared_inputs: PreparedPredictionInputs,
) -> Gaussians3D:
    """Run the model forward pass and return NDC Gaussians."""
    return predictor(
        prepared_inputs.image_resized_pt,
        prepared_inputs.disparity_factor,
    )


def prepare_prediction_postprocess_inputs(
    gaussians_ndc: Gaussians3D,
    prepared_inputs: PreparedPredictionInputs,
) -> PreparedGaussians3D:
    """Prepare exact postprocess inputs on the current device."""
    device = prepared_inputs.image_resized_pt.device
    extrinsics = torch.eye(4, device=device, dtype=prepared_inputs.intrinsics_resized.dtype)
    unprojection_matrix = get_unprojection_matrix(
        extrinsics,
        prepared_inputs.intrinsics_resized,
        prepared_inputs.internal_shape,
    )
    return prepare_gaussians_for_decomposition(gaussians_ndc, unprojection_matrix[:3])


def prepare_prediction_postprocess_inputs_cpu(
    gaussians_ndc: Gaussians3D,
    prepared_inputs: PreparedPredictionInputs,
) -> PreparedGaussians3D:
    """Prepare exact postprocess inputs on CPU for concurrent finalization."""
    prepared_gaussians = prepare_prediction_postprocess_inputs(gaussians_ndc, prepared_inputs)
    return move_prepared_gaussians_to_cpu(prepared_gaussians)

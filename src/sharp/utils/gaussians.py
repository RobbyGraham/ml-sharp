"""Contains basic data structures and functionality for 3D Gaussians.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from sharp.utils import color_space as cs_utils
from sharp.utils import linalg

LOGGER = logging.getLogger(__name__)


BackgroundColor = Literal["black", "white", "random_color", "random_pixel"]


class Gaussians3D(NamedTuple):
    """Represents a collection of 3D Gaussians."""

    mean_vectors: torch.Tensor
    singular_values: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor

    def to(self, device: torch.device) -> Gaussians3D:
        """Move Gaussians to device."""
        return Gaussians3D(
            mean_vectors=self.mean_vectors.to(device),
            singular_values=self.singular_values.to(device),
            quaternions=self.quaternions.to(device),
            colors=self.colors.to(device),
            opacities=self.opacities.to(device),
        )


class PreparedGaussians3D(NamedTuple):
    """Intermediate representation used before covariance decomposition."""

    mean_vectors: torch.Tensor
    covariance_matrices: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor


class CovarianceDecompositionResult(NamedTuple):
    """Exact covariance decomposition result with timing breakdown."""

    quaternions: torch.Tensor
    singular_values: torch.Tensor
    svd_seconds: float
    reflection_seconds: float
    quaternion_conversion_seconds: float
    singular_value_seconds: float


class FinalizedGaussiansResult(NamedTuple):
    """Finalized Gaussians with exact CPU timing breakdown."""

    gaussians: Gaussians3D
    cpu_svd_seconds: float
    cpu_finalize_seconds: float
    cpu_reflection_seconds: float
    cpu_quaternion_conversion_seconds: float
    cpu_singular_value_seconds: float
    cpu_assembly_seconds: float


class PlySaveResult(NamedTuple):
    """PLY export result with timing breakdown."""

    plydata: PlyData
    ply_tensor_export_seconds: float
    ply_vertex_fill_seconds: float
    ply_metadata_pack_seconds: float
    ply_pack_seconds: float
    ply_write_seconds: float


class SceneMetaData(NamedTuple):
    """Meta data about Gaussian scene."""

    focal_length_px: float
    resolution_px: tuple[int, int]
    color_space: cs_utils.ColorSpace


def get_unprojection_matrix(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    """Compute unprojection matrix to transform Gaussians to Euclidean space.

    Args:
        extrinsics: The 4x4 extrinsics matrix of the camera view.
        intrinsics: The 4x4 intrinsics matrix of the camera view.
        image_shape: The (width, height) of the input image.

    Returns:
        A 4x4 matrix to transform Gaussians from NDC space to Euclidean space.
    """
    device = intrinsics.device
    image_width, image_height = image_shape
    # This matrix converts OpenCV pixel coordinates to NDC coordinates where
    # (-1, 1) denotes the top left and (1, 1) the bottom right of the image.
    #
    # Note that premultiplying the intrinsics with ndc_matrix typically yields a matrix
    # that simply scales the x-axis by 2 * focal_length / image_width and the y-axis by
    # 2 * focal_length / image_height.
    ndc_matrix = torch.tensor(
        [
            [2.0 / image_width, 0.0, -1.0, 0.0],
            [0.0, 2.0 / image_height, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )
    return torch.linalg.inv(ndc_matrix @ intrinsics @ extrinsics)


def unproject_gaussians(
    gaussians_ndc: Gaussians3D,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> Gaussians3D:
    """Unproject Gaussians from NDC space to world coordinates."""
    unprojection_matrix = get_unprojection_matrix(extrinsics, intrinsics, image_shape)
    gaussians = apply_transform(gaussians_ndc, unprojection_matrix[:3])
    return gaussians


def apply_transform(gaussians: Gaussians3D, transform: torch.Tensor) -> Gaussians3D:
    """Apply an affine transformation to 3D Gaussians.

    Args:
        gaussians: The Gaussians to transform.
        transform: An affine transform with shape 3x4.

    Returns:
        The transformed Gaussians.

    Note: This operation is not differentiable.
    """
    prepared_gaussians = prepare_gaussians_for_decomposition(gaussians, transform)
    return finalize_prepared_gaussians(
        prepared_gaussians,
        output_device=gaussians.mean_vectors.device,
        output_dtype=gaussians.mean_vectors.dtype,
    )


def decompose_covariance_matrices(
    covariance_matrices: torch.Tensor,
    output_dtype: torch.dtype | None = None,
    output_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose 3D covariance matrices into quaternions and singular values.

    Args:
        covariance_matrices: The covariance matrices to decompose.

    Returns:
        Quaternion and singular values corresponding to the orientation and scales of
        the diagonalized matrix.

    Note: This operation is not differentiable.
    """
    result = decompose_covariance_matrices_profiled(
        covariance_matrices,
        output_dtype=output_dtype,
        output_device=output_device,
    )
    return result.quaternions, result.singular_values


def decompose_covariance_matrices_profiled(
    covariance_matrices: torch.Tensor,
    output_dtype: torch.dtype | None = None,
    output_device: torch.device | None = None,
) -> CovarianceDecompositionResult:
    """Decompose covariance matrices and capture exact CPU timing."""
    device = covariance_matrices.device
    dtype = covariance_matrices.dtype
    if output_dtype is None:
        output_dtype = dtype
    if output_device is None:
        output_device = device

    covariance_matrices = covariance_matrices.detach().to(device="cpu", dtype=torch.float64)

    svd_start = time.perf_counter()
    rotations, singular_values_2, _ = torch.linalg.svd(covariance_matrices)
    svd_seconds = time.perf_counter() - svd_start

    reflection_start = time.perf_counter()
    batch_idx, gaussian_idx = torch.where(torch.linalg.det(rotations) < 0)
    num_reflections = len(gaussian_idx)
    if num_reflections > 0:
        LOGGER.warning(
            "Received %d reflection matrices from SVD. Flipping them to rotations.",
            num_reflections,
        )
        rotations[batch_idx, gaussian_idx, :, -1] *= -1
    reflection_seconds = time.perf_counter() - reflection_start

    quaternion_start = time.perf_counter()
    quaternions = linalg.quaternions_from_rotation_matrices(rotations)
    quaternions = quaternions.to(dtype=output_dtype, device=output_device)
    quaternion_conversion_seconds = time.perf_counter() - quaternion_start

    singular_value_start = time.perf_counter()
    singular_values = singular_values_2.sqrt().to(dtype=output_dtype, device=output_device)
    singular_value_seconds = time.perf_counter() - singular_value_start

    return CovarianceDecompositionResult(
        quaternions=quaternions,
        singular_values=singular_values,
        svd_seconds=svd_seconds,
        reflection_seconds=reflection_seconds,
        quaternion_conversion_seconds=quaternion_conversion_seconds,
        singular_value_seconds=singular_value_seconds,
    )


def prepare_gaussians_for_decomposition(
    gaussians: Gaussians3D,
    transform: torch.Tensor,
) -> PreparedGaussians3D:
    """Apply an affine transform and keep covariance matrices pre-SVD."""
    transform_linear = transform[..., :3, :3]
    transform_offset = transform[..., :3, 3]

    mean_vectors = gaussians.mean_vectors @ transform_linear.T + transform_offset
    covariance_matrices = compose_covariance_matrices(
        gaussians.quaternions, gaussians.singular_values
    )
    covariance_matrices = (
        transform_linear @ covariance_matrices @ transform_linear.transpose(-1, -2)
    )
    return PreparedGaussians3D(
        mean_vectors=mean_vectors,
        covariance_matrices=covariance_matrices,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def move_prepared_gaussians_to_cpu(
    prepared_gaussians: PreparedGaussians3D,
    output_dtype: torch.dtype = torch.float32,
) -> PreparedGaussians3D:
    """Move prepared Gaussians to CPU while preserving exact decomposition inputs."""
    return PreparedGaussians3D(
        mean_vectors=prepared_gaussians.mean_vectors.detach().to(device="cpu", dtype=output_dtype),
        covariance_matrices=prepared_gaussians.covariance_matrices.detach().to(
            device="cpu",
            dtype=torch.float64,
        ),
        colors=prepared_gaussians.colors.detach().to(device="cpu", dtype=output_dtype),
        opacities=prepared_gaussians.opacities.detach().to(device="cpu", dtype=output_dtype),
    )


def finalize_prepared_gaussians(
    prepared_gaussians: PreparedGaussians3D,
    output_device: torch.device | None = None,
    output_dtype: torch.dtype | None = None,
) -> Gaussians3D:
    """Finalize prepared Gaussians by running exact covariance decomposition."""
    return finalize_prepared_gaussians_profiled(
        prepared_gaussians,
        output_device=output_device,
        output_dtype=output_dtype,
    ).gaussians


def finalize_prepared_gaussians_profiled(
    prepared_gaussians: PreparedGaussians3D,
    output_device: torch.device | None = None,
    output_dtype: torch.dtype | None = None,
) -> FinalizedGaussiansResult:
    """Finalize prepared Gaussians and capture exact CPU timing."""
    if output_device is None:
        output_device = prepared_gaussians.mean_vectors.device
    if output_dtype is None:
        output_dtype = prepared_gaussians.mean_vectors.dtype

    decomposition = decompose_covariance_matrices_profiled(
        prepared_gaussians.covariance_matrices,
        output_dtype=output_dtype,
        output_device=output_device,
    )

    assembly_start = time.perf_counter()

    if (
        prepared_gaussians.mean_vectors.device == output_device
        and prepared_gaussians.mean_vectors.dtype == output_dtype
    ):
        mean_vectors = prepared_gaussians.mean_vectors
    else:
        mean_vectors = prepared_gaussians.mean_vectors.to(device=output_device, dtype=output_dtype)

    if prepared_gaussians.colors.device == output_device and prepared_gaussians.colors.dtype == output_dtype:
        colors = prepared_gaussians.colors
    else:
        colors = prepared_gaussians.colors.to(device=output_device, dtype=output_dtype)

    if (
        prepared_gaussians.opacities.device == output_device
        and prepared_gaussians.opacities.dtype == output_dtype
    ):
        opacities = prepared_gaussians.opacities
    else:
        opacities = prepared_gaussians.opacities.to(device=output_device, dtype=output_dtype)

    gaussians = Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=decomposition.singular_values,
        quaternions=decomposition.quaternions,
        colors=colors,
        opacities=opacities,
    )
    cpu_assembly_seconds = time.perf_counter() - assembly_start
    cpu_finalize_seconds = (
        decomposition.reflection_seconds
        + decomposition.quaternion_conversion_seconds
        + decomposition.singular_value_seconds
        + cpu_assembly_seconds
    )
    return FinalizedGaussiansResult(
        gaussians=gaussians,
        cpu_svd_seconds=decomposition.svd_seconds,
        cpu_finalize_seconds=cpu_finalize_seconds,
        cpu_reflection_seconds=decomposition.reflection_seconds,
        cpu_quaternion_conversion_seconds=decomposition.quaternion_conversion_seconds,
        cpu_singular_value_seconds=decomposition.singular_value_seconds,
        cpu_assembly_seconds=cpu_assembly_seconds,
    )


def compose_covariance_matrices(
    quaternions: torch.Tensor, singular_values: torch.Tensor
) -> torch.Tensor:
    """Compose 3D covariance matrices into quaternions and singular values.

    Args:
        quaternions: The quaternions describing the principal basis.
        singular_values: The scales of the diagonalized matrix.

    Returns:
        The 3x3 covariances matrices.
    """
    device = quaternions.device
    rotations = linalg.rotation_matrices_from_quaternions(quaternions)
    diagonal_matrix = torch.eye(3, device=device) * singular_values[..., :, None]
    return rotations @ diagonal_matrix.square() @ rotations.transpose(-1, -2)


def convert_spherical_harmonics_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    """Convert degree-0 spherical harmonics to RGB.

    Reference:
        https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    """
    coeff_degree0 = np.sqrt(1.0 / (4.0 * np.pi))
    return sh0 * coeff_degree0 + 0.5


def convert_rgb_to_spherical_harmonics(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to degree-0 spherical harmonics.

    Reference:
        https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    """
    coeff_degree0 = np.sqrt(1.0 / (4.0 * np.pi))
    return (rgb - 0.5) / coeff_degree0


def load_ply(path: Path) -> tuple[Gaussians3D, SceneMetaData]:
    """Loads a ply from a file."""
    plydata = PlyData.read(path)

    vertices = next(filter(lambda x: x.name == "vertex", plydata.elements))

    properties = ["x", "y", "z"]
    properties.extend([f"f_dc_{i}" for i in range(3)])
    properties.extend([f"scale_{i}" for i in range(3)])
    properties.extend([f"rot_{i}" for i in range(3)])

    for prop in properties:
        if prop not in vertices:
            raise KeyError(f"Incompatible ply file: property {prop} not found in ply elements.")
    mean_vectors = np.stack(
        (
            np.asarray(vertices["x"]),
            np.asarray(vertices["y"]),
            np.asarray(vertices["z"]),
        ),
        axis=1,
    )

    scale_logits = np.stack(
        (
            np.asarray(vertices["scale_0"]),
            np.asarray(vertices["scale_1"]),
            np.asarray(vertices["scale_2"]),
        ),
        axis=1,
    )

    quaternions = np.stack(
        (
            np.asarray(vertices["rot_0"]),
            np.asarray(vertices["rot_1"]),
            np.asarray(vertices["rot_2"]),
            np.asarray(vertices["rot_3"]),
        ),
        axis=1,
    )

    spherical_harmonics_deg0 = np.stack(
        (
            np.asarray(vertices["f_dc_0"]),
            np.asarray(vertices["f_dc_1"]),
            np.asarray(vertices["f_dc_2"]),
        ),
        axis=1,
    )

    colors = convert_spherical_harmonics_to_rgb(spherical_harmonics_deg0)

    opacity_logits = np.asarray(vertices["opacity"])[..., None]

    supplement_elements = [element for element in plydata.elements if element.name != "vertex"]
    supplement_data: dict[str, Any] = {}
    supplement_keys = ["extrinsic", "intrinsic", "color_space", "image_size"]

    for element in supplement_elements:
        for key in supplement_keys:
            if key not in supplement_data and key in element:
                supplement_data[key] = np.asarray(element[key])

    # Parse intrinsics and image_size.
    if "intrinsic" in supplement_data:
        intrinsics_data = supplement_data["intrinsic"]

        # Legacy: image_size is contained in intrinsic element.
        if "image_size" not in supplement_data:
            if len(intrinsics_data) != 4:
                raise ValueError(
                    "Expect legacy intrinsics with len=4 containing image size, "
                    f"but received len={len(intrinsics_data)}"
                )
            focal_length_px = (intrinsics_data[0], intrinsics_data[1])
            width = int(intrinsics_data[2])
            height = int(intrinsics_data[3])

        else:
            if len(intrinsics_data) != 9:
                raise ValueError(
                    "Expect 9 elements in intrinsics, " f"but received {len(intrinsics_data)}."
                )
            intrinsics_matrix = intrinsics_data.reshape((3, 3))
            focal_length_px = (intrinsics_matrix[0, 0], intrinsics_matrix[1, 1])

            image_size_data = supplement_data["image_size"]
            width = image_size_data[0]
            height = image_size_data[1]

    # Default to VGA resolution: focal length = 512, image size = (640, 480).
    else:
        focal_length_px = (512, 512)
        width = 640
        height = 480

    # Parse extrinsics.
    extrinsics_data = supplement_data.get("extrinsic", np.eye(4).flatten())
    extrinsics_matrix = np.eye(4)

    # Legacy: extrinsics store 12 elements.
    if len(extrinsics_data) == 12:
        extrinsics_matrix[:3] = extrinsics_data.reshape((3, 4))
        extrinsics_matrix[:3, :3] = extrinsics_matrix[:3, :3].copy().T
    elif len(extrinsics_data) == 16:
        extrinsics_matrix[:] = extrinsics_data.reshape((4, 4))
    else:
        raise ValueError(f"Unrecognized extrinsics matrix shape {len(extrinsics_data)}")

    # Parse color space.
    color_space_index = supplement_data.get("color_space", 1)
    color_space = cs_utils.decode_color_space(color_space_index)
    colors = torch.from_numpy(colors).view(1, -1, 3).float()

    if color_space == "sRGB":
        # Convert to linearRGB for proper alpha blending.
        colors = cs_utils.sRGB2linearRGB(colors.flatten(0, 1)).view(1, -1, 3)
        color_space = "linearRGB"

    mean_vectors = torch.from_numpy(mean_vectors).view(1, -1, 3).float()
    quaternions = torch.from_numpy(quaternions).view(1, -1, 4).float()
    singular_values = torch.exp(torch.from_numpy(scale_logits).view(1, -1, 3)).float()
    opacities = torch.sigmoid(torch.from_numpy(opacity_logits).view(1, -1)).float()

    gaussians = Gaussians3D(
        mean_vectors=mean_vectors,
        quaternions=quaternions,
        singular_values=singular_values,
        opacities=opacities,
        colors=colors,
    )
    metadata = SceneMetaData(focal_length_px[0], (width, height), color_space)
    return gaussians, metadata


@torch.no_grad()
def build_plydata(
    gaussians: Gaussians3D,
    f_px: float,
    image_shape: tuple[int, int],
) -> PlyData:
    """Build PlyData for a predicted Gaussian3D without writing it to disk."""
    return build_plydata_profiled(gaussians, f_px, image_shape).plydata


@torch.no_grad()
def build_plydata_profiled(
    gaussians: Gaussians3D,
    f_px: float,
    image_shape: tuple[int, int],
) -> PlySaveResult:
    """Build PlyData for a predicted Gaussian3D and capture pack timings."""

    def _inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(tensor / (1.0 - tensor))

    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.detach()
        if tensor.device.type == "cpu":
            return tensor.numpy()
        return tensor.cpu().numpy()

    xyz = gaussians.mean_vectors.flatten(0, 1)
    scale_logits = torch.log(gaussians.singular_values).flatten(0, 1)
    quaternions = gaussians.quaternions.flatten(0, 1)

    # SHARP takes an image, convert it to sRGB color space as input,
    # and predicts linearRGB Gaussians as output.
    # The SHARP renderer would blend linearRGB Gaussians and convert rendered images and videos
    # back to sRGB for the best display quality.
    #
    # However, public renderers do not have such linear2sRGB conversions after rendering.
    # If they render linearRGB Gaussians as-is, the output would be dark without Gamma correction.
    #
    # To make it compatible to public renderers, we force convert linearRGB to sRGB during export.
    # - The SHARP renderer will still handle conversions properly.
    # - Public renderers will be mostly working fine when regarding sRGB images as linearRGB images,
    #   although for the best performance, it is recommended to apply the conversions.
    colors = convert_rgb_to_spherical_harmonics(
        cs_utils.linearRGB2sRGB(gaussians.colors.flatten(0, 1))
    )
    color_space_index = cs_utils.encode_color_space("sRGB")

    # Store opacity logits.
    opacity_logits = _inverse_sigmoid(gaussians.opacities).flatten(0, 1).unsqueeze(-1)

    attributes = torch.cat(
        (
            xyz,
            colors,
            opacity_logits,
            scale_logits,
            quaternions,
        ),
        dim=1,
    )
    tensor_export_start = time.perf_counter()
    attributes_np = np.ascontiguousarray(_to_numpy(attributes), dtype=np.float32)
    ply_tensor_export_seconds = time.perf_counter() - tensor_export_start

    dtype_full = [
        (attribute, "f4")
        for attribute in ["x", "y", "z"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    ]

    num_gaussians = len(xyz)
    vertex_fill_start = time.perf_counter()
    elements = np.empty(num_gaussians, dtype=dtype_full)
    field_names = [field for field, _ in dtype_full]
    for index, field_name in enumerate(field_names):
        elements[field_name] = attributes_np[:, index]
    ply_vertex_fill_seconds = time.perf_counter() - vertex_fill_start

    metadata_pack_start = time.perf_counter()
    vertex_elements = PlyElement.describe(elements, "vertex")

    # Load image-wise metadata.
    image_height, image_width = image_shape

    # Export image size.
    dtype_image_size = [("image_size", "u4")]
    image_size_array = np.empty(2, dtype=dtype_image_size)
    image_size_array[:] = np.array([image_width, image_height])
    image_size_element = PlyElement.describe(image_size_array, "image_size")

    # Export intrinsics.
    dtype_intrinsic = [("intrinsic", "f4")]
    intrinsic_array = np.empty(9, dtype=dtype_intrinsic)
    intrinsic = np.array(
        [
            f_px,
            0,
            image_width * 0.5,
            0,
            f_px,
            image_height * 0.5,
            0,
            0,
            1,
        ]
    )
    intrinsic_array[:] = intrinsic.flatten()
    intrinsic_element = PlyElement.describe(intrinsic_array, "intrinsic")

    # Export dummy extrinsics.
    dtype_extrinsic = [("extrinsic", "f4")]
    extrinsic_array = np.empty(16, dtype=dtype_extrinsic)
    extrinsic_array[:] = np.eye(4).flatten()
    extrinsic_element = PlyElement.describe(extrinsic_array, "extrinsic")

    # Export number of frames and particles per frame.
    dtype_frames = [("frame", "i4")]
    frame_array = np.empty(2, dtype=dtype_frames)
    frame_array[:] = np.array([1, num_gaussians], dtype=np.int32)
    frame_element = PlyElement.describe(frame_array, "frame")

    # Export disparity ranges for transform.
    dtype_disparity = [("disparity", "f4")]
    disparity_array = np.empty(2, dtype=dtype_disparity)

    disparity = 1.0 / gaussians.mean_vectors[0, ..., -1]
    quantiles = (
        torch.quantile(disparity, q=torch.tensor([0.1, 0.9], device=disparity.device))
        .float()
        .cpu()
        .numpy()
    )
    disparity_array[:] = quantiles
    disparity_element = PlyElement.describe(disparity_array, "disparity")

    # Export colorspace.
    dtype_color_space = [("color_space", "u1")]
    color_space_array = np.empty(1, dtype=dtype_color_space)
    color_space_array[:] = np.array([color_space_index]).flatten()
    color_space_element = PlyElement.describe(color_space_array, "color_space")

    dtype_version = [("version", "u1")]
    version_array = np.empty(3, dtype=dtype_version)
    version_array[:] = np.array([1, 5, 0], dtype=np.uint8).flatten()
    version_element = PlyElement.describe(version_array, "version")

    plydata = PlyData(
        [
            vertex_elements,
            extrinsic_element,
            intrinsic_element,
            image_size_element,
            frame_element,
            disparity_element,
            color_space_element,
            version_element,
        ]
    )
    ply_metadata_pack_seconds = time.perf_counter() - metadata_pack_start
    ply_pack_seconds = (
        ply_tensor_export_seconds + ply_vertex_fill_seconds + ply_metadata_pack_seconds
    )
    return PlySaveResult(
        plydata=plydata,
        ply_tensor_export_seconds=ply_tensor_export_seconds,
        ply_vertex_fill_seconds=ply_vertex_fill_seconds,
        ply_metadata_pack_seconds=ply_metadata_pack_seconds,
        ply_pack_seconds=ply_pack_seconds,
        ply_write_seconds=0.0,
    )


@torch.no_grad()
def save_ply_profiled(
    gaussians: Gaussians3D,
    f_px: float,
    image_shape: tuple[int, int],
    path: Path,
) -> PlySaveResult:
    """Save a predicted Gaussian3D to a ply file with timing breakdown."""
    build_result = build_plydata_profiled(gaussians, f_px, image_shape)

    write_start = time.perf_counter()
    build_result.plydata.write(path)
    ply_write_seconds = time.perf_counter() - write_start
    return PlySaveResult(
        plydata=build_result.plydata,
        ply_tensor_export_seconds=build_result.ply_tensor_export_seconds,
        ply_vertex_fill_seconds=build_result.ply_vertex_fill_seconds,
        ply_metadata_pack_seconds=build_result.ply_metadata_pack_seconds,
        ply_pack_seconds=build_result.ply_pack_seconds,
        ply_write_seconds=ply_write_seconds,
    )


@torch.no_grad()
def save_ply(
    gaussians: Gaussians3D, f_px: float, image_shape: tuple[int, int], path: Path
) -> PlyData:
    """Save a predicted Gaussian3D to a ply file."""
    return save_ply_profiled(gaussians, f_px, image_shape, path).plydata

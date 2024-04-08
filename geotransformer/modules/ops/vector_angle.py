import torch
import numpy as np


def rad2deg(rad: torch.Tensor) -> torch.Tensor:
    factor = 180.0 / np.pi
    deg = rad * factor
    return deg


def deg2rad(deg: torch.Tensor) -> torch.Tensor:
    factor = np.pi / 180.0
    rad = deg * factor
    return rad


def my_atan2(y, x):
    pi = torch.from_numpy(np.array([np.pi])).to(y.device, y.dtype)
    ans = torch.atan(y / (x + 1E-12))
    ans = torch.where((y >= 0) * (x < 0), ans + pi, ans)  # upper left quadrant
    ans = torch.where((y < 0) * (x < 0), ans - pi, ans)  # lower left quadrant
    # upper right quadrant and lower right quadrant, do nothing
    return ans


def vector_angle(x: torch.Tensor, y: torch.Tensor, dim: int, use_degree: bool = False):
    r"""Compute the angles between two set of 3D vectors.

    Args:
        x (Tensor): set of vectors (*, 3, *)
        y (Tensor): set of vectors (*, 3, *).
        dim (int): dimension index of the coordinates.
        use_degree (bool=False): If True, return angles in degree instead of rad.

    Returns:
        angles (Tensor): (*)
    """
    cross = torch.linalg.norm(torch.cross(x, y, dim=dim), dim=dim)  # (*, 3 *) x (*, 3, *) -> (*, 3, *) -> (*)
    dot = torch.sum(x * y, dim=dim)  # (*, 3 *) x (*, 3, *) -> (*)
    angles = my_atan2(cross, dot)  # (*)
    # angles = torch.atan(cross/(dot+1E-6))
    if use_degree:
        angles = rad2deg(angles)
    return angles

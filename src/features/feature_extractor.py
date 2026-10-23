"""
src/features/feature_extractor.py
158D Invariant Mixed Features with Context Gate (upgraded from 157D / 112D).
Structure: 76D hand1 + 76D hand2 + 5D inter-hand + 1D Context Gate Mode = 158D
  Each hand 76D: 63D (21pts x,y,z canonical) + 10D angles_3d + 3D velocity
  Inter-hand 5D: rel_dist_3d, dx, dy, dz, rel_angle
  Context Gate 1D: normalized mode_id [0.0..1.0] (mode_id / MAX_MODES)
Why Z matters:
  - Occluded fingers (fist): 2D projection makes curled look like extended
  - Palm orientation (front vs back): identical in 2D, z discriminates
  - Foreshortening: finger pointing at camera shrinks to a dot in 2D
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Any

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


def _get_coords(pt: Any) -> Tuple[float, float, float]:
    """Extract (x, y, z) from landmark object, dict, 4-tuple [id,x,y,z], 3-tuple [x,y,z], or 2-tuple."""
    if hasattr(pt, "x") and hasattr(pt, "y"):
        return float(pt.x), float(pt.y), float(getattr(pt, "z", 0.0))
    if isinstance(pt, dict):
        return float(pt.get("x", 0.0)), float(pt.get("y", 0.0)), float(pt.get("z", 0.0))
    if len(pt) >= 4:
        return float(pt[1]), float(pt[2]), float(pt[3])
    if len(pt) == 3:
        return float(pt[0]), float(pt[1]), float(pt[2])
    if len(pt) == 2:
        return float(pt[0]), float(pt[1]), 0.0
    return 0.0, 0.0, 0.0


def calculate_angle_3pts_3d(a, b, c) -> float:
    """Angle at vertex b in 3D space (0 to 180 degrees)."""
    v1 = (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    v2 = (c[0]-b[0], c[1]-b[1], c[2]-b[2])
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 180.0
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2)))))


def calculate_angle_3pts(a, b, c) -> float:
    """2D backward-compat alias -> delegates to 3D version with z=0."""
    return calculate_angle_3pts_3d(
        (float(a[0]), float(a[1]), 0.0),
        (float(b[0]), float(b[1]), 0.0),
        (float(c[0]), float(c[1]), 0.0),
    )


def get_palm_rotation_and_size(lm_list: List[Any]) -> Tuple[float, float]:
    """Palm axis angle theta (vs vertical up) and palm size (2D XY plane)."""
    wx, wy, _ = _get_coords(lm_list[WRIST])
    mx, my, _ = _get_coords(lm_list[MIDDLE_MCP])
    vx, vy = mx - wx, my - wy
    palm_size = max(10.0, math.hypot(vx, vy))
    theta = math.atan2(vy, vx) + (math.pi / 2.0)
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta <= -math.pi:
        theta += 2.0 * math.pi
    return theta, palm_size


def extract_single_hand_features(
    lm_list: List[Any],
    velocity: Tuple = (0.0, 0.0, 0.0),
) -> List[float]:
    """Extract 76D feature vector for one hand (rotation & scale invariant).

    76D = 63D canonical coords (21 pts x 3) + 10D joint angles + 3D velocity
    """
    if not lm_list or len(lm_list) < 21:
        return [0.0] * 76
    theta, palm_size = get_palm_rotation_and_size(lm_list)
    wx, wy, wz = _get_coords(lm_list[WRIST])
    cos_a = math.cos(-theta)
    sin_a = math.sin(-theta)
    coords_63 = []
    for i in range(21):
        px, py, pz = _get_coords(lm_list[i])
        dx, dy, dz = px - wx, py - wy, pz - wz
        coords_63.append((dx * cos_a - dy * sin_a) / palm_size)
        coords_63.append((dx * sin_a + dy * cos_a) / palm_size)
        coords_63.append(dz / palm_size)
    raw_pts = [_get_coords(lm_list[i]) for i in range(21)]
    angles_10 = [
        calculate_angle_3pts_3d(raw_pts[THUMB_CMC],  raw_pts[THUMB_MCP],  raw_pts[THUMB_IP])   / 180.0,
        calculate_angle_3pts_3d(raw_pts[THUMB_MCP],  raw_pts[THUMB_IP],   raw_pts[THUMB_TIP])  / 180.0,
        calculate_angle_3pts_3d(raw_pts[INDEX_MCP],  raw_pts[INDEX_PIP],  raw_pts[INDEX_DIP])  / 180.0,
        calculate_angle_3pts_3d(raw_pts[INDEX_PIP],  raw_pts[INDEX_DIP],  raw_pts[INDEX_TIP])  / 180.0,
        calculate_angle_3pts_3d(raw_pts[MIDDLE_MCP], raw_pts[MIDDLE_PIP], raw_pts[MIDDLE_DIP]) / 180.0,
        calculate_angle_3pts_3d(raw_pts[MIDDLE_PIP], raw_pts[MIDDLE_DIP], raw_pts[MIDDLE_TIP]) / 180.0,
        calculate_angle_3pts_3d(raw_pts[RING_MCP],   raw_pts[RING_PIP],   raw_pts[RING_DIP])   / 180.0,
        calculate_angle_3pts_3d(raw_pts[RING_PIP],   raw_pts[RING_DIP],   raw_pts[RING_TIP])   / 180.0,
        calculate_angle_3pts_3d(raw_pts[PINKY_MCP],  raw_pts[PINKY_PIP],  raw_pts[PINKY_DIP])  / 180.0,
        calculate_angle_3pts_3d(raw_pts[PINKY_PIP],  raw_pts[PINKY_DIP],  raw_pts[PINKY_TIP])  / 180.0,
    ]
    if not velocity:
        vx_n, vy_n, vz_n = 0.0, 0.0, 0.0
    else:
        vx = float(velocity[0]) if len(velocity) > 0 else 0.0
        vy = float(velocity[1]) if len(velocity) > 1 else 0.0
        vz = float(velocity[2]) if len(velocity) > 2 else 0.0
        # Xoay vector vận tốc theo ma trận [-theta] của hệ quy chiếu Canonical
        vx_n = float((vx * cos_a - vy * sin_a) / palm_size)
        vy_n = float((vx * sin_a + vy * cos_a) / palm_size)
        vz_n = float(vz / palm_size)
    return [float(v) for v in coords_63] + angles_10 + [vx_n, vy_n, vz_n]


def extract_157d_features(
    hands_list: List[List[Any]],
    velocities: Optional[List[Tuple]] = None,
) -> List[float]:
    """Extract complete 157D feature vector (1 or 2 hands).

    Returns: feat_hand1(76D) + feat_hand2(76D, zeros if absent) + inter_hand(5D)
    """
    if velocities is None:
        velocities = []
    vel = []
    for v in velocities:
        if v and len(v) >= 3:
            vel.append((float(v[0]), float(v[1]), float(v[2])))
        elif v and len(v) == 2:
            vel.append((float(v[0]), float(v[1]), 0.0))
        else:
            vel.append((0.0, 0.0, 0.0))
    while len(vel) < 2:
        vel.append((0.0, 0.0, 0.0))

    if not hands_list:
        return [0.0] * 157

    h1_valid = hands_list[0] is not None and len(hands_list[0]) >= 21
    feat1 = extract_single_hand_features(hands_list[0], vel[0]) if h1_valid else [0.0] * 76

    h2_valid = len(hands_list) >= 2 and hands_list[1] is not None and len(hands_list[1]) >= 21
    feat2 = extract_single_hand_features(hands_list[1], vel[1]) if h2_valid else [0.0] * 76

    if h1_valid and h2_valid:
        w1x, w1y, w1z = _get_coords(hands_list[0][WRIST])
        w2x, w2y, w2z = _get_coords(hands_list[1][WRIST])
        th1, ps1 = get_palm_rotation_and_size(hands_list[0])
        th2, ps2 = get_palm_rotation_and_size(hands_list[1])
        avg = (ps1 + ps2) / 2.0
        rel_dist = math.sqrt((w2x-w1x)**2 + (w2y-w1y)**2 + (w2z-w1z)**2) / avg

        # Xoay vector chênh lệch vị trí 2 cổ tay theo góc lòng bàn tay 1 [-th1]
        raw_dx = (w2x - w1x) / avg
        raw_dy = (w2y - w1y) / avg
        cos_th1 = math.cos(-th1)
        sin_th1 = math.sin(-th1)
        rel_dx = raw_dx * cos_th1 - raw_dy * sin_th1
        rel_dy = raw_dx * sin_th1 + raw_dy * cos_th1
        rel_dz = (w2z - w1z) / avg

        diff_th = th2 - th1
        while diff_th > math.pi: diff_th -= 2 * math.pi
        while diff_th <= -math.pi: diff_th += 2 * math.pi
        inter5 = [float(rel_dist), float(rel_dx), float(rel_dy), float(rel_dz), float(diff_th / math.pi)]
    else:
        inter5 = [0.0] * 5

    return feat1 + feat2 + inter5


def extract_158d_features(
    hands_list: List[List[Any]],
    velocities: Optional[List[Tuple]] = None,
    mode_id: int = 0,
) -> List[float]:
    """Extract complete 158D feature vector with Context Gate (1 or 2 hands).

    Structure: feat_hand1(76D) + feat_hand2(76D) + inter_hand(5D) + mode_id_norm(1D) = 158D
    The 158-th dimension encodes the active operational mode normalized by MAX_MODES.
    """
    feat_157 = extract_157d_features(hands_list, velocities)
    max_modes = 4.0
    mode_map = {}
    try:
        import config
        max_modes = float(getattr(config, "MAX_MODES", 4))
        mode_map = getattr(config, "MODE_MAP", {})
    except Exception:
        pass
    if max_modes <= 0.0:
        max_modes = 4.0

    if isinstance(mode_id, str):
        mode_upper = mode_id.strip().upper()
        if mode_upper in mode_map:
            m_id = float(mode_map[mode_upper])
        else:
            try:
                m_id = float(mode_id)
            except (ValueError, TypeError):
                m_id = 0.0
    else:
        try:
            m_id = float(mode_id) if mode_id is not None else 0.0
        except (ValueError, TypeError):
            m_id = 0.0

    mode_norm = max(0.0, min(1.0, m_id / max_modes))
    return feat_157 + [mode_norm]


def extract_features(
    hands_list: List[List[Any]],
    velocities: Optional[List[Tuple]] = None,
    mode_id: int = 0,
) -> List[float]:
    """Main feature extraction interface: returns 158D vector with Context Gate."""
    return extract_158d_features(hands_list, velocities, mode_id=mode_id)


# Primary alias matching plan specification: feature_extractor.extract(...)
extract = extract_features


def extract_112d_features(
    hands_list: List[List[Any]],
    velocities: Optional[List[Tuple]] = None,
) -> List[float]:
    """Backward-compat alias -> returns 157D."""
    return extract_157d_features(hands_list, velocities)

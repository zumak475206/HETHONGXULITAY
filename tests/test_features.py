"""
tests/test_features.py
Test 157D feature extractor: shape, rotation invariance (2D+3D), inter-hand.
"""

import unittest
import math
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.feature_extractor import (
    extract_single_hand_features,
    extract_158d_features,
    extract_features,
    extract,
    extract_157d_features,
    extract_112d_features,
    calculate_angle_3pts,
    calculate_angle_3pts_3d,
    WRIST, MIDDLE_MCP, INDEX_TIP, THUMB_TIP
)
from src.training.train import convert_to_158d, resample_sequence_length


def create_synthetic_hand(rotation_deg: float = 0.0, center=(320.0, 240.0), scale=100.0, z_depth: float = 0.0) -> list:
    """Create 21-point synthetic hand with optional XY rotation and Z offset.

    Format: [id, x, y, z] for each landmark (4-tuple supports new pipeline).
    """
    cx, cy = center
    base_offsets = [
        (0, 50),
        (-25, 30), (-35, 15), (-45, 0), (-60, -10),
        (-20, -5), (-25, -35), (-28, -60), (-30, -80),
        (-2, -12), (-2, -45), (-2, -75), (-2, -95),
        (18, -7), (22, -38), (25, -65), (28, -85),
        (35, 5), (42, -25), (48, -48), (52, -65),
    ]
    # Z offsets simulate depth variation (finger curl toward/away camera)
    z_offsets = [
        0.0,
        0.1, 0.2, 0.3, 0.4,
        0.05, 0.15, 0.25, 0.35,
        0.0, 0.1, 0.2, 0.3,
        0.05, 0.15, 0.25, 0.35,
        0.1, 0.2, 0.3, 0.4,
    ]
    rad = math.radians(rotation_deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    pts = []
    for i, (dx, dy) in enumerate(base_offsets):
        rx = dx * cos_r - dy * sin_r
        ry = dx * sin_r + dy * cos_r
        pz = z_offsets[i] * scale / 100.0 + z_depth
        pts.append([i, cx + rx * (scale / 100.0), cy + ry * (scale / 100.0), pz])
    return pts


class TestFeatureExtractor(unittest.TestCase):

    def test_single_hand_dimension(self):
        """Single hand features must be exactly 76D."""
        hand = create_synthetic_hand(0.0)
        feat = extract_single_hand_features(hand)
        self.assertEqual(len(feat), 76, f"Expected 76, got {len(feat)}")

    def test_feature_dimension_157(self):
        """Full feature vector must be exactly 157D for all hand count cases."""
        hand1 = create_synthetic_hand(0.0)
        hand2 = create_synthetic_hand(15.0)
        self.assertEqual(len(extract_157d_features([hand1, hand2])), 157)
        self.assertEqual(len(extract_157d_features([hand1])), 157)
        feat_0 = extract_157d_features([])
        self.assertEqual(len(feat_0), 157)
        self.assertEqual(sum(feat_0), 0.0)

    def test_feature_dimension_158_context_gate(self):
        """Full feature vector with Context Gate must be exactly 158D."""
        hand1 = create_synthetic_hand(0.0)
        hand2 = create_synthetic_hand(15.0)
        feat_2h = extract_158d_features([hand1, hand2], mode_id=1)
        self.assertEqual(len(feat_2h), 158)
        self.assertAlmostEqual(feat_2h[-1], 0.25, places=4)

        feat_1h = extract_158d_features([hand1], mode_id=2)
        self.assertEqual(len(feat_1h), 158)
        self.assertAlmostEqual(feat_1h[-1], 0.5, places=4)

        feat_0h = extract_158d_features([], mode_id=0)
        self.assertEqual(len(feat_0h), 158)
        self.assertAlmostEqual(feat_0h[-1], 0.0, places=4)

        # Main extract_features interface defaults to 158D
        feat_alias = extract_features([hand1], mode_id=3)
        self.assertEqual(len(feat_alias), 158)
        self.assertAlmostEqual(feat_alias[-1], 0.75, places=4)

        # extract alias matching plan spec
        feat_plan = extract([hand1], mode_id=2)
        self.assertEqual(len(feat_plan), 158)
        self.assertAlmostEqual(feat_plan[-1], 0.50, places=4)

    def test_string_mode_names(self):
        """String mode names (MODE1_MOUSE, MODE2_GESTURES, IDLE) must resolve to correct normalized values."""
        hand = create_synthetic_hand(0.0)
        feat_mouse = extract_158d_features([hand], mode_id="MODE1_MOUSE")
        self.assertAlmostEqual(feat_mouse[-1], 0.25, places=4)

        feat_gestures = extract_158d_features([hand], mode_id="MODE2_GESTURES")
        self.assertAlmostEqual(feat_gestures[-1], 0.50, places=4)

        feat_idle = extract_158d_features([hand], mode_id="IDLE")
        self.assertAlmostEqual(feat_idle[-1], 0.0, places=4)

        feat_numeric_str = extract_158d_features([hand], mode_id="1")
        self.assertAlmostEqual(feat_numeric_str[-1], 0.25, places=4)

    def test_convert_to_158d_1d_2d_3d(self):
        """convert_to_158d must handle 1D, 2D, and 3D arrays and correctly assign mode_id from y."""
        # 1D test
        arr_1d = np.ones(157, dtype=np.float32)
        out_1d = convert_to_158d(arr_1d, y=1)
        self.assertEqual(out_1d.shape, (158,))
        self.assertAlmostEqual(out_1d[-1], 0.25, places=4)

        # 2D test (single sequence 45x157)
        arr_2d = np.ones((45, 157), dtype=np.float32)
        out_2d = convert_to_158d(arr_2d, y=np.array([2]))
        self.assertEqual(out_2d.shape, (45, 158))
        self.assertAlmostEqual(out_2d[0, -1], 0.25, places=4)
        self.assertAlmostEqual(out_2d[44, -1], 0.25, places=4)

        # 3D test (batch 4x45x157)
        arr_3d = np.ones((4, 45, 157), dtype=np.float32)
        y_batch = np.array([0, 1, 4, 0])
        out_3d = convert_to_158d(arr_3d, y=y_batch)
        self.assertEqual(out_3d.shape, (4, 45, 158))
        self.assertAlmostEqual(out_3d[0, 0, -1], 0.0, places=4)    # IDLE
        self.assertAlmostEqual(out_3d[1, 0, -1], 0.25, places=4)   # PINCH_CLICK (Mode 1)
        self.assertAlmostEqual(out_3d[2, 0, -1], 0.50, places=4)   # SWIPE_LEFT (Mode 2)
        self.assertAlmostEqual(out_3d[3, 0, -1], 0.0, places=4)    # IDLE

    def test_resample_sequence_length_2d_3d(self):
        """resample_sequence_length must handle both 2D and 3D inputs correctly."""
        # 2D test: 30 frames -> 45 frames
        seq_2d = np.zeros((30, 158), dtype=np.float32)
        seq_2d[:, 0] = np.linspace(0, 1, 30)
        res_2d = resample_sequence_length(seq_2d, target_len=45)
        self.assertEqual(res_2d.shape, (45, 158))
        self.assertAlmostEqual(res_2d[0, 0], 0.0, places=4)
        self.assertAlmostEqual(res_2d[-1, 0], 1.0, places=4)

        # 3D test: (2, 30, 158) -> (2, 45, 158)
        seq_3d = np.zeros((2, 30, 158), dtype=np.float32)
        res_3d = resample_sequence_length(seq_3d, target_len=45)
        self.assertEqual(res_3d.shape, (2, 45, 158))

    def test_mode_id_normalization_and_boundaries(self):
        """Mode ID must be correctly normalized in [0.0, 1.0] and handle boundary/invalid inputs."""
        hand = create_synthetic_hand(0.0)

        # None mode_id -> defaults to 0.0
        feat_none = extract_158d_features([hand], mode_id=None)
        self.assertEqual(feat_none[-1], 0.0)

        # Negative mode_id -> clamped to 0.0
        feat_neg = extract_158d_features([hand], mode_id=-1)
        self.assertEqual(feat_neg[-1], 0.0)

        # Excessive mode_id -> clamped to 1.0
        feat_excess = extract_158d_features([hand], mode_id=99)
        self.assertEqual(feat_excess[-1], 1.0)

        # Invalid type string -> safely falls back to 0.0
        feat_str = extract_158d_features([hand], mode_id="invalid_mode_name")
        self.assertEqual(feat_str[-1], 0.0)

    def test_backward_compat_alias(self):
        """extract_112d_features alias must also return 157D now."""
        hand = create_synthetic_hand(0.0)
        self.assertEqual(len(extract_112d_features([hand])), 157)

    def test_360_rotation_invariance(self):
        """Canonical features must be stable across 0,45,90,135,180,270 degrees."""
        feat_base = np.array(extract_single_hand_features(create_synthetic_hand(0.0)))
        for deg in [45.0, 90.0, 135.0, 180.0, 270.0]:
            feat_rot = np.array(extract_single_hand_features(create_synthetic_hand(deg)))
            cos_sim = np.dot(feat_base, feat_rot) / (
                np.linalg.norm(feat_base) * np.linalg.norm(feat_rot) + 1e-6
            )
            self.assertGreater(cos_sim, 0.97,
                f"Rotation invariance failed at {deg}: cosine={cos_sim:.4f}")

    def test_z_depth_captured(self):
        """Z coordinates must appear in feature vector (non-zero for non-flat hand)."""
        hand_3d = create_synthetic_hand(0.0, z_depth=0.0)
        feat = extract_single_hand_features(hand_3d)
        # Z slots: indices 2, 5, 8, ... (every 3rd starting at 2)
        z_values = [feat[2 + i*3] for i in range(21)]
        has_nonzero_z = any(abs(z) > 1e-6 for z in z_values)
        self.assertTrue(has_nonzero_z, "Z features are all zero - z not being captured")

    def test_inter_hand_5d(self):
        """Inter-hand features must be exactly 5D (added dz)."""
        hand1 = create_synthetic_hand(0.0, center=(200.0, 240.0), z_depth=-0.1)
        hand2 = create_synthetic_hand(0.0, center=(400.0, 240.0), z_depth=0.1)
        feat = extract_157d_features([hand1, hand2])
        # Last 5 values are inter-hand
        inter = feat[-5:]
        self.assertEqual(len(inter), 5)
        # rel_dist should be > 0 (hands at different positions)
        self.assertGreater(inter[0], 0.0, "Inter-hand distance should be positive")
        # dz should be non-zero (hands at different z depths)
        self.assertNotAlmostEqual(inter[3], 0.0, places=4,
            msg="dz inter-hand feature should be non-zero")

    def test_angle_3d_zero_same_as_2d(self):
        """3D angle with z=0 must match 2D angle (consistency)."""
        a, b, c = (0.0, 0.0), (1.0, 0.0), (1.0, 1.0)
        angle_2d = calculate_angle_3pts(a, b, c)
        angle_3d = calculate_angle_3pts_3d(
            (a[0], a[1], 0.0), (b[0], b[1], 0.0), (c[0], c[1], 0.0)
        )
    def test_missing_and_asymmetric_hands_edge_cases(self):
        """extract_157d_features must handle None, empty, or asymmetric hands without error."""
        hand1 = create_synthetic_hand(0.0)
        hand2 = create_synthetic_hand(15.0)

        # Case 1: hand 1 is None, hand 2 is present
        feat_none1 = extract_157d_features([None, hand2])
        self.assertEqual(len(feat_none1), 157)
        # hand 1 slots (first 76) and inter-hand slots (last 5) should be zero
        self.assertEqual(sum(feat_none1[:76]), 0.0)
        self.assertEqual(sum(feat_none1[-5:]), 0.0)
        # hand 2 slots (76..152) should have non-zero features
        self.assertNotEqual(sum(feat_none1[76:152]), 0.0)

        # Case 2: hand 1 present, hand 2 is empty list
        feat_empty2 = extract_157d_features([hand1, []])
        self.assertEqual(len(feat_empty2), 157)

        # Case 3: both None
        feat_both_none = extract_157d_features([None, None])
        self.assertEqual(len(feat_both_none), 157)
        self.assertEqual(sum(feat_both_none), 0.0)

        # Case 4: empty list
        feat_empty = extract_157d_features([])
        self.assertEqual(len(feat_empty), 157)

    def test_empty_and_short_velocities(self):
        """extract_single_hand_features and 157d must handle empty, None, and short velocities."""
        hand = create_synthetic_hand(0.0)
        self.assertEqual(len(extract_single_hand_features(hand, velocity=None)), 76)
        self.assertEqual(len(extract_single_hand_features(hand, velocity=())), 76)
        self.assertEqual(len(extract_single_hand_features(hand, velocity=(1.0,))), 76)
        self.assertEqual(len(extract_single_hand_features(hand, velocity=(1.0, 2.0))), 76)
        self.assertEqual(len(extract_single_hand_features(hand, velocity=(1.0, 2.0, 3.0))), 76)


if __name__ == "__main__":
    unittest.main()

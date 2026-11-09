# -*- coding: utf-8 -*-
"""M3 单元自检：不依赖 M1/M2/M4，只验证 calc_goertzel_spectrum.py。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calc_goertzel_spectrum import (  # noqa: E402
    NOTE_NAMES,
    TARGET_FREQS,
    build_templates,
    calc_goertzel_spectrum,
)

SR = 44100


class TestM3(unittest.TestCase):
    def setUp(self):
        t = np.arange(int(0.2 * SR)) / SR
        self.x = np.sin(2 * np.pi * 261.6256 * t)   # C4

    def test_shape_and_norm(self):
        S = calc_goertzel_spectrum(self.x, SR, TARGET_FREQS)
        self.assertEqual(S.shape, (26,))
        self.assertEqual(S.dtype, np.float64)
        self.assertAlmostEqual(float(np.linalg.norm(S)), 1.0, places=12)
        self.assertTrue(np.all(S >= 0))

    def test_silence(self):
        S = calc_goertzel_spectrum(np.zeros(1000), SR, TARGET_FREQS)
        self.assertEqual(S.shape, (26,))
        self.assertTrue(np.all(S == 0))

    def test_invalid_signal(self):
        bad_inputs = (
            np.zeros(0),
            np.zeros((10, 2)),
            np.array([1.0, np.nan]),
        )
        for bad in bad_inputs:
            with self.assertRaises(ValueError):
                calc_goertzel_spectrum(bad, SR, TARGET_FREQS)

    def test_invalid_freq(self):
        for bad in ([0.0], [22050.0], [-1.0]):
            with self.assertRaises(ValueError):
                calc_goertzel_spectrum(np.ones(100), SR, bad)

    def test_templates(self):
        T = build_templates([self.x] * len(NOTE_NAMES), SR, TARGET_FREQS)
        self.assertEqual(T.shape, (26, 26))
        np.testing.assert_allclose(
            np.linalg.norm(T, axis=1), 1.0, rtol=1e-12, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()

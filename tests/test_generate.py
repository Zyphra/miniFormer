"""Checks for a fixed, class-independent mask on the dominant-frequency boost."""

import unittest

import numpy as np

from synthetic_data.sinusoid_amplitude.generate import make_dataset


class ChannelBoostTests(unittest.TestCase):
    def test_only_selected_channels_receive_the_class_boost(self):
        data = make_dataset(samples_per_class=3, frequency_spacing=0.2,
                            noise_std=0, amplitude_jitter=0, boost_channels=["C3", "Cz", "C4"])
        mask = data["boost_channel_mask"]
        np.testing.assert_array_equal(data["boost_channels"], ["C3", "Cz", "C4"])
        # FFT-aligned, noiseless tones have power A^2/2 at each frequency.
        amplitudes = np.full(data["channel_power"].shape, 0.5)
        amplitudes += 1.5 * mask[None, :, None] * np.eye(4)[data["y"], None, :]
        expected = (amplitudes * data["channel_gains"][None, :, None]) ** 2 / 2
        np.testing.assert_allclose(data["channel_power"], expected, rtol=1e-6, atol=1e-7)

    def test_mask_preserves_randomness_and_selected_recordings(self):
        full = make_dataset(samples_per_class=3)
        explicit = make_dataset(samples_per_class=3, boost_channels=full["channel_names"].tolist())
        subset = make_dataset(samples_per_class=3, boost_channels=["C3", "Cz", "C4"])
        for key in full:
            np.testing.assert_array_equal(full[key], explicit[key])
        mask = subset["boost_channel_mask"]
        np.testing.assert_array_equal(full["y"], subset["y"])
        np.testing.assert_array_equal(full["X"][:, mask], subset["X"][:, mask])
        expected_difference = full["channel_gains"][None, ~mask, None] * full["boost_signal"][:, None, :]
        np.testing.assert_allclose(full["X"][:, ~mask] - subset["X"][:, ~mask],
                                   np.broadcast_to(expected_difference, full["X"][:, ~mask].shape),
                                   rtol=1e-5, atol=1e-6)

    def test_invalid_channel_sets_fail(self):
        for names in ([], ["Cz", "Cz"], ["unknown"], "Cz"):
            with self.subTest(names=names), self.assertRaises(AssertionError):
                make_dataset(samples_per_class=2, boost_channels=names)


if __name__ == "__main__":
    unittest.main()

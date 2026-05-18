from __future__ import annotations

import unittest
import wave

from resources.paths import (
    example_wav_path,
    mosnet_model_path,
    packaged_dataset_audio_paths,
    packaged_dataset_manifest,
    resource_files,
)


class ResourceTests(unittest.TestCase):
    def test_example_wav_is_available(self) -> None:
        self.assertTrue(resource_files("audio", "example.wav").is_file())

        with example_wav_path() as wav_path:
            with wave.open(str(wav_path), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 16000)
                self.assertGreater(wav_file.getnframes(), 0)

    def test_packaged_dataset_paths_match_manifest(self) -> None:
        manifest = packaged_dataset_manifest()
        self.assertEqual(set(manifest), {"vctk", "librispeech"})

        with packaged_dataset_audio_paths() as dataset_paths:
            self.assertEqual(set(dataset_paths), {"vctk", "librispeech"})
            self.assertEqual(len(dataset_paths["vctk"]), 10)
            self.assertEqual(len(dataset_paths["librispeech"]), 11)
            for group_paths in dataset_paths.values():
                for path in group_paths:
                    self.assertTrue(path.exists())

    def test_mosnet_model_is_available_as_real_path(self) -> None:
        with mosnet_model_path() as model_path:
            self.assertEqual(model_path.name, "cnn_blstm.h5")
            self.assertTrue(model_path.exists())


if __name__ == "__main__":
    unittest.main()

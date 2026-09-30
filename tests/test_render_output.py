from concurrent.futures import Future
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from test_release_contracts import source_functions


ROOT = Path(__file__).resolve().parents[1]


class RenderOutputTests(unittest.TestCase):
    def run_render(self, root, *, error=None, bad_rgb=False, video=False, write=True, views=True):
        futures = []
        waited = []

        class CheckedFuture(Future):
            def result(self, *args, **kwargs):
                waited.append(self)
                return super().result(*args, **kwargs)

        def save(image, path):
            future = CheckedFuture()
            if error:
                future.set_exception(error)
            else:
                future.set_result(None)
            futures.append(future)
            return future

        def make_video(folder, path, **kwargs):
            self.assertEqual(len(waited), len(futures))
            self.assertEqual(Path(path).parent, Path(root) / "test/ours_100000/renders")

        videos = Mock(side_effect=make_video)
        base = torch.full((3, 4, 4), float("nan") if bad_rgb else 0.25)
        namespace = {"os": os, "makedirs": os.makedirs, "torch": torch,
                     "create_render_streams": lambda: (None, None),
                     "tqdm": lambda values, **kwargs: values,
                     "render": Mock(return_value={"render": base, "shadow": torch.ones(1, 4, 4),
                                                  "other_effects": torch.zeros(3, 4, 4)}),
                     "save_render_image": save, "images_to_video": videos,
                     "print": lambda *args: None}
        source_functions("render.py", ["render_set"], namespace)
        gaussians = SimpleNamespace(get_local_axis=None,
                                    asg_func=SimpleNamespace(get_asg_lam_miu=None, get_asg_axis=None))
        cameras = [SimpleNamespace(original_image=torch.zeros(3, 4, 4))] if views else []
        namespace["render_set"](SimpleNamespace(model_path=str(root)), "test", 100000,
                                cameras, gaussians, None, None, False, False,
                                write_images=write, synthesize_video=video)
        return futures, waited, videos

    def test_image_writes_finish_before_video_and_function_return(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            futures, waited, videos = self.run_render(temp, video=True)
            self.assertEqual(len(futures), 5)
            self.assertEqual(len(waited), 5)
            self.assertEqual(videos.call_count, 5)

    def test_background_writer_errors_are_propagated(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, self.assertRaisesRegex(OSError, "disk full"):
            self.run_render(temp, error=OSError("disk full"))

    def test_nonfinite_final_images_are_not_saved(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, self.assertRaises(FloatingPointError):
            self.run_render(temp, bad_rgb=True)

    def test_empty_views_and_video_without_images_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            with self.assertRaisesRegex(ValueError, "No cameras"):
                self.run_render(temp, views=False)
            with self.assertRaisesRegex(ValueError, "requires --write_images"):
                self.run_render(temp, video=True, write=False)


if __name__ == "__main__":
    unittest.main()

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

from vedio.cineparse import (CineParse, CineParseConfig, LABELS, apply_cineparse,
                            collate_windows, read_window, save_checkpoint, load_checkpoint, PREPROCESS)
from vedio.train_cineparse import load_manifests, supervised_loss


def tiny_backbone():
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 768))
    return model.requires_grad_(False)


class CineParseTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def model(self):
        return CineParse(tiny_backbone(), CineParseConfig(adapter_dim=8, temporal_hidden=4))

    def inputs(self):
        return torch.randn(2, 4, 3, 2, 2), [2, 4], torch.tensor([[0., .5, 0., 0.], [0., .5, 1., 1.5]])

    def test_padding_does_not_change_predictions_and_backward_freezes_encoder(self):
        model = self.model().train()
        frames, lengths, times = self.inputs()
        output = model(frames, lengths, times)
        modified = frames.clone()
        modified[0, 2:] = float("nan")
        changed = model(modified, lengths, times)
        torch.testing.assert_close(output["prior_5d"], changed["prior_5d"], rtol=0, atol=0)
        single = model(frames[:1, :2], [2], times[:1, :2])
        torch.testing.assert_close(single["prior_5d"][0], output["prior_5d"][0])
        before = {k: p.clone() for k, p in model.backbone.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        loss, _, _ = supervised_loss(output, torch.zeros(2, 5), torch.zeros(2, 4), output["mask"])
        loss.backward()
        self.assertGreater(model.prior_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.hit_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.temporal.weight_ih_l0.grad.abs().sum().item(), 0)
        optimizer.step()
        for key, p in model.backbone.named_parameters():
            self.assertIsNone(p.grad)
            self.assertTrue(torch.equal(p, before[key]))

    def test_unannotated_hits_have_no_gradient(self):
        model = self.model()
        output = model(*self.inputs())
        loss, _, hit_loss = supervised_loss(output, torch.zeros(2, 5), torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.bool))
        loss.backward()
        self.assertEqual(hit_loss.item(), 0)
        self.assertIsNone(model.hit_head.weight.grad)

    def test_timestamps_and_config_validation(self):
        with self.assertRaises(ValueError):
            CineParseConfig(sample_fps=float("nan"))
        with self.assertRaises(ValueError):
            CineParseConfig(trainable_backbone_blocks=-1)
        model = self.model()
        frames, lengths, times = self.inputs()
        times[1, 2] = times[1, 1]
        with self.assertRaisesRegex(ValueError, "increase"):
            model(frames, lengths, times)

    def test_checkpoint_roundtrip_and_integrity(self):
        from experiments.provenance import sha256_file
        model = self.model().eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.pt"
            torch.save(model.backbone.state_dict(), base)
            provenance = {"path": str(base), "sha256": sha256_file(base), "preprocess": PREPROCESS}
            checkpoint = root / "model.pt"
            with self.assertRaisesRegex(ValueError, "untrained"):
                save_checkpoint(checkpoint, model, provenance, steps=0)
            save_checkpoint(checkpoint, model, provenance, steps=1)
            def make_base(*_):
                backbone = tiny_backbone()
                backbone.load_state_dict(torch.load(base, weights_only=True))
                return backbone
            with patch("vedio.cineparse.load_backbone", side_effect=make_base):
                loaded, metadata = load_checkpoint(checkpoint)
            inputs = self.inputs()
            torch.testing.assert_close(model(*inputs)["prior_5d"], loaded(*inputs)["prior_5d"], atol=0, rtol=0)
            self.assertEqual(metadata["steps"], 1)
            base.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_checkpoint(checkpoint)

    def test_replacing_priors_preserves_measured_evidence(self):
        pack = {"video_path": "unused", "extractor": {"backend": "opencv"},
                "summary": {"duration": 2., "means": dict.fromkeys(LABELS, .2)},
                "frame_features": {"times": [0., .5], "cut_flags": [0, 1]},
                "prior_5d": [{"start_time": 0., "end_time": 2., **dict.fromkeys(LABELS, .2)}]}
        original = copy.deepcopy(pack)
        model = self.model().eval()
        window = (torch.randn(4, 3, 2, 2), torch.arange(4, dtype=torch.float64) * .5)
        with patch("vedio.cineparse.load_checkpoint", return_value=(model, {"rhythm_steps": 0})), \
             patch("vedio.cineparse.read_window", return_value=window):
            output = apply_cineparse(pack, "checkpoint.pt")
        self.assertEqual(pack, original)
        self.assertEqual(output["frame_features"], pack["frame_features"])
        self.assertEqual(output["measured_prior_5d"], pack["prior_5d"])
        self.assertIsNone(output["learned_window_predictions"][0]["hit_probability"])
        self.assertNotEqual(output["prior_5d"], pack["prior_5d"])

    def test_decoding_uses_actual_timestamps_and_enforces_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 7., (32, 24))
            self.assertTrue(writer.isOpened())
            for index in range(14):
                writer.write(np.full((24, 32, 3), 10 * index, dtype=np.uint8))
            writer.release()
            frames, times = read_window(path, 0., 2., CineParseConfig())
            self.assertEqual(frames.shape, (4, 3, 224, 224))
            np.testing.assert_allclose(times.numpy(), [0., 4/7, 1., 11/7])
            with self.assertRaisesRegex(ValueError, "beyond"):
                read_window(path, 0., 3., CineParseConfig())
            with self.assertRaisesRegex(ValueError, "max_frames"):
                read_window(path, 0., 2., CineParseConfig(max_frames=2))

    def test_manifest_leakage_is_checked_by_content_and_movie(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first.avi").write_bytes(b"video-one")
            (root / "second.avi").write_bytes(b"video-two")
            row = {"sample_id": "s1", "split": "train", "dataset": "synthetic", "movie_id": "movie1",
                   "video": "first.avi", "start_time": 0., "end_time": 1., "prior_5d": [.2] * 5,
                   "label_source": "synthetic_fixture", "label_protocol": "test_only", "hit_times": [.5]}
            train, validation = root / "train.jsonl", root / "validation.jsonl"
            train.write_text(json.dumps(row))
            other = dict(row, sample_id="s2", split="validation", video="second.avi", movie_id="movie2")
            validation.write_text(json.dumps(other))
            self.assertEqual(len(load_manifests(train, validation)["validation"]), 1)
            validation.write_text(json.dumps(dict(other, movie_id="movie1")))
            with self.assertRaisesRegex(ValueError, "movie_id crosses"):
                load_manifests(train, validation)
            validation.write_text(json.dumps(other))
            (root / "second.avi").write_bytes(b"video-one")
            with self.assertRaisesRegex(ValueError, "video_sha256 crosses"):
                load_manifests(train, validation)
            train.write_text(json.dumps(dict(row, split="test")))
            with self.assertRaisesRegex(ValueError, "requires split=train"):
                load_manifests(train)

    def test_bad_labels_hits_and_hashes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "video").write_bytes(b"video")
            row = {"sample_id": "s1", "split": "train", "dataset": "synthetic", "movie_id": "movie1",
                   "video": "video", "start_time": 0., "end_time": 1., "prior_5d": [.2] * 5,
                   "label_source": "synthetic", "label_protocol": "test_only"}
            path = root / "train.jsonl"
            for change in ({"prior_5d": [float("nan")] * 5}, {"hit_times": [1.]},
                           {"video_sha256": "wrong"}, {"label_source": ""}):
                path.write_text(json.dumps(dict(row, **change)))
                with self.assertRaises(ValueError):
                    load_manifests(path)

    def test_pipeline_forwards_trained_visual_options(self):
        import run_coma_pipeline as pipeline
        with patch.object(pipeline, "extract_prior_5d", return_value={"prior_5d": []}) as extract, \
             patch.object(pipeline, "run_pipeline_from_prior") as continuation:
            pipeline.run_pipeline(video_path="example.avi", output_dir="unused", segment_duration=2.,
                segment_overlap=.5, sample_fps=2., cut_threshold=.35, refine_with_rl=False,
                refine_with_hierarchical_rl=False, rl_episodes=0, enable_repair_cycle=False,
                critic_feedback_mode="rule", human_feedback=None, llm_feedback=None,
                cineparse_checkpoint="trained.pt", cineparse_backbone_checkpoint="base.pt", cineparse_device="cpu")
            self.assertEqual(extract.call_args.kwargs["cineparse_checkpoint"], "trained.pt")
            self.assertEqual(extract.call_args.kwargs["cineparse_backbone_checkpoint"], "base.pt")
            self.assertEqual(continuation.call_args.kwargs["prior"], {"prior_5d": []})


if __name__ == "__main__":
    unittest.main()

import math
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

if __package__ is None or __package__ == "":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

try:  # pragma: no cover - only used when OpenCV is absent
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.CAP_PROP_FPS = 5
    cv2_stub.CAP_PROP_FRAME_COUNT = 7

    def _not_available(*args, **kwargs):
        raise RuntimeError("OpenCV is not available in this test environment")

    cv2_stub.VideoCapture = _not_available
    cv2_stub.cvtColor = _not_available
    cv2_stub.COLOR_BGR2RGB = 4
    sys.modules["cv2"] = cv2_stub

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from boundry_segmentation.qwen import QwenTemporalSegmenterFixed


class RealisticProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text_parts = []
        for item in messages[0]["content"]:
            if item.get("type") == "text":
                text_parts.append(item["text"])
        return "\n".join(text_parts)

    def __call__(self, text, images=None, videos=None, padding=True, return_tensors="pt"):
        return self.tokenizer(
            text,
            padding=padding,
            return_tensors=return_tensors,
        )

    def batch_decode(self, sequences, skip_special_tokens=True):
        return self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
        )


class FakeGenerateOutput:
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores


class FakeModel:
    def __init__(self, tokenizer, generated_token_id: int, score_vector: torch.Tensor):
        self.tokenizer = tokenizer
        self.generated_token_id = generated_token_id
        self.score_vector = score_vector
        self.device = torch.device("cpu")

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids)

        device = input_ids.device
        generated = torch.tensor([[self.generated_token_id]], dtype=torch.long, device=device)
        sequences = torch.cat([input_ids, generated], dim=1)
        scores = [self.score_vector.to(device).unsqueeze(0)]
        return FakeGenerateOutput(sequences, scores)


class BinaryModeConfidenceTest(unittest.TestCase):
    BACKGROUND_LOGIT = -50.0

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                trust_remote_code=True,
            )
        except Exception as exc:  # pragma: no cover - depends on external assets
            raise unittest.SkipTest(f"Qwen tokenizer unavailable: {exc}") from exc

        cls.true_token_id = cls._single_token_id([" true", "true", "True"])
        cls.false_token_id = cls._single_token_id([" false", "false", "False"])

        print("[TEST] Tokenizer type:", type(cls.tokenizer))
        print("[TEST] Resolved true token id:", cls.true_token_id)
        print("[TEST] Resolved false token id:", cls.false_token_id)

    @classmethod
    def _single_token_id(cls, candidates):
        for candidate in candidates:
            encoded = cls.tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if isinstance(encoded, list) and len(encoded) == 1:
                return encoded[0]
        raise unittest.SkipTest("Tokenizer could not map true/false to single tokens")

    def _run_segmenter(self, generated_token_id, score_vector):
        processor = RealisticProcessor(self.tokenizer)
        model = FakeModel(self.tokenizer, generated_token_id, score_vector)

        with patch("boundry_segmentation.qwen.AutoModelForVision2Seq.from_pretrained", return_value=model), \
             patch("boundry_segmentation.qwen.AutoProcessor.from_pretrained", return_value=processor), \
             patch("boundry_segmentation.qwen.process_vision_info", return_value=([], None)):
            segmenter = QwenTemporalSegmenterFixed(
                model_name="stub",
                prompt_type="small",
                response_mode="binary",
            )
            self.assertEqual(segmenter.true_token_ids, [self.true_token_id])
            self.assertEqual(segmenter.false_token_ids, [self.false_token_id])
            frames = [Image.new("RGB", (2, 2), color="white") for _ in range(2)]
            return segmenter.ask_boundary_native(frames, 0, 1)

    def test_confidence_prefers_true_token(self):
        scores = torch.full((self.tokenizer.vocab_size,), self.BACKGROUND_LOGIT, dtype=torch.float32)
        scores[self.true_token_id] = 5.0
        scores[self.false_token_id] = 0.0

        result = self._run_segmenter(self.true_token_id, scores)

        probs = torch.softmax(scores, dim=-1)
        expected_prob = float(probs[self.true_token_id].item())
        expected_prob_false = float(probs[self.false_token_id].item())
        expected_margin = float(scores[self.true_token_id].item() - scores[self.false_token_id].item())
        expected_entropy = float(-expected_prob * math.log(expected_prob) - expected_prob_false * math.log(expected_prob_false))

        self.assertTrue(result["boundary"])
        self.assertAlmostEqual(result["confidence"], expected_prob, places=6)
        self.assertAlmostEqual(result["prob_true"], expected_prob, places=6)
        self.assertAlmostEqual(result["prob_false"], expected_prob_false, places=6)
        self.assertAlmostEqual(result["logit_true"], float(scores[self.true_token_id].item()), places=6)
        self.assertAlmostEqual(result["logit_false"], float(scores[self.false_token_id].item()), places=6)
        self.assertAlmostEqual(result["logit_margin"], expected_margin, places=6)
        self.assertAlmostEqual(result["entropy"], expected_entropy, places=6)
        self.assertEqual(result["raw_response"], "true")

    def test_confidence_prefers_false_token(self):
        scores = torch.full((self.tokenizer.vocab_size,), self.BACKGROUND_LOGIT, dtype=torch.float32)
        scores[self.true_token_id] = -5.0
        scores[self.false_token_id] = 5.0

        result = self._run_segmenter(self.false_token_id, scores)

        probs = torch.softmax(scores, dim=-1)
        expected_prob = float(probs[self.true_token_id].item())
        expected_prob_false = float(probs[self.false_token_id].item())
        expected_margin = float(scores[self.true_token_id].item() - scores[self.false_token_id].item())
        expected_entropy = float(-expected_prob * math.log(expected_prob) - expected_prob_false * math.log(expected_prob_false))

        self.assertFalse(result["boundary"])
        self.assertAlmostEqual(result["confidence"], expected_prob, places=6)
        self.assertAlmostEqual(result["prob_true"], expected_prob, places=6)
        self.assertAlmostEqual(result["prob_false"], expected_prob_false, places=6)
        self.assertAlmostEqual(result["logit_true"], float(scores[self.true_token_id].item()), places=6)
        self.assertAlmostEqual(result["logit_false"], float(scores[self.false_token_id].item()), places=6)
        self.assertAlmostEqual(result["logit_margin"], expected_margin, places=6)
        self.assertAlmostEqual(result["entropy"], expected_entropy, places=6)
        self.assertEqual(result["raw_response"], "false")

    def test_video_samples_plot(self):
        video_path = Path(__file__).resolve().parents[1] / "sherlock.mp4"
        if not video_path.exists():
            self.skipTest(f"Test video not available at {video_path}")

        processor = RealisticProcessor(self.tokenizer)
        base_scores = torch.full((self.tokenizer.vocab_size,), self.BACKGROUND_LOGIT, dtype=torch.float32)
        model = FakeModel(self.tokenizer, self.true_token_id, base_scores.clone())

        with patch("boundry_segmentation.qwen.AutoModelForVision2Seq.from_pretrained", return_value=model), \
             patch("boundry_segmentation.qwen.AutoProcessor.from_pretrained", return_value=processor), \
             patch("boundry_segmentation.qwen.process_vision_info", return_value=([], None)):
            segmenter = QwenTemporalSegmenterFixed(
                model_name="stub",
                prompt_type="small",
                response_mode="binary",
            )

            frames, frame_indices, fps, _ = segmenter.sample_frames(
                str(video_path), sample_fps=1.0
            )

            window_size = 8
            total_frames = len(frames)
            if total_frames < window_size:
                self.skipTest("Not enough frames sampled from video for the requested window size")

            sample_count = min(5, total_frames - window_size + 1)
            if sample_count <= 0:
                self.skipTest("Unable to extract the required number of samples from the video")

            if sample_count == 1:
                start_indices = [0]
            else:
                span = total_frames - window_size
                step = span / (sample_count - 1)
                start_indices = [int(round(i * step)) for i in range(sample_count)]

            left_local = window_size // 2 - 1
            right_local = window_size // 2

            probabilities = []
            times = []
            diff_scores = []

            for idx, start in enumerate(start_indices):
                window_frames = frames[start:start + window_size]

                left_frame = np.asarray(window_frames[left_local], dtype=np.float32) / 255.0
                right_frame = np.asarray(window_frames[right_local], dtype=np.float32) / 255.0
                diff_score = float(np.mean(np.abs(right_frame - left_frame)))

                # Map difference score to probability via logistic transform
                prob = 1.0 / (1.0 + math.exp(-12.0 * (diff_score - 0.12)))
                prob = min(1.0 - 1e-4, max(1e-4, prob))

                logit = math.log(prob / (1.0 - prob))
                score_vector = base_scores.clone()
                score_vector[self.true_token_id] = logit
                score_vector[self.false_token_id] = 0.0
                model.score_vector = score_vector
                model.generated_token_id = self.true_token_id if prob >= 0.5 else self.false_token_id

                result = segmenter.ask_boundary_native(window_frames, left_local, right_local)

                self.assertAlmostEqual(result["confidence"], prob, places=6)
                self.assertAlmostEqual(result["prob_true"], prob, places=6)
                self.assertAlmostEqual(result["prob_false"], 1.0 - prob, places=6)
                expected_entropy = float(-prob * math.log(prob) - (1.0 - prob) * math.log(1.0 - prob))
                self.assertAlmostEqual(result["entropy"], expected_entropy, places=6)
                self.assertAlmostEqual(result["logit_margin"], logit, places=6)
                probabilities.append(result["confidence"])
                times.append(frame_indices[start + right_local] / fps)
                diff_scores.append(diff_score)

                print(
                    f"[TEST] Sample {idx}: time={times[-1]:.2f}s diff={diff_score:.4f} prob={probabilities[-1]:.3f}"
                )

        output_dir = Path(__file__).resolve().parent / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "sherlock_binary_confidence.png"

        fig, ax = plt.subplots(figsize=(6, 4))
        x = range(len(probabilities))
        ax.plot(x, probabilities, marker="o", linewidth=1.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{t:.1f}s" for t in times])
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Sample center time")
        ax.set_ylabel("Boundary probability")
        ax.set_title("Qwen Binary Mode – Sherlock samples")

        for idx, (xp, yp) in enumerate(zip(x, probabilities)):
            ax.annotate(f"{yp:.2f}", (xp, yp), textcoords="offset points", xytext=(0, 6), ha="center")

        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

        print(f"[TEST] Saved binary confidence plot to {plot_path}")

        self.assertTrue(plot_path.exists())


if __name__ == "__main__":
    unittest.main()
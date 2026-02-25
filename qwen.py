#!/usr/bin/env python3
"""
qwen_temporal_segmentation_fixed.py
Temporal video segmentation for Qwen vision-language models.
Tested with Qwen3-VL model families.
"""

import argparse
import os
import json
import math
import re
import traceback
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.tokenization_utils_base import BatchEncoding

try:
    from .qwen_vl_utils import process_vision_info  # type: ignore  # Required for Qwen2.5-VL
except ImportError:  # pragma: no cover - fallback for script execution
    from qwen_vl_utils import process_vision_info  # type: ignore  # Required for Qwen2.5-VL
from tqdm import tqdm

# Optional HMM dependencies
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


def device_name():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class QwenTemporalSegmenterFixed:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        device: Optional[str] = None,
        prompt_type: str = "small",
        response_mode: str = "json",
        model: Optional[Any] = None,
        processor: Optional[Any] = None,
    ):
        """
        Initialize with Qwen VL model loading.
        AutoModelForVision2Seq handles architecture-specific loading.
        """
        valid_modes = {"json", "binary"}
        if response_mode not in valid_modes:
            raise ValueError(f"Unsupported response_mode '{response_mode}'. Valid options: {valid_modes}")

        if (model is None) ^ (processor is None):
            raise ValueError("model and processor must either both be provided or both be None")

        self.model_name = model_name
        self.device = device or device_name()
        self.prompt_type = prompt_type
        self.response_mode = response_mode
        print(f"[INFO] Using device: {self.device}")

        if model is not None:
            print(f"[INFO] Using provided model and processor for '{model_name}'")
            self.model = model
            self.processor = processor
        else:
            # Load the correct model class
            # AutoModelForVision2Seq supports Qwen VL variants (including Qwen3-VL).
            print(f"[INFO] Loading Qwen VL model: {model_name}")
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto"
            )

            # Load processor (handles both text and images)
            self.processor = AutoProcessor.from_pretrained(model_name)

        if not hasattr(self.model, "device"):
            self.model.device = torch.device(self.device)

        # Generation config for deterministic JSON output
        self.generation_config = {
            "max_new_tokens": 256,
            "temperature": 0.1,  # Near-deterministic
            "do_sample": False,
            "top_p": 0.1,
        }

        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self.true_token_ids = self._resolve_token_ids([" true", "true", "True"])
        self.false_token_ids = self._resolve_token_ids([" false", "false", "False"])
        if self.response_mode == "binary":
            if not self.true_token_ids:
                print("[WARN] No single-token encoding resolved for 'true'; confidence scores will default to 0.0")
            if not self.false_token_ids:
                print("[WARN] No single-token encoding resolved for 'false'; fallback parsing may be unreliable")

    def sample_frames(self, video_path: str, sample_fps: float = 2.0,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None) -> Tuple[List[Image.Image], List[int], float, Tuple[int, int]]:
        """Sample frames from video at specified fps within optional [start_time, end_time] seconds.
        Returns frames, frame_indices, video_fps, and (start_frame, end_frame).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
            
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Compute frame bounds
        start_frame = 0 if start_time is None else max(0, int(round(start_time * video_fps)))
        end_frame = (total_frames - 1) if end_time is None else min(total_frames - 1, int(round(end_time * video_fps)))
        if end_frame < start_frame:
            raise ValueError("end_time must be greater than start_time")
        
        frame_interval = max(1, int(round(video_fps / sample_fps)))
        
        frames = []
        frame_indices = []
        idx = 0
        # Seek to start_frame if needed
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            idx = start_frame
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx > end_frame:
                break
            if idx % frame_interval == 0:
                # Convert BGR to RGB
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(pil_frame)
                frame_indices.append(idx)
            idx += 1
            
        cap.release()
        print(f"[INFO] Sampled {len(frames)} frames from {total_frames} total (segment {start_frame}..{end_frame})")
        return frames, frame_indices, video_fps, (start_frame, end_frame)

    def encode_frame_to_base64(self, frame: Image.Image) -> str:
        """Convert PIL image to base64 for Qwen2.5-VL input"""
        buffered = BytesIO()
        frame.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def _resolve_token_ids(self, candidates: List[str]) -> List[int]:
        if self.tokenizer is None:
            return []

        resolved: List[int] = []
        for candidate in candidates:
            token_ids: List[int] = []
            try:
                encoded = self.tokenizer(candidate, add_special_tokens=False)
            except TypeError:
                encoded = self.tokenizer.encode(candidate, add_special_tokens=False)

            if isinstance(encoded, BatchEncoding):
                token_ids = encoded.get("input_ids", []) or []
            elif isinstance(encoded, dict):
                token_ids = encoded.get("input_ids", []) or []
            elif isinstance(encoded, (list, tuple)):
                token_ids = list(encoded)
            elif hasattr(encoded, "tolist"):
                token_ids = list(encoded.tolist())
            else:
                token_ids = []

            if isinstance(token_ids, list) and len(token_ids) == 1:
                resolved.append(int(token_ids[0]))
                break

        return resolved

    @staticmethod
    def compute_token_probability(
        scores: torch.Tensor,
        token_ids: List[int],
        other_token_ids: Optional[List[int]] = None,
        mode: str = "pair",
    ) -> float:
        if scores.ndim > 1:
            scores = scores.squeeze(0)

        if not token_ids or scores.numel() == 0:
            return 0.0

        vocab_size = scores.shape[-1]
        primary_id = None
        for token_id in token_ids:
            if 0 <= token_id < vocab_size:
                primary_id = token_id
                break

        if primary_id is None:
            return 0.0

        if mode == "full":
            probs = F.softmax(scores, dim=-1)
            return float(probs[primary_id].item())

        compare_ids = [primary_id]
        if other_token_ids:
            for token_id in other_token_ids:
                if 0 <= token_id < vocab_size and token_id != primary_id:
                    compare_ids.append(token_id)
                    break

        selected = scores.new_tensor([scores[idx] for idx in compare_ids])
        probs = F.softmax(selected, dim=-1)
        return float(probs[0].item())

    @staticmethod
    def _select_primary_logit(scores: torch.Tensor, token_ids: List[int]) -> Optional[float]:
        if scores.ndim > 1:
            scores = scores.squeeze(0)

        vocab_size = scores.shape[-1]
        for token_id in token_ids:
            if 0 <= token_id < vocab_size:
                return float(scores[token_id].item())
        return None

    @staticmethod
    def plot_boundary_confidences(
        window_times: List[float],
        window_confidences: List[float],
        boundary_times: List[float],
        output_path: str,
        window_entropies: Optional[List[Optional[float]]] = None,
    ) -> Optional[float]:
        if not window_times or not window_confidences:
            print("[WARN] No window confidences available; skipping boundary confidence plot")
            return None

        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError as exc:
            print(f"[WARN] Matplotlib not available ({exc}); skipping boundary confidence plot")
            return

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        records: List[Tuple[float, float, Optional[float]]] = []
        for idx, (time_val, conf_val) in enumerate(zip(window_times, window_confidences)):
            entropy_val: Optional[float] = None
            if window_entropies and idx < len(window_entropies):
                entropy_candidate = window_entropies[idx]
                if entropy_candidate is not None:
                    try:
                        entropy_val = float(entropy_candidate)
                    except (TypeError, ValueError):
                        entropy_val = None
            records.append((float(time_val), float(conf_val), entropy_val))

        records.sort(key=lambda r: r[0])
        sorted_times = [r[0] for r in records]
        sorted_confidences = [r[1] for r in records]
        sorted_entropies = [r[2] for r in records]

        max_conf = max((abs(c) for c in sorted_confidences if math.isfinite(c)), default=0.0)
        scale_factor = 1.0
        scale_exp = 0
        if max_conf > 0.0 and max_conf < 1e-2:
            scale_exp = max(0, int(math.floor(-math.log10(max_conf))))
            scale_factor = 10 ** scale_exp
            sorted_confidences = [c * scale_factor for c in sorted_confidences]
            print(f"[INFO] Applied scaling factor ×10^{scale_exp} to boundary confidence plot")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sorted_times, sorted_confidences, linewidth=1.5, label="Window confidence")

        if boundary_times:
            ymin, ymax = ax.get_ylim()
            for t in boundary_times:
                ax.axvline(t, color="red", linestyle="--", alpha=0.35)
            ax.set_ylim(ymin, ymax)

        ax.set_xlabel("Time (s)")
        if scale_factor != 1.0:
            ax.set_ylabel(f"Boundary probability (×10^{scale_exp})")
        else:
            ax.set_ylabel("Boundary probability")
        ax.set_title("Qwen Binary Boundary Confidence")
        if sorted_confidences:
            ymin = min(sorted_confidences)
            ymax = max(sorted_confidences)
            if ymin == ymax:
                pad = max(0.05, abs(ymin) * 0.1)
                ymin -= pad
                ymax += pad
            else:
                pad = (ymax - ymin) * 0.1
                ymin -= pad
                ymax += pad
            ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle="--", alpha=0.2)

        entropy_points = [
            (t, ent)
            for t, ent in zip(sorted_times, sorted_entropies)
            if ent is not None and math.isfinite(ent)
        ]

        if entropy_points:
            entropy_times = [p[0] for p in entropy_points]
            entropy_values = [p[1] for p in entropy_points]
            ax2 = ax.twinx()
            ax2.plot(
                entropy_times,
                entropy_values,
                color="orange",
                linestyle="--",
                linewidth=1.0,
                label="Window entropy",
            )
            ax2.set_ylabel("Binary entropy (nats)")
            ent_min = min(entropy_values)
            ent_max = max(entropy_values)
            if ent_min == ent_max:
                pad = max(0.01, abs(ent_min) * 0.1)
                ent_min -= pad
                ent_max += pad
            else:
                pad = (ent_max - ent_min) * 0.1
                ent_min -= pad
                ent_max += pad
            ax2.set_ylim(ent_min, ent_max)

            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(handles + handles2, labels + labels2, loc="upper right")
        else:
            ax.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        print(f"[INFO] Saved boundary confidence plot to {output_path}")
        return scale_factor

    def _build_prompt(self, num_frames: int, left_frame_num: int, right_frame_num: int) -> str:
        base_prompts = {
            "small": (
                f"You are analyzing a sequence of {num_frames} consecutive video frames shown in order.\n\n"
                f"Task: Determine if there is an EVENT BOUNDARY between frame {left_frame_num} and frame {right_frame_num} (the two central frames).\n\n"
                "An event boundary occurs when:\n"
                "- The scene changes completely (different location/setting)\n"
                "- The activity/action changes significantly\n"
                "- There's a temporal jump (cut to different time)\n"
                "- The camera angle changes dramatically\n\n"
                "Instructions:\n"
                "1. Review all frames to understand the temporal context.\n"
                f"2. Compare frames {left_frame_num} and {right_frame_num} closely while considering the surrounding frames.\n"
            ),
            "narrative": (
                f"You are analyzing {num_frames} frames spanning several seconds of video.\n\n"
                f"Task: Determine if there is a MAJOR SCENE TRANSITION between frame {left_frame_num} and frame {right_frame_num}.\n\n"
                "A MAJOR scene transition occurs when:\n"
                "- The narrative context changes completely (e.g., dream to reality, past to present)\n"
                "- Different story segment begins (e.g., action scene to dialogue scene)\n"
                "- Location AND characters change together\n"
                "- Thematic shift occurs (e.g., comedy to drama, peace to conflict)\n"
                "- Time period changes significantly (flashback, flash-forward, next day)\n\n"
                "This is NOT a major transition if:\n"
                "- Same scene continues with just camera angle changes\n"
                "- Same conversation or action continues\n"
                "- Minor cuts within the same sequence\n\n"
                "Analyze the broader context and narrative flow, not just visual differences.\n"
            ),
            "context": (
                f"Examine these {num_frames} video frames as a sequence.\n\n"
                "Questions to consider:\n"
                "1. Do the frames before and after the center represent the SAME SCENE or DIFFERENT SCENES?\n"
                "2. Is there a change in: setting/location, time period, story thread, or activity type?\n"
                "3. Would a viewer understand these as part of the same scene or separate scenes?\n\n"
                f"Focus on frames {left_frame_num} and {right_frame_num} as the potential boundary.\n"
                "Only mark boundaries between distinctly different scenes (like chapter breaks in a story).\n"
            ),
            "semantic": (
                f"Analyze this sequence of {num_frames} frames for SEMANTIC SCENE BOUNDARIES.\n\n"
                "You should detect a boundary when there is a change in:\n"
                "- Semantic context (what is happening in the story)\n"
                "- Temporal continuity (jump in time/place)\n"
                "- Narrative unit (one complete action/event ends, another begins)\n\n"
                "Examples of major boundaries:\n"
                "- Commercial/intro ends → main content begins\n"
                "- Indoor scene → outdoor scene with different characters\n"
                "- Present day → flashback/dream sequence\n"
                "- Action sequence → quiet character moment\n\n"
                f"Is there such a boundary between frames {left_frame_num} and {right_frame_num}?\n"
            ),
        }

        base_prompt = base_prompts.get(self.prompt_type, base_prompts["small"])

        if self.response_mode == "binary":
            suffix = (
                "Respond with exactly one lowercase word: 'true' if there is an event boundary between the two center frames,"
                " otherwise respond 'false'. Do not include punctuation or any extra text."
            )
        else:
            json_suffixes = {
                "small": (
                    "3. Respond with ONLY valid JSON, no other text.\n\n"
                    "Required JSON format:\n"
                    "{\"boundary\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"brief explanation\"}\n\n"
                    "Example responses:\n"
                    "{\"boundary\": true, \"confidence\": 0.9, \"reason\": \"scene change from indoor to outdoor\"}\n"
                    "{\"boundary\": false, \"confidence\": 0.8, \"reason\": \"continuous action\"}"
                ),
                "narrative": "{\"boundary\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"narrative reason\"}",
                "context": (
                    "Respond with JSON only:\n"
                    "{\"boundary\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"scene change type\"}"
                ),
                "semantic": "{\"boundary\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"type of transition\"}",
            }
            suffix = json_suffixes.get(self.prompt_type, json_suffixes["small"])

        return base_prompt + suffix

    def ask_boundary_native(self, frames: List[Image.Image], left_idx: int, right_idx: int) -> Dict:
        """
        Use Qwen2.5-VL's native multi-frame input capability
        Ask if there's a boundary between the two middle frames
        """
        if not frames:
            return {"boundary": False, "confidence": 0.0, "reason": "no_frames"}

        num_frames = len(frames)
        left_idx = max(0, min(left_idx, num_frames - 1))
        right_idx = max(0, min(right_idx, num_frames - 1))
        if right_idx == left_idx and num_frames > 1:
            right_idx = min(num_frames - 1, left_idx + 1)

        # Convert frames to base64 for Qwen input format
        frame_urls = [self.encode_frame_to_base64(f) for f in frames]

        # Build the message with multiple images
        image_inputs = [{"type": "image", "image": url} for url in frame_urls]

        left_frame_num = left_idx + 1
        right_frame_num = right_idx + 1

        text_prompt = self._build_prompt(num_frames, left_frame_num, right_frame_num)

        # Construct message in Qwen2.5-VL format
        messages = [
            {
                "role": "user",
                "content": image_inputs + [{"type": "text", "text": text_prompt}]
            }
        ]

        # Process with the model
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision info (required for Qwen2.5-VL)
        image_inputs, video_inputs = process_vision_info(messages)
        # If no videos present, pass None so processor skips video branch
        if not video_inputs or all((not v for v in video_inputs)):
            video_inputs = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        input_ids = inputs["input_ids"] if isinstance(inputs, dict) or hasattr(inputs, "__getitem__") else inputs.input_ids
        prompt_length = input_ids.size(1)

        if self.response_mode == "binary":
            gen_kwargs = dict(self.generation_config)
            gen_kwargs.update({
                "max_new_tokens": 3,
                "return_dict_in_generate": True,
                "output_scores": True,
            })

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )

            sequences = outputs.sequences
            generated_ids = sequences[:, prompt_length:]
            decoded = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            raw_response = decoded[0].strip().lower() if decoded else ""

            first_step_scores = outputs.scores[0] if getattr(outputs, "scores", None) else None
            logit_true = self._select_primary_logit(first_step_scores[0], self.true_token_ids) if first_step_scores is not None else None
            logit_false = self._select_primary_logit(first_step_scores[0], self.false_token_ids) if first_step_scores is not None and self.false_token_ids else None

            if first_step_scores is not None:
                prob_true = self.compute_token_probability(
                    first_step_scores[0],
                    self.true_token_ids,
                    self.false_token_ids,
                    mode="full",
                )
                if self.false_token_ids:
                    prob_false = self.compute_token_probability(
                        first_step_scores[0],
                        self.false_token_ids,
                        self.true_token_ids,
                        mode="full",
                    )
                else:
                    prob_false = max(0.0, 1.0 - prob_true)
            else:
                prob_true = 0.0
                prob_false = 0.0

            logit_margin: Optional[float] = None
            if logit_true is not None and logit_false is not None:
                logit_margin = float(logit_true - logit_false)

            entropy: Optional[float] = None
            if prob_true > 0.0 and prob_false > 0.0:
                entropy = float(-prob_true * math.log(prob_true) - prob_false * math.log(prob_false))

            boundary = prob_true >= 0.5
            if raw_response.startswith("true"):
                boundary = True
            elif raw_response.startswith("false"):
                boundary = False

            result = {
                "boundary": boundary,
                "confidence": prob_true,
                "prob_true": prob_true,
                "prob_false": prob_false,
                "logit_true": logit_true,
                "logit_false": logit_false,
                "logit_margin": logit_margin,
                "entropy": entropy,
                "raw_response": raw_response,
            }

            return result

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_config
            )

        # Decode only the generated part
        generated_ids = generated_ids[:, prompt_length:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return self.parse_json_response(response)

    def parse_json_response(self, text: str) -> Dict:
        """Robustly parse JSON from model response"""
        # Try to find JSON object
        json_match = re.search(r'\{[^}]*\}', text, re.DOTALL)
        
        if json_match:
            try:
                # Clean up common issues
                json_str = json_match.group(0)
                json_str = json_str.replace("'", '"')  # Single to double quotes
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Quote keys
                json_str = json_str.replace("true", "true").replace("false", "false")
                
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to parse keywords
        text_lower = text.lower()
        result = {"boundary": False, "confidence": 0.5}
        
        if "boundary: true" in text_lower or '"boundary": true' in text_lower:
            result["boundary"] = True
            result["confidence"] = 0.7
        elif "boundary: false" in text_lower or '"boundary": false' in text_lower:
            result["boundary"] = False
            result["confidence"] = 0.7
            
        # Try to extract confidence
        conf_match = re.search(r'confidence["\s:]+(\d*\.?\d+)', text_lower)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except:
                pass
                
        return result

    def sliding_window_detection(
        self,
        video_path: str,
        sample_fps: float = 2.0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        window_size: int = 7,  # Odd number for clear center
        stride: int = 1,
        vote_threshold: int = 1,
        merge_distance_sec: float = 0.5,
        output_dir: str = "outputs",
        plot_file: Optional[str] = None,
    ) -> Dict:
        """
        Sliding window boundary detection using native Qwen2.5-VL multi-frame input
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample frames
        frames, frame_indices, video_fps, (start_frame, end_frame) = self.sample_frames(
            video_path, sample_fps, start_time, end_time
        )
        n_frames = len(frames)
        
        if n_frames < window_size:
            print(f"[WARN] Video too short ({n_frames} frames) for window size {window_size}")
            window_size = n_frames
        
        # Ensure even window size to evaluate boundary between middle frames
        if window_size % 2 != 0:
            window_size += 1
            
        results = []
        boundary_votes = defaultdict(int)
        boundary_confidences = defaultdict(list)
        boundary_entropies = defaultdict(list)
        boundary_margins = defaultdict(list)
        window_times: List[float] = []
        window_confidences: List[float] = []
        window_entropies: List[Optional[float]] = []
        window_margins: List[Optional[float]] = []
        
        # Sliding windows
        print(f"[INFO] Processing {n_frames} frames with window_size={window_size}, stride={stride}")

        total_candidates = max(1, n_frames - window_size + 1)
        total_windows = ((total_candidates - 1) // stride) + 1
        pbar = tqdm(total=total_windows, desc="Analyzing windows", unit="win")

        for start_idx in range(0, total_candidates, stride):
            end_idx = min(start_idx + window_size, n_frames)
            window_frames = frames[start_idx:end_idx]

            # Pad if necessary (repeat last frame)
            while len(window_frames) < window_size:
                window_frames.append(window_frames[-1])

            right_local_idx = window_size // 2
            left_local_idx = right_local_idx - 1 if right_local_idx > 0 else 0
            global_left_idx = start_idx + left_local_idx
            global_right_idx = start_idx + right_local_idx

            # Ask model about boundary between middle frames
            try:
                result = self.ask_boundary_native(window_frames, left_local_idx, right_local_idx)
            except Exception as e:
                print(f"[ERROR] Failed at window starting {start_idx}: {e}")
                print(traceback.format_exc())
                result = {"boundary": False, "confidence": 0.0}

            results.append({
                "window_start": start_idx,
                "global_left": global_left_idx,
                "global_right": global_right_idx,
                "result": result
            })

            # Vote if boundary detected
            if result.get("boundary", False):
                boundary_votes[(global_left_idx, global_right_idx)] += 1

            conf_value = result.get("confidence")
            entropy_value = result.get("entropy")
            margin_value = result.get("logit_margin")

            entropy_float: Optional[float] = None
            if entropy_value is not None:
                try:
                    entropy_float = float(entropy_value)
                except (TypeError, ValueError):
                    entropy_float = None

            margin_float: Optional[float] = None
            if margin_value is not None:
                try:
                    margin_float = float(margin_value)
                except (TypeError, ValueError):
                    margin_float = None

            if conf_value is not None:
                try:
                    conf_float = float(conf_value)
                except (TypeError, ValueError):
                    conf_float = None
                else:
                    boundary_confidences[(global_left_idx, global_right_idx)].append(conf_float)
                    if entropy_float is not None:
                        boundary_entropies[(global_left_idx, global_right_idx)].append(entropy_float)
                    if margin_float is not None:
                        boundary_margins[(global_left_idx, global_right_idx)].append(margin_float)

                    if 0 <= global_right_idx < len(frame_indices):
                        time_sec = frame_indices[global_right_idx] / video_fps
                        window_times.append(float(time_sec))
                        window_confidences.append(conf_float)
                        window_entropies.append(entropy_float)
                        window_margins.append(margin_float)
            
            # Progress
            pbar.update(1)
        pbar.close()
        
        # Aggregate votes into boundaries
        boundary_pairs = [pair for pair, votes in boundary_votes.items()
                          if votes >= vote_threshold]
        boundary_pairs.sort()

        pair_confidences = {
            pair: float(np.mean(values))
            for pair, values in boundary_confidences.items()
            if values
        }
        pair_entropies = {
            pair: float(np.mean([v for v in values if v is not None]))
            for pair, values in boundary_entropies.items()
            if values and any(v is not None for v in values)
        }
        pair_margins = {
            pair: float(np.mean([v for v in values if v is not None]))
            for pair, values in boundary_margins.items()
            if values and any(v is not None for v in values)
        }

        # Merge nearby boundaries (based on right index)
        merged_boundaries: List[float] = []
        merged_confidences: List[Optional[float]] = []
        merged_entropies: List[Optional[float]] = []
        merged_margins: List[Optional[float]] = []
        merge_distance_frames = int(merge_distance_sec * sample_fps)

        clusters: List[List[Tuple[float, Optional[float], Optional[float], Optional[float]]]] = []
        for left_idx, right_idx in boundary_pairs:
            center_idx = (left_idx + right_idx) / 2.0
            conf_val = pair_confidences.get((left_idx, right_idx))
            ent_val = pair_entropies.get((left_idx, right_idx))
            margin_val = pair_margins.get((left_idx, right_idx))
            data_point = (center_idx, conf_val, ent_val, margin_val)

            if not clusters:
                clusters.append([data_point])
                continue

            prev_center = clusters[-1][-1][0]
            if center_idx - prev_center <= merge_distance_frames:
                clusters[-1].append(data_point)
            else:
                clusters.append([data_point])

        for cluster in clusters:
            centers = [c for c, _, _, _ in cluster]
            merged_boundaries.append(sum(centers) / len(centers))

            confs = [c for _, c, _, _ in cluster if c is not None]
            merged_confidences.append(float(sum(confs) / len(confs)) if confs else None)

            ents = [e for _, _, e, _ in cluster if e is not None]
            merged_entropies.append(float(sum(ents) / len(ents)) if ents else None)

            margins = [m for _, _, _, m in cluster if m is not None]
            merged_margins.append(float(sum(margins) / len(margins)) if margins else None)

        # Convert to timestamps
        boundary_times: List[float] = []
        boundary_time_confidences: List[Optional[float]] = []
        boundary_time_entropies: List[Optional[float]] = []
        boundary_time_margins: List[Optional[float]] = []
        for center, conf, ent, margin in zip(merged_boundaries, merged_confidences, merged_entropies, merged_margins):
            int_center = int(round(center))
            if 0 <= int_center < len(frame_indices):
                time_sec = frame_indices[int_center] / video_fps
                boundary_times.append(time_sec)
                boundary_time_confidences.append(conf)
                boundary_time_entropies.append(ent)
                boundary_time_margins.append(margin)

        # Save results
        output_data = {
            "video_path": video_path,
            "model": self.model_name,
            "response_mode": self.response_mode,
            "fps": video_fps,
            "sample_fps": sample_fps,
            "start_time": start_time,
            "end_time": end_time,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "n_frames_analyzed": n_frames,
            "frame_indices": frame_indices,
            "window_size": window_size,
            "stride": stride,
            "boundary_centers": merged_boundaries,
            "boundary_frame_pairs": boundary_pairs,
            "boundary_times": boundary_times,
            "boundary_confidences": boundary_time_confidences,
            "boundary_entropies": boundary_time_entropies,
            "boundary_margins": boundary_time_margins,
            "window_times": window_times,
            "window_confidences": window_confidences,
            "window_entropies": window_entropies,
            "window_margins": window_margins,
            "raw_votes": {
                f"{left}-{right}": votes
                for (left, right), votes in boundary_votes.items()
            },
            "pair_confidences": {
                f"{left}-{right}": value
                for (left, right), value in pair_confidences.items()
            },
            "pair_entropies": {
                f"{left}-{right}": value
                for (left, right), value in pair_entropies.items()
            },
            "pair_margins": {
                f"{left}-{right}": value
                for (left, right), value in pair_margins.items()
            },
            "raw_confidences": {
                f"{left}-{right}": values
                for (left, right), values in boundary_confidences.items()
            },
            "raw_entropies": {
                f"{left}-{right}": values
                for (left, right), values in boundary_entropies.items()
            },
            "raw_margins": {
                f"{left}-{right}": values
                for (left, right), values in boundary_margins.items()
            },
        }

        plot_path = None
        if plot_file:
            plot_path = os.path.abspath(plot_file)
            try:
                scale_factor = self.plot_boundary_confidences(
                    window_times,
                    window_confidences,
                    boundary_times,
                    plot_path,
                    window_entropies,
                )
                output_data["plot_file"] = plot_path
                if scale_factor and scale_factor != 1.0:
                    output_data["plot_scale"] = scale_factor
            except Exception as exc:
                print(f"[WARN] Failed to generate boundary confidence plot: {exc}")
                plot_path = None

        # Save main output
        output_file = os.path.join(output_dir, "boundaries.json")
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        # Save simple text format
        txt_file = os.path.join(output_dir, "boundary_times.txt")
        with open(txt_file, "w") as f:
            f.write("# Event boundaries\n")
            f.write("# center_idx\tseconds\tconfidence\tentropy\tlogit_margin\n")
            for center, time, conf, ent, margin in zip(
                merged_boundaries,
                boundary_times,
                boundary_time_confidences,
                boundary_time_entropies,
                boundary_time_margins,
            ):
                conf_str = "n/a" if conf is None else f"{conf:.6f}"
                ent_str = "n/a" if ent is None else f"{ent:.6f}"
                margin_str = "n/a" if margin is None else f"{margin:.6f}"
                f.write(f"{center:.2f}\t{time:.3f}\t{conf_str}\t{ent_str}\t{margin_str}\n")
        
        print(f"[INFO] Found {len(boundary_times)} boundaries")
        print(f"[INFO] Results saved to {output_dir}/")
        if plot_path:
            print(f"[INFO] Boundary confidence plot saved to {plot_path}")
        
        return output_data

    def extract_features_for_hmm(self, frames: List[Image.Image], batch_size: int = 8) -> np.ndarray:
        """
        Extract visual features for HMM using Qwen2.5-VL's vision encoder
        Note: This requires accessing internal model components
        """
        features = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            
            # Convert to model inputs
            frame_urls = [self.encode_frame_to_base64(f) for f in batch_frames]
            
            # Process each frame to get vision features
            for frame_url in frame_urls:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame_url},
                        {"type": "text", "text": "Describe this image."}  # Dummy text
                    ]
                }]
                
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(self.model.device)
                
                # Get hidden states from vision encoder
                with torch.no_grad():
                    # This requires accessing model internals
                    # Note: Implementation depends on exact model architecture
                    outputs = self.model.model(**inputs, output_hidden_states=True)
                    
                    # Use pooled vision features (typically from vision encoder)
                    if hasattr(outputs, 'hidden_states'):
                        # Take mean of last hidden state as frame feature
                        hidden = outputs.hidden_states[-1]
                        pooled = hidden.mean(dim=1)
                        features.append(pooled.cpu().numpy())
        
        return np.vstack(features) if features else np.array([])


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL temporal segmentation")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                       help="Model name (e.g., Qwen/Qwen3-VL-2B-Instruct or Qwen/Qwen3-VL-30B-A3B-Instruct)")
    parser.add_argument("--prompt-type", type=str, choices=["small", "narrative", "context", "semantic"], default="small",
                       help="Prompt style: 'small' (default), or major-scene prompts: 'narrative' | 'context' | 'semantic'")
    parser.add_argument("--response-mode", type=str, choices=["json", "binary"], default="json",
                       help="Model response mode. Use 'binary' to force true/false output with logit-derived confidence.")
    parser.add_argument("--sample-fps", type=float, default=2.0,
                       help="Frame sampling rate")
    parser.add_argument("--start-time", type=float, default=None,
                       help="Start time in seconds (inclusive)")
    parser.add_argument("--end-time", type=float, default=None,
                       help="End time in seconds (inclusive)")
    parser.add_argument("--window-size", type=int, default=8,
                       help="Window size (should be odd)")
    parser.add_argument("--stride", type=int, default=2,
                       help="Stride for sliding window")
    parser.add_argument("--vote-threshold", type=int, default=1,
                       help="Minimum votes to confirm boundary")
    parser.add_argument("--merge-distance", type=float, default=0.5,
                       help="Merge boundaries within this many seconds")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--plot-file", type=str, default=None,
                       help="Optional path to save a boundary confidence plot (PNG)")
    
    args = parser.parse_args()
    
    # Initialize segmenter
    segmenter = QwenTemporalSegmenterFixed(
        model_name=args.model,
        prompt_type=args.prompt_type,
        response_mode=args.response_mode,
    )
    
    # Run detection
    results = segmenter.sliding_window_detection(
        video_path=args.video,
        sample_fps=args.sample_fps,
        window_size=args.window_size,
        stride=args.stride,
        vote_threshold=args.vote_threshold,
        merge_distance_sec=args.merge_distance,
        output_dir=args.output_dir,
        start_time=args.start_time,
        end_time=args.end_time,
        plot_file=args.plot_file,
    )
    
    times = results.get("boundary_times", [])
    confs = results.get("boundary_confidences", [])
    print("\nDetected boundaries (seconds):")
    if times and confs and len(times) == len(confs):
        for t, c in zip(times, confs):
            conf_str = "n/a" if c is None else f"{c:.3f}"
            print(f"  {t:.2f}s (confidence {conf_str})")
    else:
        for t in times:
            print(f"  {t:.2f}s")


if __name__ == "__main__":
    main()
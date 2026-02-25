#!/usr/bin/env python3
"""
qwen3_omni_audiovisual_segmentation.py
Temporal video segmentation using Qwen3-Omni with audio-visual analysis
"""

import argparse
import os
import json
import re
import traceback
import tempfile
import subprocess
import shutil
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import torch
import librosa
import soundfile as sf
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from tqdm import tqdm


def device_name():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class QwenOmniAudioVisualSegmenter:
    def __init__(self, model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct", device: Optional[str] = None):
        """
        Initialize with Qwen3-Omni model for audio-visual temporal segmentation
        """
        self.model_name = model_name
        self.device = device or device_name()
        print(f"[INFO] Using device: {self.device}")

        # Robustness configuration
        self.audio_sample_rate = 16000
        self.min_audio_duration = 0.1  # seconds
        self.audio_fallback_enabled = True
        self.skip_last_window = True

        # Cached dtype for float tensors
        self.model_dtype = None
        
        # Load Qwen3-Omni model
        print(f"[INFO] Loading Qwen3-Omni model: {model_name}")
        requested_attn = "flash_attention_2" if self.device != "cpu" else "eager"
        fallback_attn = "sdpa" if self.device != "cpu" else "eager"
        try:
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto",
                attn_implementation=requested_attn,
            )
            self.attn_implementation = requested_attn
            self.model_dtype = getattr(self.model, "dtype", None)
        except Exception as e:
            print(f"[WARN] Failed to initialize with {requested_attn}: {e}")
            if fallback_attn != requested_attn:
                print(f"[INFO] Falling back to {fallback_attn} attention implementation.")
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto",
                attn_implementation=fallback_attn,
            )
            self.attn_implementation = fallback_attn
            self.model_dtype = getattr(self.model, "dtype", None)
        if self.model_dtype is None:
            try:
                self.model_dtype = next(self.model.parameters()).dtype
            except StopIteration:
                self.model_dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        # Load processor
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
        
        # Generation config for deterministic JSON output
        self.generation_config = {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "do_sample": False,
            "top_p": 0.1,
        }
        
        # Temp audio file tracking for cleanup
        self.temp_audio_files = []
        
        # One-time logging toggle for generate() schema
        self._logged_generate_schema = False
        
        # Control verbose debug prints (artifacts still saved regardless)
        self.enable_debug_prints = False
        
        # Cache for boolean token ids used in logit-based confidence
        self._true_token_ids = None
        self._false_token_ids = None

    def create_silent_audio_segment(self, duration_sec: float, dest_path: Optional[str] = None) -> str:
        """Create a silent audio segment and track it for cleanup if temporary."""
        duration_sec = max(duration_sec, self.min_audio_duration)
        silent_audio = np.zeros(int(self.audio_sample_rate * duration_sec), dtype=np.float32)

        if dest_path:
            dest_dir = os.path.dirname(dest_path)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)
            target_path = dest_path
            is_temp = False
        else:
            fd, target_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            is_temp = True

        sf.write(target_path, silent_audio, self.audio_sample_rate)

        if is_temp:
            self.temp_audio_files.append(target_path)

        return target_path

    def validate_audio_file(self, audio_path: str) -> bool:
        """Validate that an audio file can be processed by the model."""
        try:
            audio, sr = librosa.load(
                audio_path,
                sr=self.audio_sample_rate,
                duration=self.min_audio_duration,
                mono=True,
            )
            if len(audio) == 0:
                return False
            if np.all(audio == 0) or np.any(np.isnan(audio)):
                return False
            return True
        except Exception:
            return False

    def sample_frames(self, video_path: str, sample_fps: float = 2.0,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None) -> Tuple[List[Image.Image], List[int], float, Tuple[int, int]]:
        """
        Sample frames from video at specified fps within optional [start_time, end_time] seconds.
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
        """Convert PIL image to base64 for Qwen3-Omni input"""
        buffered = BytesIO()
        frame.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def _build_instruction_text(
        self,
        num_frames: int,
        left_frame_num: int,
        right_frame_num: int,
        window_range: Tuple[int, int],
    ) -> str:
        return (
            f"You are analyzing {num_frames} consecutive video frames with a single corresponding audio segment.\n\n"
            "AUDIO:\n"
            f"- The audio corresponds to frames {window_range[0]} through {window_range[1]} (entire window)\n\n"
            f"TASK: Determine if there is an EVENT BOUNDARY between frame {left_frame_num} and frame {right_frame_num}.\n\n"
            "Analyze BOTH modalities:\n\n"
            "VISUAL CUES for boundaries:\n"
            "- Scene changes (different location/setting/environment)\n"
            "- Activity or action changes\n"
            "- Camera angle or shot changes\n"
            "- Temporal jumps or cuts\n"
            "- Lighting changes\n\n"
            "AUDIO CUES for boundaries:\n"
            "- Speech stopping/starting or speaker changes\n"
            "- Music transitions, genre changes, or tempo shifts\n"
            "- Sound effects changes or new environmental sounds\n"
            "- Silence gaps or volume changes\n"
            "- Audio tone, mood, or energy shifts\n"
            "- Background noise changes\n\n"
            "IMPORTANT: Strong boundaries typically show changes in BOTH modalities. Weigh evidence from both audio and visual channels.\n\n"
            "Output ONLY valid JSON with no other text:\n"
            '{"boundary": true/false, "confidence": 0.0-1.0, "visual_cue": "brief description", "audio_cue": "brief description"}\n\n'
            "Examples:\n"
            '{"boundary": true, "confidence": 0.9, "visual_cue": "cut from indoor to outdoor scene", "audio_cue": "music stops, ambient nature sounds begin"}\n'
            '{"boundary": false, "confidence": 0.8, "visual_cue": "continuous action", "audio_cue": "ongoing dialogue"}'
        )

    def _prepare_messages(
        self,
        frames: List[Image.Image],
        audio_path: str,
        left_frame_num: int,
        right_frame_num: int,
        window_range: Tuple[int, int],
    ) -> Tuple[List[Dict], str]:
        frame_urls = [self.encode_frame_to_base64(f) for f in frames]
        instruction_text = self._build_instruction_text(
            num_frames=len(frames),
            left_frame_num=left_frame_num,
            right_frame_num=right_frame_num,
            window_range=window_range,
        )

        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": url} for url in frame_urls],
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": instruction_text},
            ],
        }]

        return messages, instruction_text

    def _prepare_window_payload(
        self,
        frames: List[Image.Image],
        frame_indices: List[int],
        left_idx: int,
        right_idx: int,
        video_path: str,
        video_fps: float,
        debug_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        num_frames = len(frames)
        left_idx = max(0, min(left_idx, num_frames - 1))
        right_idx = max(0, min(right_idx, num_frames - 1))
        if right_idx == left_idx and num_frames > 1:
            right_idx = min(num_frames - 1, left_idx + 1)
        
        mid_point = num_frames // 2 if num_frames > 1 else 0
        
        if not frame_indices:
            print("[WARN] No frame indices available for audio extraction; using silent segment")
            audio_path = self.create_silent_audio_segment(
                self.min_audio_duration,
                dest_path=os.path.join(debug_dir, "window_audio.wav") if debug_dir else None,
            )

            left_frame_num = 1
            right_frame_num = min(2, num_frames) if num_frames else 1
            window_range = (1, num_frames or 1)

            messages, instruction_text = self._prepare_messages(
                frames,
                audio_path,
                left_frame_num,
                right_frame_num,
                window_range,
            )

            return {
                "messages": messages,
                "prompt": instruction_text,
                "audio_path": audio_path,
                "left_frame_num": left_frame_num,
                "right_frame_num": right_frame_num,
                "window_range": window_range,
                "mid_point": mid_point,
            }

        # Continuous window audio: from first to last frame index in this window
        window_start_frame = max(0, frame_indices[0])
        window_end_frame = max(window_start_frame + 1, frame_indices[-1])

        audio_dest = os.path.join(debug_dir, "window_audio.wav") if debug_dir else None
        audio_path = self.extract_audio_segment(
            video_path,
            window_start_frame,
            window_end_frame,
            video_fps,
            dest_path=audio_dest,
        )

        if not self.validate_audio_file(audio_path):
            print(
                f"[WARN] Invalid window audio segment for frames {window_start_frame}-{window_end_frame}, using silent fallback"
            )
            audio_path = self.create_silent_audio_segment(
                self.min_audio_duration,
                dest_path=audio_dest,
            )

        left_frame_num = left_idx + 1
        right_frame_num = right_idx + 1
        
        window_range = (1, num_frames or 1)

        messages, instruction_text = self._prepare_messages(
            frames,
            audio_path,
            left_frame_num,
            right_frame_num,
            window_range,
        )

        return {
            "messages": messages,
            "prompt": instruction_text,
            "audio_path": audio_path,
            "left_frame_num": left_frame_num,
            "right_frame_num": right_frame_num,
            "window_range": window_range,
            "mid_point": mid_point,
        }

    def extract_audio_segment(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        video_fps: float,
        dest_path: Optional[str] = None,
    ) -> str:
        """Extract audio segment corresponding to frame range with robust fallbacks."""
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        start_time = max(start_frame / max(video_fps, 1e-6), 0.0)
        end_time = max(end_frame / max(video_fps, 1e-6), start_time)
        duration = max(end_time - start_time, self.min_audio_duration)

        if dest_path:
            dest_dir = os.path.dirname(dest_path)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)
            temp_audio_path = dest_path
            created_temp = False
        else:
            fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            created_temp = True

        extractor = shutil.which("ffmpeg")
        ffmpeg_success = False
        if extractor:
            ffmpeg_cmd = [
                extractor,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_time:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                video_path,
                "-ac",
                "1",
                "-ar",
                str(self.audio_sample_rate),
                "-y",
                temp_audio_path,
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                ffmpeg_success = True
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr.decode(errors="ignore") if e.stderr else str(e)
                print(f"[WARN] ffmpeg audio extraction failed: {err_msg}")

        if not ffmpeg_success:
            try:
                audio, sr = librosa.load(
                    video_path,
                    sr=self.audio_sample_rate,
                    offset=start_time,
                    duration=duration,
                    mono=True,
                )
                if len(audio) < int(self.audio_sample_rate * self.min_audio_duration):
                    target_length = int(self.audio_sample_rate * self.min_audio_duration)
                    audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
                audio = np.asarray(audio, dtype=np.float32).flatten()
                sf.write(temp_audio_path, audio, self.audio_sample_rate)
                ffmpeg_success = True
            except Exception as e:
                print(f"[WARN] Fallback audio extraction failed: {e}")

        if not ffmpeg_success:
            print(
                f"[WARN] Unable to extract audio for frames {start_frame}-{end_frame}, using silent segment"
            )
            silent_samples = max(int(round(self.audio_sample_rate * duration)), 1)
            silent_audio = np.zeros(silent_samples, dtype=np.float32)
            sf.write(temp_audio_path, silent_audio, self.audio_sample_rate)

        if created_temp:
            self.temp_audio_files.append(temp_audio_path)
        return temp_audio_path

    def ask_boundary_with_audio(self, frames: List[Image.Image], 
                                video_path: str, frame_indices: List[int],
                                left_idx: int, right_idx: int, 
                                video_fps: float,
                                debug_dir: Optional[str] = None,
                                inspect_dir: Optional[str] = None) -> Dict:
        """
        Enhanced boundary detection using both visual frames and corresponding audio segments.
        When debug_dir is provided, we save the window assets and skip model generation.
        """
        if not frames:
            return {"boundary": False, "confidence": 0.0, "visual_cue": "no_frames", "audio_cue": "no_audio"}

        os.makedirs(debug_dir, exist_ok=True) if debug_dir else None
        os.makedirs(inspect_dir, exist_ok=True) if inspect_dir else None

        try:
            payload = self._prepare_window_payload(
                frames,
                frame_indices,
                left_idx,
                right_idx,
                video_path,
                video_fps,
                debug_dir=debug_dir,
            )
        except Exception as e:
            print(f"[WARN] Failed to prepare multimodal payload: {e}")
            print(traceback.format_exc())
            if self.audio_fallback_enabled:
                return self.ask_boundary_visual_only(frames, left_idx, right_idx)
            raise

        if debug_dir:
            prompt_path = os.path.join(debug_dir, "prompt.txt")
            with open(prompt_path, "w") as f:
                f.write(payload["prompt"])

            # Save frames sequentially
            for idx, frame in enumerate(frames):
                frame_path = os.path.join(debug_dir, f"frame_{idx:02d}.jpg")
                if not os.path.exists(frame_path):
                    frame.save(frame_path)

            return {
                "boundary": False,
                "confidence": 0.0,
                "visual_cue": "debug_mode",
                "audio_cue": "debug_mode",
            }

        try:
            messages = payload["messages"]
            text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            audio_lists, images, videos = process_mm_info(messages, use_audio_in_video=False)
            audios = [clip for per_message in audio_lists for clip in per_message]
            if not audios:
                audios = None
            else:
                if self.enable_debug_prints:
                    print(f"[DEBUG] Window audio clips: {len(audios)}")
                    for idx, clip in enumerate(audios):
                        arr = np.asarray(clip)
                        print(f"[DEBUG]  clip {idx}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")

            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=None,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            inputs = inputs.to(self.model.device)
            target_dtype = self.model_dtype
            feature = inputs.get("input_features") if isinstance(inputs, dict) else None
            if isinstance(feature, torch.Tensor):
                print(f"[DEBUG] audio input_features dtype before cast: {feature.dtype}, shape={feature.shape}")
            if target_dtype in (torch.float16, torch.bfloat16):
                for key, value in list(inputs.items()):
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        inputs[key] = value.to(target_dtype)
            feature = inputs.get("input_features") if isinstance(inputs, dict) else None
            if isinstance(feature, torch.Tensor):
                print(f"[DEBUG] audio input_features dtype after cast: {feature.dtype}")

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **self.generation_config,
                )

            # Introspect generate() output and select sequences based on actual structure
            input_ids_tensor = inputs["input_ids"] if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)

            def tensor_info(t: torch.Tensor) -> Dict[str, Any]:
                return {
                    "is_tensor": True,
                    "dtype": str(t.dtype),
                    "ndim": int(t.ndim),
                    "shape": list(t.shape),
                }

            def summarize_generated(gen_obj: Any) -> Dict[str, Any]:
                summary: Dict[str, Any] = {"type": str(type(gen_obj))}
                try:
                    if torch.is_tensor(gen_obj):
                        summary.update(tensor_info(gen_obj))
                    elif hasattr(gen_obj, "sequences") and torch.is_tensor(getattr(gen_obj, "sequences")):
                        seq = getattr(gen_obj, "sequences")
                        summary["has_sequences_attr"] = True
                        summary["sequences"] = tensor_info(seq)
                    elif isinstance(gen_obj, dict):
                        summary["dict_keys"] = list(gen_obj.keys())
                        if "sequences" in gen_obj and torch.is_tensor(gen_obj["sequences"]):
                            summary["sequences"] = tensor_info(gen_obj["sequences"])
                    elif isinstance(gen_obj, (list, tuple)):
                        elem_summaries = []
                        for i, elem in enumerate(list(gen_obj)[:5]):
                            if torch.is_tensor(elem):
                                elem_summaries.append({"index": i, **tensor_info(elem)})
                            elif hasattr(elem, "sequences") and torch.is_tensor(getattr(elem, "sequences")):
                                elem_summaries.append({"index": i, "has_sequences_attr": True, "sequences": tensor_info(getattr(elem, "sequences"))})
                            elif isinstance(elem, dict) and "sequences" in elem and torch.is_tensor(elem["sequences"]):
                                elem_summaries.append({"index": i, "dict_has_sequences": True, "sequences": tensor_info(elem["sequences"])})
                            else:
                                elem_summaries.append({"index": i, "type": str(type(elem))})
                        summary["elements"] = elem_summaries
                except Exception as _:
                    pass
                return summary

            gen_summary = summarize_generated(generated)

            # Save/print summary for debugging
            try:
                summary_text = json.dumps({
                    "generate_return": gen_summary,
                    "input_ids": tensor_info(input_ids_tensor) if isinstance(input_ids_tensor, torch.Tensor) else None,
                }, indent=2)
            except Exception:
                summary_text = str(gen_summary)

            wrote_summary = False
            save_dir = inspect_dir or debug_dir
            if save_dir:
                try:
                    summary_path = os.path.join(save_dir, "generate_return.json")
                    with open(summary_path, "w") as fsum:
                        fsum.write(summary_text)
                    wrote_summary = True
                except Exception:
                    wrote_summary = False
            if self.enable_debug_prints and not wrote_summary and not self._logged_generate_schema:
                print("[DEBUG] generate() return summary:")
                print(summary_text)
                self._logged_generate_schema = True

            # Choose sequences tensor by inspecting candidates
            candidate_tensors: List[torch.Tensor] = []

            if torch.is_tensor(generated):
                candidate_tensors.append(generated)
            if hasattr(generated, "sequences") and torch.is_tensor(getattr(generated, "sequences")):
                candidate_tensors.append(getattr(generated, "sequences"))
            if isinstance(generated, dict) and "sequences" in generated and torch.is_tensor(generated["sequences"]):
                candidate_tensors.append(generated["sequences"])
            if isinstance(generated, (list, tuple)) and len(generated) > 0:
                for item in generated:
                    if torch.is_tensor(item):
                        candidate_tensors.append(item)
                    if hasattr(item, "sequences") and torch.is_tensor(getattr(item, "sequences")):
                        candidate_tensors.append(getattr(item, "sequences"))
                    if isinstance(item, dict) and "sequences" in item and torch.is_tensor(item["sequences"]):
                        candidate_tensors.append(item["sequences"])

            sequences = None
            # Prefer 2D Long tensors, then 2D any dtype, then tensors with ndim>=2
            for t in candidate_tensors:
                if isinstance(t, torch.Tensor) and t.ndim == 2 and t.dtype == torch.long:
                    sequences = t
                    break
            if sequences is None:
                for t in candidate_tensors:
                    if isinstance(t, torch.Tensor) and t.ndim == 2:
                        sequences = t
                        break
            if sequences is None:
                for t in candidate_tensors:
                    if isinstance(t, torch.Tensor) and t.ndim >= 2:
                        sequences = t
                        break
            if sequences is None and candidate_tensors:
                sequences = candidate_tensors[0]
            if sequences is None:
                raise TypeError(f"Unable to locate sequences tensor. Summary: {summary_text}")

            if sequences.ndim == 1:
                sequences = sequences.unsqueeze(0)

            # Remove the prompt tokens when possible
            if input_ids_tensor is not None and sequences.size(1) >= input_ids_tensor.size(1):
                generated_ids = sequences[:, input_ids_tensor.size(1):]
            else:
                generated_ids = sequences

            decoded = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            response = decoded[0] if decoded else ""
            # Persist and preview raw model text for inspection
            try:
                save_dir = inspect_dir or debug_dir
                if save_dir:
                    with open(os.path.join(save_dir, "model_response.txt"), "w") as f_txt:
                        f_txt.write(response)
                if self.enable_debug_prints and not self._logged_generate_schema:
                    preview = (response[:240] + ("..." if len(response) > 240 else "")) if isinstance(response, str) else str(response)
                    print(f"[DEBUG] model response preview: {preview}")
            except Exception:
                pass

            result = self.parse_json_response(response)

            # Compute logit-based boolean confidence from first 'true/false' decision step
            try:
                # 1) Try using generation scores if present
                scores_list, scores_source = self._extract_scores_list(generated)
                if scores_list is not None:
                    logit_info = self._compute_boundary_logit_confidence(generated_ids, scores_list)
                    if isinstance(logit_info, dict) and logit_info.get("logit_confidence") is not None:
                        logit_info["method"] = "scores"
                else:
                    logit_info = {"status": "unavailable", "reason": "scores_not_available", "scores_source": scores_source}

                # 2) If still unavailable, try a single forward pass at the decision step
                if not isinstance(logit_info, dict) or logit_info.get("logit_confidence") is None:
                    bool_loc = self._find_boolean_token_index(generated_ids)
                    if bool_loc.get("token_index") is not None and bool_loc.get("observed") in ("true", "false"):
                        forward_info = self._compute_logit_confidence_via_forward(
                            sequences=sequences,
                            input_ids_tensor=input_ids_tensor,
                            inputs=inputs,
                            token_index_in_generated=int(bool_loc["token_index"]),
                            observed_bool=str(bool_loc["observed"]),
                        )
                        # Prefer forward method if it produced a confidence
                        if isinstance(forward_info, dict) and forward_info.get("logit_confidence") is not None:
                            logit_info = forward_info
                        else:
                            logit_info = logit_info if isinstance(logit_info, dict) else {"status": "unavailable"}
                            logit_info.setdefault("fallback_forward", forward_info)
                    else:
                        logit_info = logit_info if isinstance(logit_info, dict) else {"status": "unavailable"}
                        logit_info.setdefault("boolean_locate", bool_loc)
            except Exception as e_compute:
                logit_info = {"status": "error", "reason": f"exception: {type(e_compute).__name__}"}

            # Attach to result if available and persist details regardless
            if isinstance(logit_info, dict) and logit_info.get("logit_confidence") is not None:
                result["logit_confidence"] = float(logit_info["logit_confidence"])
            if save_dir:
                try:
                    with open(os.path.join(save_dir, "logit_confidence.json"), "w") as f_lc:
                        json.dump(logit_info, f_lc, indent=2)
                except Exception:
                    pass

            return result
        except Exception as e:
            print(f"[WARN] Audio-visual inference failed: {e}")
            print(traceback.format_exc())
            if self.audio_fallback_enabled:
                return self.ask_boundary_visual_only(frames, left_idx, right_idx)
            raise

    def ask_boundary_visual_only(
        self,
        frames: List[Image.Image],
        left_idx: int,
        right_idx: int,
    ) -> Dict[str, Any]:
        """Fallback boundary detection using visual cues only."""
        if not frames:
            return {
                "boundary": False,
                "confidence": 0.0,
                "visual_cue": "no_frames",
                "audio_cue": "visual_only_fallback",
            }

        left_idx = max(0, min(left_idx, len(frames) - 1))
        right_idx = max(0, min(right_idx, len(frames) - 1))

        left_frame = np.asarray(frames[left_idx]).astype(np.float32)
        right_frame = np.asarray(frames[right_idx]).astype(np.float32)

        if left_frame.shape != right_frame.shape:
            min_height = min(left_frame.shape[0], right_frame.shape[0])
            min_width = min(left_frame.shape[1], right_frame.shape[1])
            left_frame = left_frame[:min_height, :min_width]
            right_frame = right_frame[:min_height, :min_width]

        diff = np.mean(np.abs(left_frame - right_frame)) / 255.0
        boundary = diff > 0.12
        confidence = float(min(1.0, max(0.05, (diff - 0.05) * 3.0))) if boundary else float(max(0.05, diff * 2.0))

        visual_cue = "significant visual change" if boundary else "visual continuity"

        return {
            "boundary": boundary,
            "confidence": confidence,
            "visual_cue": visual_cue,
            "audio_cue": "visual_only_fallback",
        }

    def parse_json_response(self, text: str) -> Dict:
        """Robustly parse JSON from model response"""
        # Try to find JSON object
        json_match = re.search(r'\{[^}]*\}', text, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group(0)
                # Fix common issues
                json_str = json_str.replace("'", '"')  # Single to double quotes
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Quote keys
                json_str = json_str.replace("True", "true").replace("False", "false")
                
                result = json.loads(json_str)
                # Ensure all expected fields exist
                if "boundary" not in result:
                    result["boundary"] = False
                if "confidence" not in result:
                    result["confidence"] = 0.5
                if "visual_cue" not in result:
                    result["visual_cue"] = ""
                if "audio_cue" not in result:
                    result["audio_cue"] = ""
                return result
            except json.JSONDecodeError:
                pass
        
        # Fallback parsing
        text_lower = text.lower()
        result = {
            "boundary": False, 
            "confidence": 0.5,
            "visual_cue": "",
            "audio_cue": ""
        }
        
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

    def _ensure_boolean_token_sets(self) -> None:
        """Initialize cached token id sets for boolean strings if needed."""
        if self._true_token_ids is not None and self._false_token_ids is not None:
            return
        tokenizer = getattr(self.processor, "tokenizer", None)
        self._true_token_ids = set()
        self._false_token_ids = set()
        if tokenizer is None:
            return
        candidates_true = [
            "true", " true", "True", " True",
        ]
        candidates_false = [
            "false", " false", "False", " False",
        ]
        for s in candidates_true:
            try:
                ids = tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    self._true_token_ids.add(int(ids[0]))
            except Exception:
                pass
        for s in candidates_false:
            try:
                ids = tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    self._false_token_ids.add(int(ids[0]))
            except Exception:
                pass

    def _compute_boundary_logit_confidence(
        self,
        generated_ids: torch.Tensor,
        scores_list: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """Estimate a boolean confidence using first-token probabilities for true/false.

        This uses the generation scores at the step where the boolean begins.
        """
        if not isinstance(generated_ids, torch.Tensor) or generated_ids.ndim != 2:
            return {}
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return {}

        token_ids = generated_ids[0].tolist()
        # Decode each token to build a position map
        token_pieces: List[str] = []
        for tid in token_ids:
            try:
                token_pieces.append(tokenizer.decode([int(tid)], skip_special_tokens=True))
            except Exception:
                token_pieces.append("")
        cumulative = ""
        starts: List[int] = []
        ends: List[int] = []
        for piece in token_pieces:
            starts.append(len(cumulative))
            cumulative += piece
            ends.append(len(cumulative))

        # Find the boolean position after the boundary key
        lowered = cumulative.lower()
        match = re.search(r'"boundary"\s*:\s*(true|false)', lowered)
        observed_bool = None
        bool_char_start = None
        if match:
            observed_bool = match.group(1)
            bool_char_start = match.start(1)
        else:
            # fallback: first standalone true/false
            match2 = re.search(r'\b(true|false)\b', lowered)
            if match2:
                observed_bool = match2.group(1)
                bool_char_start = match2.start(1)

        if observed_bool is None or bool_char_start is None:
            return {"note": "boolean_not_found_in_output"}

        # Map char index to token index
        t_idx = None
        for i, (s, e) in enumerate(zip(starts, ends)):
            if s <= bool_char_start < e:
                t_idx = i
                break
        if t_idx is None:
            # choose closest preceding
            for i in reversed(range(len(starts))):
                if starts[i] <= bool_char_start:
                    t_idx = i
                    break
        if t_idx is None:
            return {"note": "token_index_not_found"}

        # Ensure token id sets are ready
        self._ensure_boolean_token_sets()
        true_ids = self._true_token_ids or set()
        false_ids = self._false_token_ids or set()

        # Align score step with token index; clamp to bounds
        s_idx = max(0, min(t_idx, len(scores_list) - 1))
        step_scores = scores_list[s_idx]
        if isinstance(step_scores, torch.Tensor) and step_scores.ndim == 2:
            step_scores = step_scores[0]
        if not isinstance(step_scores, torch.Tensor) or step_scores.ndim != 1:
            return {"note": "unexpected_scores_shape"}

        probs = torch.softmax(step_scores.float(), dim=-1)
        vocab = probs.shape[0]
        p_true = float(sum(probs[i].item() for i in true_ids if 0 <= i < vocab))
        p_false = float(sum(probs[i].item() for i in false_ids if 0 <= i < vocab))
        denom = p_true + p_false
        if denom <= 0:
            conf = None
        else:
            if observed_bool == "true":
                conf = p_true / denom
            else:
                conf = p_false / denom
        return {
            "boundary_token_index": int(t_idx),
            "observed": observed_bool,
            "p_true_first_token": p_true,
            "p_false_first_token": p_false,
            "logit_confidence": conf,
            "true_first_token_ids": sorted(list(true_ids)),
            "false_first_token_ids": sorted(list(false_ids)),
        }

    def _extract_scores_list(self, generated: Any) -> Tuple[Optional[List[torch.Tensor]], str]:
        """Extract the list of per-step scores/logits from generate() output with provenance.

        Returns (scores_list, source) where source describes where it was found.
        """
        # Direct attribute
        scores = getattr(generated, "scores", None)
        if scores is not None:
            return scores, "generated.scores"
        # Dict container
        if isinstance(generated, dict) and "scores" in generated:
            return generated["scores"], "dict.scores"
        # Tuple/list: check first few elements for objects/dicts with scores
        if isinstance(generated, (list, tuple)):
            for i, item in enumerate(generated[:5]):
                s = getattr(item, "scores", None)
                if s is not None:
                    return s, f"tuple[{i}].scores"
                if isinstance(item, dict) and "scores" in item:
                    return item["scores"], f"tuple[{i}].dict.scores"
        return None, "not_found"

    def _find_boolean_token_index(self, generated_ids: torch.Tensor) -> Dict[str, Any]:
        """Locate the first boolean token ('true'/'false') position within generated ids.

        Returns a dict with keys: observed (str or None), token_index (int or None), note.
        """
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or not isinstance(generated_ids, torch.Tensor) or generated_ids.ndim != 2:
            return {"observed": None, "token_index": None, "note": "invalid_inputs"}

        token_ids = generated_ids[0].tolist()
        token_pieces: List[str] = []
        for tid in token_ids:
            try:
                token_pieces.append(tokenizer.decode([int(tid)], skip_special_tokens=True))
            except Exception:
                token_pieces.append("")
        cumulative = ""
        starts: List[int] = []
        ends: List[int] = []
        for piece in token_pieces:
            starts.append(len(cumulative))
            cumulative += piece
            ends.append(len(cumulative))

        lowered = cumulative.lower()
        match = re.search(r'"boundary"\s*:\s*(true|false)', lowered)
        observed_bool = None
        bool_char_start = None
        if match:
            observed_bool = match.group(1)
            bool_char_start = match.start(1)
        else:
            match2 = re.search(r'\b(true|false)\b', lowered)
            if match2:
                observed_bool = match2.group(1)
                bool_char_start = match2.start(1)

        if observed_bool is None or bool_char_start is None:
            return {"observed": None, "token_index": None, "note": "boolean_not_found_in_output"}

        t_idx = None
        for i, (s, e) in enumerate(zip(starts, ends)):
            if s <= bool_char_start < e:
                t_idx = i
                break
        if t_idx is None:
            for i in reversed(range(len(starts))):
                if starts[i] <= bool_char_start:
                    t_idx = i
                    break
        return {"observed": observed_bool, "token_index": t_idx, "note": "ok"}

    def _compute_logit_confidence_via_forward(
        self,
        sequences: torch.Tensor,
        input_ids_tensor: Optional[torch.Tensor],
        inputs: Dict[str, Any],
        token_index_in_generated: int,
        observed_bool: str,
    ) -> Dict[str, Any]:
        """Compute boolean confidence by a single forward pass at the decision step.

        We run the model on the prefix up to the boolean token and take the next-token logits.
        """
        try:
            if sequences is None or sequences.ndim != 2 or input_ids_tensor is None:
                return {"status": "error", "reason": "invalid_sequences_or_input_ids"}
            full_step = int(input_ids_tensor.size(1)) + int(token_index_in_generated)
            full_step = max(1, min(full_step, int(sequences.size(1)) - 1))
            prefix_ids = sequences[:, :full_step]

            eval_inputs: Dict[str, Any] = {}
            # Preserve multimodal fields
            for k, v in inputs.items():
                if k not in ("input_ids", "attention_mask"):
                    eval_inputs[k] = v
            eval_inputs["input_ids"] = prefix_ids
            # Ensure attention mask exists and matches shape
            attn = inputs.get("attention_mask") if isinstance(inputs, dict) else None
            if isinstance(attn, torch.Tensor) and attn.ndim == 2 and attn.size(1) >= prefix_ids.size(1):
                eval_inputs["attention_mask"] = attn[:, :prefix_ids.size(1)]
            else:
                eval_inputs["attention_mask"] = torch.ones_like(prefix_ids, dtype=torch.long, device=prefix_ids.device)

            with torch.no_grad():
                out = self.model(**eval_inputs, return_dict=True, use_cache=False)
            logits = getattr(out, "logits", None)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                return {"status": "error", "reason": "logits_unavailable"}
            step_logits = logits[:, -1, :]  # (B, V)
            if step_logits.ndim != 2:
                return {"status": "error", "reason": "unexpected_step_logits_shape"}
            probs = torch.softmax(step_logits[0].float(), dim=-1)

            # Ensure token sets
            self._ensure_boolean_token_sets()
            true_ids = self._true_token_ids or set()
            false_ids = self._false_token_ids or set()
            vocab = probs.shape[0]
            p_true = float(sum(probs[i].item() for i in true_ids if 0 <= i < vocab))
            p_false = float(sum(probs[i].item() for i in false_ids if 0 <= i < vocab))
            denom = p_true + p_false
            conf = None
            if denom > 0:
                conf = p_true / denom if observed_bool == "true" else p_false / denom
            return {
                "status": "ok",
                "method": "forward",
                "boundary_token_full_index": int(full_step),
                "p_true_first_token": p_true,
                "p_false_first_token": p_false,
                "logit_confidence": conf,
            }
        except Exception as e:
            return {"status": "error", "reason": f"exception: {type(e).__name__}"}

    def sliding_window_detection(
        self,
        video_path: str,
        sample_fps: float = 2.0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        window_size: int = 8,  # Default to 8 for 4+4 audio split
        stride: int = 2,
        vote_threshold: int = 1,
        merge_distance_sec: float = 0.5,
        output_dir: str = "outputs",
        debug_dir: Optional[str] = None,
    ) -> Dict:
        """
        Sliding window boundary detection using Qwen3-Omni with audio-visual analysis.

        When debug_dir is provided, only the first window is processed and assets are
        dumped to that directory; the returned result will reflect the debug placeholder
        and no boundaries are computed.
        """
        os.makedirs(output_dir, exist_ok=True)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        
        # Sample frames
        frames, frame_indices, video_fps, (start_frame, end_frame) = self.sample_frames(
            video_path, sample_fps, start_time, end_time
        )
        n_frames = len(frames)
        
        if n_frames < window_size:
            print(f"[WARN] Video too short ({n_frames} frames) for window size {window_size}")
            window_size = n_frames
        
        # Ensure even window size for clean split
        if window_size % 2 != 0:
            window_size += 1
            
        results = []
        boundary_votes = defaultdict(lambda: {
            "votes": 0, 
            "max_confidence": 0,
            "visual_cues": [],
            "audio_cues": []
        })
        
        # Sliding windows
        print(f"[INFO] Processing {n_frames} frames with window_size={window_size}, stride={stride}")
        print(f"[INFO] Using audio-visual analysis with Qwen3-Omni")

        total_candidates = max(1, n_frames - window_size + 1)
        total_windows = ((total_candidates - 1) // stride) + 1
        pbar = tqdm(total=total_windows, desc="Analyzing audio-visual windows", unit="win")

        for start_idx in range(0, total_candidates, stride):
            if self.skip_last_window and start_idx >= total_candidates - 1:
                print("[INFO] Skipping final window to avoid edge-case audio extraction")
                break

            end_idx = min(start_idx + window_size, n_frames)
            window_frames = frames[start_idx:end_idx]
            window_frame_indices = frame_indices[start_idx:end_idx]

            if debug_dir:
                window_debug_dir = os.path.join(debug_dir, f"window_{start_idx:05d}")
                os.makedirs(window_debug_dir, exist_ok=True)
            else:
                window_debug_dir = None

            # Pad if necessary
            while len(window_frames) < window_size:
                window_frames.append(window_frames[-1])
                window_frame_indices.append(window_frame_indices[-1])

            # Center boundary indices
            right_local_idx = window_size // 2
            left_local_idx = right_local_idx - 1 if right_local_idx > 0 else 0
            global_left_idx = start_idx + left_local_idx
            global_right_idx = start_idx + right_local_idx

            # Ask model about boundary with audio
            try:
                inspect_dir = None
                if not debug_dir:
                    # always create per-window inspect dir under output_dir
                    inspect_dir = os.path.join(output_dir, "omni_debug", f"window_{start_idx:05d}")
                    os.makedirs(inspect_dir, exist_ok=True)
                result = self.ask_boundary_with_audio(
                    window_frames,
                    video_path,
                    window_frame_indices,
                    left_local_idx,
                    right_local_idx,
                    video_fps,
                    debug_dir=window_debug_dir,
                    inspect_dir=inspect_dir,
                )
            except Exception as e:
                print(f"\n[ERROR] Failed at window starting {start_idx}: {e}")
                print(traceback.format_exc())
                result = {
                    "boundary": False,
                    "confidence": 0.0,
                    "visual_cue": "error",
                    "audio_cue": "error",
                }

            results.append(
                {
                    "window_start": start_idx,
                    "global_left": global_left_idx,
                    "global_right": global_right_idx,
                    "result": result,
                }
            )

            if debug_dir:
                print(
                    f"[INFO] Debug assets saved to {os.path.join(debug_dir, f'window_{start_idx:05d}')}"
                )
                break

            # Vote if boundary detected
            if result.get("boundary", False):
                pair = (global_left_idx, global_right_idx)
                boundary_votes[pair]["votes"] += 1
                boundary_votes[pair]["max_confidence"] = max(
                    boundary_votes[pair]["max_confidence"],
                    result.get("confidence", 0.0)
                )
                if result.get("visual_cue") and result["visual_cue"] != "error":
                    boundary_votes[pair]["visual_cues"].append(result["visual_cue"])
                if result.get("audio_cue") and result["audio_cue"] != "error":
                    boundary_votes[pair]["audio_cues"].append(result["audio_cue"])
            
            pbar.update(1)
        pbar.close()
        
        # Clean up temp audio files
        for audio_file in self.temp_audio_files:
            try:
                os.unlink(audio_file)
            except:
                pass
        self.temp_audio_files = []
        
        # Aggregate votes into boundaries
        boundary_pairs = [
            (pair, data) for pair, data in boundary_votes.items()
            if data["votes"] >= vote_threshold
        ]
        boundary_pairs.sort(key=lambda x: x[0])

        # Merge nearby boundaries
        merged_boundaries = []
        merge_distance_frames = int(merge_distance_sec * sample_fps)

        for (left_idx, right_idx), data in boundary_pairs:
            center_idx = (left_idx + right_idx) / 2.0
            if not merged_boundaries:
                merged_boundaries.append({
                    "center": center_idx,
                    "data": data
                })
            elif center_idx - merged_boundaries[-1]["center"] <= merge_distance_frames:
                # Merge with previous boundary
                prev = merged_boundaries[-1]
                prev["center"] = (prev["center"] + center_idx) / 2.0
                prev["data"]["votes"] += data["votes"]
                prev["data"]["max_confidence"] = max(prev["data"]["max_confidence"], data["max_confidence"])
                prev["data"]["visual_cues"].extend(data["visual_cues"])
                prev["data"]["audio_cues"].extend(data["audio_cues"])
            else:
                merged_boundaries.append({
                    "center": center_idx,
                    "data": data
                })

        # Convert to timestamps
        boundary_times = []
        boundary_details = []
        for boundary in merged_boundaries:
            int_center = int(round(boundary["center"]))
            if 0 <= int_center < len(frame_indices):
                time_sec = frame_indices[int_center] / video_fps
                boundary_times.append(time_sec)
                boundary_details.append({
                    "time": time_sec,
                    "frame_index": int_center,
                    "confidence": boundary["data"]["max_confidence"],
                    "votes": boundary["data"]["votes"],
                    "visual_cues": list(set(boundary["data"]["visual_cues"]))[:3],  # Top 3 unique
                    "audio_cues": list(set(boundary["data"]["audio_cues"]))[:3]
                })

        # Save results
        output_data = {
            "video_path": video_path,
            "model": self.model_name,
            "fps": video_fps,
            "sample_fps": sample_fps,
            "start_time": start_time,
            "end_time": end_time,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "n_frames_analyzed": n_frames,
            "window_size": window_size,
            "stride": stride,
            "vote_threshold": vote_threshold,
            "merge_distance_sec": merge_distance_sec,
            "boundaries": boundary_details,
            "boundary_times": boundary_times,
            "n_boundaries": len(boundary_times)
        }

        # Save main output
        output_file = os.path.join(output_dir, "av_boundaries.json")
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        # Save simple text format
        txt_file = os.path.join(output_dir, "boundary_times.txt")
        with open(txt_file, "w") as f:
            f.write("# Event boundaries detected with audio-visual analysis\n")
            f.write("# Time(s)\tConfidence\tVisual_Cue\tAudio_Cue\n")
            for detail in boundary_details:
                visual_cue = detail["visual_cues"][0] if detail["visual_cues"] else "none"
                audio_cue = detail["audio_cues"][0] if detail["audio_cues"] else "none"
                f.write(f"{detail['time']:.3f}\t{detail['confidence']:.2f}\t{visual_cue}\t{audio_cue}\n")
        
        print(f"\n[INFO] Found {len(boundary_times)} boundaries using audio-visual analysis")
        print(f"[INFO] Results saved to {output_dir}/")
        
        # Print summary
        if boundary_details:
            print("\n[INFO] Detected boundaries:")
            for detail in boundary_details[:10]:  # Show first 10
                print(f"  {detail['time']:.2f}s (conf: {detail['confidence']:.2f})")
                if detail['visual_cues']:
                    print(f"    Visual: {detail['visual_cues'][0]}")
                if detail['audio_cues']:
                    print(f"    Audio: {detail['audio_cues'][0]}")
        
        return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-Omni audio-visual temporal segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python %(prog)s video.mp4
  
  # Custom window size and sample rate
  python %(prog)s video.mp4 --window-size 8 --sample-fps 3.0
  
  # Process specific time range
  python %(prog)s video.mp4 --start-time 10 --end-time 60
  
  # Use different model variant
  python %(prog)s video.mp4 --model Qwen/Qwen3-Omni-30B-A3B-Thinking
        """
    )
    
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                       help="Model name (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)")
    parser.add_argument("--sample-fps", type=float, default=25,
                       help="Frame sampling rate in FPS (default: 2.0)")
    parser.add_argument("--start-time", type=float, default=None,
                       help="Start time in seconds (inclusive)")
    parser.add_argument("--end-time", type=float, default=None,
                       help="End time in seconds (inclusive)")
    parser.add_argument("--window-size", type=int, default=8,
                       help="Window size in frames (default: 8, will be made even)")
    parser.add_argument("--stride", type=int, default=1,
                       help="Stride for sliding window (default: 1)")
    parser.add_argument("--vote-threshold", type=int, default=1,
                       help="Minimum votes to confirm boundary (default: 1)")
    parser.add_argument("--merge-distance", type=float, default=0.5,
                       help="Merge boundaries within this many seconds (default: 0.5)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory for results (default: outputs)")
    parser.add_argument("--debug-save", type=str, default=None,
                       help="Directory to dump a single window's frames, audio, and prompt without running full detection")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        return 1
    
    # Initialize segmenter
    try:
        segmenter = QwenOmniAudioVisualSegmenter(model_name=args.model)
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
        return 1
    
    if args.debug_save:
        print(f"[INFO] Debug mode enabled. Assets will be saved to {args.debug_save}.")
    
    # Run detection
    try:
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
            debug_dir=args.debug_save,
        )

        if args.debug_save:
            print("\n[INFO] Debug extraction complete. Inspect the saved prompt, audio, and frames.")
            return 0
        
        print(f"\n[SUCCESS] Analysis complete. Found {results['n_boundaries']} boundaries.")
        
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
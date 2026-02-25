#!/usr/bin/env python3
"""
qwen3_omni_video_understander.py
Detailed video understanding using Qwen3-Omni with audio-visual analysis
"""

import argparse
import os
import json
import traceback
import tempfile
import subprocess
import shutil
from typing import List, Tuple, Dict, Optional, Any
from io import BytesIO
import base64

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


class QwenOmniVideoUnderstander:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        device: Optional[str] = None,
        frame_downsample_scale: float = 0.5,
    ):
        """
        Initialize with Qwen3-Omni model for detailed video understanding
        """
        self.model_name = model_name
        self.device = device or device_name()
        self.frame_downsample_scale = frame_downsample_scale if frame_downsample_scale > 0 else 1.0
        print(f"[INFO] Using device: {self.device}")

        # Configuration
        self.audio_sample_rate = 16000
        self.min_audio_duration = 0.1
        
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
        
        # Generation config for detailed descriptions
        self.generation_config = {
            "max_new_tokens": 768,
            "min_new_tokens": 64,
            "temperature": 0.2,
            "do_sample": False,
            "top_p": 0.9,
        }
        
        # Temp file tracking
        self.temp_audio_files = []
        self.temp_video_files = []

    def create_silent_audio_segment(self, duration_sec: float, dest_path: Optional[str] = None) -> str:
        """Create a silent audio segment."""
        duration_sec = max(duration_sec, self.min_audio_duration)
        silent_audio = np.zeros(int(self.audio_sample_rate * duration_sec), dtype=np.float32)

        if dest_path:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True) if os.path.dirname(dest_path) else None
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

    def sample_frames(self, video_path: str, sample_fps: float = 2.0,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None) -> Tuple[List[Image.Image], List[int], float, Tuple[int, int]]:
        """Sample frames from video at specified fps within optional time range."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
            
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        start_frame = 0 if start_time is None else max(0, int(round(start_time * video_fps)))
        end_frame = (total_frames - 1) if end_time is None else min(total_frames - 1, int(round(end_time * video_fps)))
        
        if total_frames == 0 or start_frame >= total_frames:
            cap.release()
            return [], [], video_fps, (start_frame, start_frame)
        if end_frame < start_frame:
            end_frame = start_frame
        
        frame_interval = max(1, int(round(video_fps / sample_fps)))
        
        frames = []
        frame_indices = []
        
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        idx = start_frame
        while True:
            ret, frame = cap.read()
            if not ret or idx > end_frame:
                break
            if (idx - start_frame) % frame_interval == 0:
                if 0 < self.frame_downsample_scale < 1.0:
                    frame = cv2.resize(
                        frame,
                        (0, 0),
                        fx=self.frame_downsample_scale,
                        fy=self.frame_downsample_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(pil_frame)
                frame_indices.append(idx)
            idx += 1
            
        cap.release()
        print(f"[INFO] Sampled {len(frames)} frames from video (fps: {video_fps:.2f})")
        return frames, frame_indices, video_fps, (start_frame, end_frame)

    def encode_frame_to_base64(self, frame: Image.Image) -> str:
        """Convert PIL image to base64 for Qwen3-Omni input"""
        buffered = BytesIO()
        frame.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def extract_audio_segment(self, video_path: str, start_frame: int, end_frame: int, 
                             video_fps: float, dest_path: Optional[str] = None) -> str:
        """Extract audio segment for frame range."""
        if max(video_fps, 1e-6) == 0:
            raise ValueError("Video FPS must be greater than zero")
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        start_time = max(start_frame / max(video_fps, 1e-6), 0.0)
        end_time = max(end_frame / max(video_fps, 1e-6), start_time)
        duration = max(end_time - start_time, self.min_audio_duration)

        if dest_path:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True) if os.path.dirname(dest_path) else None
            temp_audio_path = dest_path
            created_temp = False
        else:
            fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            created_temp = True

        # Try ffmpeg first
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-ss", f"{start_time:.3f}", "-t", f"{duration:.3f}",
                "-i", video_path, "-ac", "1", "-ar", str(self.audio_sample_rate),
                "-y", temp_audio_path
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if created_temp:
                    self.temp_audio_files.append(temp_audio_path)
                return temp_audio_path
            except subprocess.CalledProcessError:
                pass

        # Fallback to librosa
        try:
            audio, _ = librosa.load(video_path, sr=self.audio_sample_rate, 
                                   offset=start_time, duration=duration, mono=True)
            if len(audio) < int(self.audio_sample_rate * self.min_audio_duration):
                target_length = int(self.audio_sample_rate * self.min_audio_duration)
                audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
            sf.write(temp_audio_path, audio.astype(np.float32), self.audio_sample_rate)
        except Exception:
            # Silent fallback
            silent_samples = max(int(round(self.audio_sample_rate * duration)), 1)
            sf.write(temp_audio_path, np.zeros(silent_samples, dtype=np.float32), self.audio_sample_rate)

        if created_temp:
            self.temp_audio_files.append(temp_audio_path)
        return temp_audio_path

    def extract_video_clip(self, video_path: str, start_frame: int, end_frame: int,
                          video_fps: float, dest_path: Optional[str] = None) -> str:
        """Extract video clip for frame range."""
        start_time = max(start_frame / max(video_fps, 1e-6), 0.0)
        end_time = max(end_frame / max(video_fps, 1e-6), start_time)
        duration = max(end_time - start_time, 0.1)

        if dest_path:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True) if os.path.dirname(dest_path) else None
            temp_video_path = dest_path
            created_temp = False
        else:
            fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            created_temp = True

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found - required for video clip extraction")

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_time:.3f}", "-t", f"{duration:.3f}",
            "-i", video_path, "-c", "copy", "-y", temp_video_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            # Fallback: re-encode if copy fails
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-ss", f"{start_time:.3f}", "-t", f"{duration:.3f}",
                "-i", video_path, "-c:v", "libx264", "-c:a", "aac",
                "-y", temp_video_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if created_temp:
            self.temp_video_files.append(temp_video_path)
        return temp_video_path

    def _build_understanding_prompt(self, window_duration_sec: float, use_video: bool = False) -> str:
        """Build prompt for detailed video understanding with strict JSON-only output."""
        modality_desc = "video clip" if use_video else "frames and audio"
        
        return (
            f"You are analyzing a {window_duration_sec:.1f} second {modality_desc} segment from a longer video.\n\n"
            "TASK: Provide a detailed, factual account of what happens in this segment.\n\n"
            "Include:\n"
            "- VISUAL CONTENT: setting, people/objects, actions, camera movement, lighting/style\n"
            "- AUDIO CONTENT: exact dialogue (transcribe), background sounds/music, tone/mood, notable effects\n"
            "- OVERALL NARRATIVE: the event/action, contribution to story, transitions/changes\n\n"
            "OUTPUT REQUIREMENTS (STRICT):\n"
            "- Return ONLY a single JSON object with exactly these keys: scene_description, audio_transcription, overall_summary\n"
            "- No extra text before or after the JSON\n"
            "- Do not copy example text; fill with content derived from THIS segment\n"
            "- Be specific and avoid generic placeholders\n"
        )

    def understand_window_frames_audio(self, frames: List[Image.Image], audio_path: str,
                                       window_duration_sec: float,
                                       debug_dir: Optional[str] = None,
                                       inspect_dir: Optional[str] = None) -> Dict[str, Any]:
        """Analyze window using frames + audio approach."""
        if not frames:
            return {
                "scene_description": "No frames available",
                "audio_transcription": "No audio available",
                "overall_summary": "Empty window"
            }

        try:
            # Prepare prompt
            prompt_text = self._build_understanding_prompt(window_duration_sec, use_video=False)
            
            # Build messages with frames and audio
            frame_urls = [self.encode_frame_to_base64(f) for f in frames]
            messages = [{
                "role": "user",
                "content": [
                    *[{"type": "image", "image": url} for url in frame_urls],
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt_text},
                ],
            }]

            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                with open(os.path.join(debug_dir, "prompt.txt"), "w") as f:
                    f.write(prompt_text)
                for idx, frame in enumerate(frames):
                    frame.save(os.path.join(debug_dir, f"frame_{idx:02d}.jpg"))
                # do not early return; still run the model and persist outputs

            # Process inputs
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audio_lists, images, videos = process_mm_info(messages, use_audio_in_video=False)
            audios = [clip for per_message in audio_lists for clip in per_message] or None

            inputs = self.processor(
                text=text, audio=audios, images=images, videos=None,
                return_tensors="pt", padding=True, use_audio_in_video=False
            )
            inputs = inputs.to(self.model.device)
            
            # Cast to model dtype
            target_dtype = self.model_dtype
            if target_dtype in (torch.float16, torch.bfloat16):
                for key, value in list(inputs.items()):
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        inputs[key] = value.to(target_dtype)

            # Generate
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    return_dict_in_generate=True,
                    **self.generation_config,
                )

            # Decode robustly regardless of generate() return type
            response = self._decode_generate_output(output, inputs)
            # Persist generation artifacts if requested
            save_dir = inspect_dir or debug_dir
            if save_dir:
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    summary = self._summarize_generated(output, inputs.get("input_ids") if isinstance(inputs, dict) else None)
                    with open(os.path.join(save_dir, "generate_return.json"), "w") as fsum:
                        json.dump(summary, fsum, indent=2)
                    with open(os.path.join(save_dir, "model_response.txt"), "w") as fr:
                        fr.write(response)
                except Exception:
                    pass
            
            # Parse JSON response
            result = self._parse_understanding_response(response)
            return result

        except Exception as e:
            print(f"[WARN] Frame+audio analysis failed: {e}")
            print(traceback.format_exc())
            return {
                "scene_description": f"Error: {str(e)}",
                "audio_transcription": "Error during processing",
                "overall_summary": "Analysis failed"
            }

    def understand_window_video(self, video_clip_path: str, window_duration_sec: float,
                               debug_dir: Optional[str] = None,
                               inspect_dir: Optional[str] = None) -> Dict[str, Any]:
        """Analyze window using video clip approach."""
        try:
            # Prepare prompt
            prompt_text = self._build_understanding_prompt(window_duration_sec, use_video=True)
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_clip_path},
                    {"type": "text", "text": prompt_text},
                ],
            }]

            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                with open(os.path.join(debug_dir, "prompt.txt"), "w") as f:
                    f.write(prompt_text)
                shutil.copy(video_clip_path, os.path.join(debug_dir, "clip.mp4"))
                # do not early return; still run the model and persist outputs

            # Process inputs
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audio_lists, images, videos = process_mm_info(messages, use_audio_in_video=True)
            
            inputs = self.processor(
                text=text, videos=videos, images=images, audio=None,
                return_tensors="pt", padding=True, use_audio_in_video=True
            )
            inputs = inputs.to(self.model.device)
            
            # Cast to model dtype
            target_dtype = self.model_dtype
            if target_dtype in (torch.float16, torch.bfloat16):
                for key, value in list(inputs.items()):
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        inputs[key] = value.to(target_dtype)

            # Generate
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    return_dict_in_generate=True,
                    **self.generation_config,
                )

            # Decode robustly regardless of generate() return type
            response = self._decode_generate_output(output, inputs)
            
            # Parse JSON response
            result = self._parse_understanding_response(response)
            # Persist generation artifacts if requested
            save_dir = inspect_dir or debug_dir
            if save_dir:
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    summary = self._summarize_generated(output, inputs.get("input_ids") if isinstance(inputs, dict) else None)
                    with open(os.path.join(save_dir, "generate_return.json"), "w") as fsum:
                        json.dump(summary, fsum, indent=2)
                    with open(os.path.join(save_dir, "model_response.txt"), "w") as fr:
                        fr.write(response)
                except Exception:
                    pass
            return result

        except Exception as e:
            print(f"[WARN] Video clip analysis failed: {e}")
            print(traceback.format_exc())
            return {
                "scene_description": f"Error: {str(e)}",
                "audio_transcription": "Error during processing",
                "overall_summary": "Analysis failed"
            }

    def _parse_understanding_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON response from model."""
        # Try to find JSON object
        import re
        json_match = re.search(r'\{[^}]*\}', text, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                # Ensure all fields exist
                result.setdefault("scene_description", "")
                result.setdefault("audio_transcription", "")
                result.setdefault("overall_summary", "")
                return result
            except json.JSONDecodeError:
                pass
        
        # Fallback: use raw text
        return {
            "scene_description": text[:500],
            "audio_transcription": "",
            "overall_summary": text[:200]
        }

    def _extract_generated_sequences(self, generated: Any) -> torch.Tensor:
        """Select the sequences tensor from diverse generate() return structures."""
        candidate_tensors: List[torch.Tensor] = []
        # Direct tensor
        if torch.is_tensor(generated):
            candidate_tensors.append(generated)
        # Attribute on object
        if hasattr(generated, "sequences") and torch.is_tensor(getattr(generated, "sequences")):
            candidate_tensors.append(getattr(generated, "sequences"))
        # Dict container
        if isinstance(generated, dict) and "sequences" in generated and torch.is_tensor(generated["sequences"]):
            candidate_tensors.append(generated["sequences"])
        # Tuple/list containers
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
            raise TypeError("Unable to locate sequences tensor from generate() return value")
        if sequences.ndim == 1:
            sequences = sequences.unsqueeze(0)
        return sequences

    def _decode_generate_output(self, generated: Any, inputs: Dict[str, Any]) -> str:
        """Decode text from generate() output by locating sequences and removing the prompt tokens."""
        sequences = self._extract_generated_sequences(generated)
        input_ids_tensor = inputs["input_ids"] if isinstance(inputs, dict) and "input_ids" in inputs else None
        if isinstance(input_ids_tensor, torch.Tensor) and sequences.size(1) >= input_ids_tensor.size(1):
            generated_ids = sequences[:, input_ids_tensor.size(1):]
        else:
            generated_ids = sequences
        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0] if decoded else ""
    def _summarize_generated(self, gen_obj: Any, input_ids_tensor: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Summarize the structure of generate() return for debugging."""
        def tensor_info(t: torch.Tensor) -> Dict[str, Any]:
            return {
                "is_tensor": True,
                "dtype": str(t.dtype),
                "ndim": int(t.ndim),
                "shape": list(t.shape),
            }
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
        except Exception:
            pass
        try:
            summary["input_ids"] = tensor_info(input_ids_tensor) if isinstance(input_ids_tensor, torch.Tensor) else None
        except Exception:
            pass
        return summary

    def analyze_video(self, video_path: str, sample_fps: float = 2.0,
                     start_time: Optional[float] = None, end_time: Optional[float] = None,
                     window_size: int = 12, stride: int = 6,
                     output_dir: str = "outputs", use_video_clips: bool = False,
                     debug_dir: Optional[str] = None) -> Dict:
        """
        Analyze video in sliding windows and generate detailed descriptions.
        
        Args:
            video_path: Path to video file
            sample_fps: Frame sampling rate (frames/second)
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            window_size: Number of frames per window
            stride: Number of frames to slide between windows
            output_dir: Directory for output files
            use_video_clips: If True, use video clip approach; if False, use frames+audio
            debug_dir: If provided, only process first window and save debug info
        
        Returns:
            Dictionary with analysis results
        """
        os.makedirs(output_dir, exist_ok=True)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        
        # Sample frames to determine windows
        frames, frame_indices, video_fps, (start_frame, end_frame) = self.sample_frames(
            video_path, sample_fps, start_time, end_time
        )
        n_frames = len(frames)
        
        if n_frames < window_size:
            print(f"[WARN] Video has {n_frames} frames, adjusting window_size")
            window_size = n_frames
        
        # Calculate window duration in seconds
        # window_size frames at sample_fps = window_size / sample_fps seconds
        window_duration_sec = window_size / sample_fps
        
        print(f"[INFO] Window duration: {window_duration_sec:.2f} seconds ({window_size} frames at {sample_fps} fps)")
        print(f"[INFO] Analysis mode: {'Video clips' if use_video_clips else 'Frames + audio'}")
        
        results = []
        
        # Calculate total windows
        total_windows = max(1, ((n_frames - window_size) // stride) + 1)
        pbar = tqdm(total=total_windows, desc="Analyzing video", unit="window")
        
        for start_idx in range(0, n_frames - window_size + 1, stride):
            end_idx = min(start_idx + window_size, n_frames)
            window_frames = frames[start_idx:end_idx]
            window_frame_indices = frame_indices[start_idx:end_idx]
            
            # Calculate time position
            window_start_frame = window_frame_indices[0]
            window_end_frame = window_frame_indices[-1]
            window_start_time = window_start_frame / video_fps
            window_end_time = window_end_frame / video_fps
            
            window_info = {
                "window_index": len(results),
                "start_time": round(window_start_time, 2),
                "end_time": round(window_end_time, 2),
                "duration": round(window_end_time - window_start_time, 2),
            }
            
            # Debug mode
            if debug_dir:
                window_debug_dir = os.path.join(debug_dir, f"window_{start_idx:05d}")
            else:
                window_debug_dir = None
            
            # Analyze based on mode
            try:
                if use_video_clips:
                    # Extract video clip
                    clip_path = self.extract_video_clip(
                        video_path, window_start_frame, window_end_frame, video_fps,
                        dest_path=os.path.join(window_debug_dir, "clip.mp4") if window_debug_dir else None
                    )
                    inspect_dir = None
                    if not debug_dir:
                        inspect_dir = os.path.join(output_dir, "omni_debug", f"window_{start_idx:05d}")
                        os.makedirs(inspect_dir, exist_ok=True)
                    understanding = self.understand_window_video(
                        clip_path, window_duration_sec, debug_dir=window_debug_dir, inspect_dir=inspect_dir
                    )
                else:
                    # Extract audio
                    audio_path = self.extract_audio_segment(
                        video_path, window_start_frame, window_end_frame, video_fps,
                        dest_path=os.path.join(window_debug_dir, "audio.wav") if window_debug_dir else None
                    )
                    inspect_dir = None
                    if not debug_dir:
                        inspect_dir = os.path.join(output_dir, "omni_debug", f"window_{start_idx:05d}")
                        os.makedirs(inspect_dir, exist_ok=True)
                    understanding = self.understand_window_frames_audio(
                        window_frames, audio_path, window_duration_sec, debug_dir=window_debug_dir, inspect_dir=inspect_dir
                    )
                
                window_info.update(understanding)
                results.append(window_info)
                
                if debug_dir:
                    print(f"[INFO] Debug assets saved to {window_debug_dir}")
                    break
                
            except Exception as e:
                print(f"\n[ERROR] Failed at window {start_idx}: {e}")
                window_info.update({
                    "scene_description": f"Error: {str(e)}",
                    "audio_transcription": "Error",
                    "overall_summary": "Processing failed"
                })
                results.append(window_info)
            
            pbar.update(1)
        
        pbar.close()
        
        # Clean up temp files
        for f in self.temp_audio_files:
            try: os.unlink(f)
            except: pass
        for f in self.temp_video_files:
            try: os.unlink(f)
            except: pass
        self.temp_audio_files = []
        self.temp_video_files = []
        
        # Build output
        output_data = {
            "video_path": video_path,
            "model": self.model_name,
            "video_fps": video_fps,
            "sample_fps": sample_fps,
            "window_size": window_size,
            "window_duration_sec": round(window_duration_sec, 2),
            "stride": stride,
            "analysis_mode": "video_clips" if use_video_clips else "frames_audio",
            "start_time": start_time,
            "end_time": end_time,
            "total_windows": len(results),
            "windows": results
        }
        
        # Save JSON output
        output_file = os.path.join(output_dir, "video_understanding.json")
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n[SUCCESS] Analyzed {len(results)} windows")
        print(f"[INFO] Results saved to {output_file}")
        
        return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-Omni detailed video understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with frames+audio (default)
  python %(prog)s video.mp4
  
  # Use video clip approach instead
  python %(prog)s video.mp4 --use-video-clips
  
  # Custom window: 12 frames at 2fps = 6 second windows
  python %(prog)s video.mp4 --sample-fps 2 --window-size 12
  
  # Process specific time range with overlap
  python %(prog)s video.mp4 --start-time 10 --end-time 60 --stride 6
        """
    )
    
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                       help="Model name")
    parser.add_argument("--sample-fps", type=float, default=2.0,
                       help="Frame sampling rate (fps). Window duration = window_size / sample_fps")
    parser.add_argument("--window-size", type=int, default=12,
                       help="Number of frames per window (default: 12)")
    parser.add_argument("--stride", type=int, default=6,
                       help="Frames to slide between windows (default: 6)")
    parser.add_argument("--start-time", type=float, default=None,
                       help="Start time in seconds")
    parser.add_argument("--end-time", type=float, default=None,
                       help="End time in seconds")
    parser.add_argument("--use-video-clips", action="store_true",
                       help="Use video clip approach instead of frames+audio")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory (default: outputs)")
    parser.add_argument("--debug-save", type=str, default=None,
                       help="Save first window debug info to this directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        return 1
    
    # Initialize understander
    try:
        understander = QwenOmniVideoUnderstander(model_name=args.model)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return 1
    
    # Analyze
    try:
        results = understander.analyze_video(
            video_path=args.video,
            sample_fps=args.sample_fps,
            window_size=args.window_size,
            stride=args.stride,
            start_time=args.start_time,
            end_time=args.end_time,
            output_dir=args.output_dir,
            use_video_clips=args.use_video_clips,
            debug_dir=args.debug_save,
        )
        
        if args.debug_save:
            print("\n[INFO] Debug mode complete")
            return 0
        
        # Print summary
        print(f"\n[SUMMARY]")
        print(f"  Total windows: {results['total_windows']}")
        print(f"  Window duration: {results['window_duration_sec']} seconds")
        print(f"\nFirst few descriptions:")
        for window in results['windows'][:3]:
            print(f"\n  [{window['start_time']:.1f}s - {window['end_time']:.1f}s]")
            print(f"    {window['overall_summary'][:100]}...")
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
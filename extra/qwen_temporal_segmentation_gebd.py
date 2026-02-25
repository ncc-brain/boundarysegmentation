#!/usr/bin/env python3
"""
qwen_temporal_segmentation_gebd.py
Implementation for Qwen2-VL and Qwen2.5-VL temporal video segmentation
Updated with GEBD-aligned prompt for Kinetics-GEBD evaluation
"""

import argparse
import os
import json
import re
import traceback
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


def device_name():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class QwenTemporalSegmenterFixed:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: Optional[str] = None, prompt_type: str = "gebd_aligned"):
        """
        Initialize with Qwen2-VL and Qwen2.5-VL model loading.
        """
        self.model_name = model_name
        self.device = device or device_name()
        self.prompt_type = prompt_type
        print(f"[INFO] Using device: {self.device}")
        print(f"[INFO] Using prompt type: {self.prompt_type}")
        
        print(f"[INFO] Loading Qwen2.5-VL model: {model_name}")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Base generation config
        if prompt_type == "aggressive":
            self.generation_config = {
                "max_new_tokens": 256,
                "temperature": 0.9,
                "do_sample": True,
                "top_p": 0.9,
            }
        else:
            self.generation_config = {
                "max_new_tokens": 256,
                "temperature": 0.1,
                "do_sample": False,
                "top_p": 0.1,
            }

    def sample_frames(self, video_path: str, sample_fps: float = 2.0,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None) -> Tuple[List[Image.Image], List[int], float, Tuple[int, int]]:
        """Sample frames from video at specified fps within optional [start_time, end_time] seconds."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
            
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start_frame = 0 if start_time is None else max(0, int(round(start_time * video_fps)))
        end_frame = (total_frames - 1) if end_time is None else min(total_frames - 1, int(round(end_time * video_fps)))
        if end_frame < start_frame:
            raise ValueError("end_time must be greater than start_time")
        
        frame_interval = max(1, int(round(video_fps / sample_fps)))
        
        frames = []
        frame_indices = []
        idx = 0
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

        frame_urls = [self.encode_frame_to_base64(f) for f in frames]
        image_inputs = [{"type": "image", "image": url} for url in frame_urls]

        left_frame_num = left_idx + 1
        right_frame_num = right_idx + 1

        # Select prompt based on self.prompt_type
        if self.prompt_type == "gebd_aligned":
            text_prompt = (
                f"You are analyzing {num_frames} consecutive frames from a video showing one main activity/event.\n\n"
                f"Task: Determine if there is an EVENT BOUNDARY between frame {left_frame_num} and frame {right_frame_num}.\n\n"
                "CRITICAL: Focus on the DOMINANT SUBJECT (main person/object performing the activity).\n"
                "Ignore background activity and distractions.\n\n"
                "Mark a boundary when the dominant subject experiences:\n"
                "1. **Change of Action**: The action/activity changes (e.g., running→jumping, one push-up→next push-up)\n"
                "2. **Change of Subject**: A new dominant subject appears or the current one disappears\n"
                "3. **Change of Object**: The subject starts/stops interacting with a different object\n"
                "4. **Change in Environment**: Significant brightness, color, or lighting change affecting the subject\n"
                "5. **Shot Change**: Scene cut, camera angle shift, panning, zooming\n\n"
                "Granularity: Segment the video-level event into its natural SUB-UNITS.\n"
                "- For a 'long jump' video: mark boundaries between run→jump→land→stand\n"
                "- For 'cooking' video: mark boundaries between chop→mix→cook→serve\n"
                "Do NOT mark every tiny movement (too fine) or only start/end (too coarse).\n\n"
                f"Does a boundary exist between frames {left_frame_num} and {right_frame_num}?\n\n"
                "Respond with ONLY valid JSON:\n"
                "{\"boundary\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"which change type\"}"
            )
        elif self.prompt_type == "narrative":
            text_prompt = (
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
                "{\"boundary\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"narrative reason\"}"
            )
        elif self.prompt_type == "context":
            text_prompt = (
                f"Examine these {num_frames} video frames as a sequence.\n\n"
                "Questions to consider:\n"
                "1. Do the frames before and after the center represent the SAME SCENE or DIFFERENT SCENES?\n"
                "2. Is there a change in: setting/location, time period, story thread, or activity type?\n"
                "3. Would a viewer understand these as part of the same scene or separate scenes?\n\n"
                f"Focus on frames {left_frame_num} and {right_frame_num} as the potential boundary.\n"
                "Only mark boundaries between distinctly different scenes (like chapter breaks in a story).\n\n"
                "Respond with JSON only:\n"
                "{\"boundary\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"scene change type\"}"
            )
        elif self.prompt_type == "semantic":
            text_prompt = (
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
                "{\"boundary\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"type of transition\"}"
            )
        elif self.prompt_type == "aggressive":
            # Aggressive boundary finding: encourage creative micro-transitions in continuous actions
            text_prompt = (
                f"You are analyzing {num_frames} consecutive frames from a single continuous action/video.\n\n"
                f"Be AGGRESSIVE in hypothesizing EVENT BOUNDARIES between frame {left_frame_num} and frame {right_frame_num}.\n\n"
                "Treat subtle sub-action transitions as valid boundaries when plausible, for example:\n"
                "- Micro phase shifts in repetitive motions (e.g., down → up in a push-up/squat)\n"
                "- Grip/pose changes, tool/object contact on/off, handoffs\n"
                "- Tempo/force changes, direction reversals, prep → execute → follow-through\n"
                "- Camera micro-cut or jumpy motion suggesting edit\n"
                "- Onset/offset of audio-visual cues (impact, clap, lighting flicker)\n\n"
                "Prefer recall: when uncertain, lean toward marking a boundary if there is a reasonable cue.\n\n"
                f"Does a boundary likely exist between frames {left_frame_num} and {right_frame_num}?\n\n"
                "Respond with ONLY valid JSON:\n"
                "{\"boundary\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"brief cue\"}"
            )
        else:
            # Small Boundary Segmentation prompt (original default)
            text_prompt = (
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
                "3. Respond with ONLY valid JSON, no other text.\n\n"
                "Required JSON format:\n"
                "{\"boundary\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"brief explanation\"}\n\n"
                "Example responses:\n"
                "{\"boundary\": true, \"confidence\": 0.9, \"reason\": \"scene change from indoor to outdoor\"}\n"
                "{\"boundary\": false, \"confidence\": 0.8, \"reason\": \"continuous action\"}"
            )

        messages = [
            {
                "role": "user",
                "content": image_inputs + [{"type": "text", "text": text_prompt}]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        if not video_inputs or all((not v for v in video_inputs)):
            video_inputs = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_config
            )

        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return self.parse_json_response(response)

    def parse_json_response(self, text: str) -> Dict:
        """Robustly parse JSON from model response"""
        json_match = re.search(r'\{[^}]*\}', text, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group(0)
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                json_str = json_str.replace("true", "true").replace("false", "false")
                
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        text_lower = text.lower()
        result = {"boundary": False, "confidence": 0.5}
        
        if "boundary: true" in text_lower or '"boundary": true' in text_lower:
            result["boundary"] = True
            result["confidence"] = 0.7
        elif "boundary: false" in text_lower or '"boundary": false' in text_lower:
            result["boundary"] = False
            result["confidence"] = 0.7
            
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
        window_size: int = 7,
        stride: int = 1,
        vote_threshold: int = 1,
        merge_distance_sec: float = 0.5,
        output_dir: str = "outputs",
        # Debug options
        debug: bool = False,
        debug_dir: Optional[str] = None,
        debug_tag: Optional[str] = None,
        debug_save_images: bool = True
    ) -> Dict:
        """
        Sliding window boundary detection using native Qwen2.5-VL multi-frame input
        """
        os.makedirs(output_dir, exist_ok=True)
        
        frames, frame_indices, video_fps, (start_frame, end_frame) = self.sample_frames(
            video_path, sample_fps, start_time, end_time
        )
        n_frames = len(frames)
        
        if n_frames < window_size:
            print(f"[WARN] Video too short ({n_frames} frames) for window size {window_size}")
            window_size = n_frames
        
        if window_size % 2 != 0:
            window_size += 1
            
        results = []
        boundary_votes = defaultdict(int)
        
        print(f"[INFO] Processing {n_frames} frames with window_size={window_size}, stride={stride}")

        # Prepare debug directory
        if debug:
            tag = debug_tag or os.path.splitext(os.path.basename(video_path))[0]
            base_debug_dir = debug_dir or os.path.join(output_dir, "debug_windows")
            debug_base = os.path.join(base_debug_dir, tag)
            os.makedirs(debug_base, exist_ok=True)

        total_candidates = max(1, n_frames - window_size + 1)
        total_windows = ((total_candidates - 1) // stride) + 1
        pbar = tqdm(total=total_windows, desc="Analyzing windows", unit="win")

        for start_idx in range(0, total_candidates, stride):
            end_idx = min(start_idx + window_size, n_frames)
            window_frames = frames[start_idx:end_idx]

            while len(window_frames) < window_size:
                window_frames.append(window_frames[-1])

            right_local_idx = window_size // 2
            left_local_idx = right_local_idx - 1 if right_local_idx > 0 else 0
            global_left_idx = start_idx + left_local_idx
            global_right_idx = start_idx + right_local_idx

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

            if result.get("boundary", False):
                boundary_votes[(global_left_idx, global_right_idx)] += 1

            # Debug dump per window
            if debug:
                win_dir = os.path.join(debug_base, f"win_{start_idx:06d}")
                os.makedirs(win_dir, exist_ok=True)
                # Save JSON result with metadata
                debug_json = {
                    "window_start": int(start_idx),
                    "global_left": int(global_left_idx),
                    "global_right": int(global_right_idx),
                    "left_local_idx": int(left_local_idx),
                    "right_local_idx": int(right_local_idx),
                    "result": result
                }
                with open(os.path.join(win_dir, "result.json"), "w") as f:
                    json.dump(debug_json, f, indent=2)
                # Save frames if enabled
                if debug_save_images:
                    for i, fimg in enumerate(window_frames):
                        out_path = os.path.join(win_dir, f"frame_{i:03d}.jpg")
                        try:
                            fimg.save(out_path, format="JPEG", quality=85)
                        except Exception:
                            pass
            
            pbar.update(1)
        pbar.close()
        
        boundary_pairs = [pair for pair, votes in boundary_votes.items()
                          if votes >= vote_threshold]
        boundary_pairs.sort()

        merged_boundaries = []
        merge_distance_frames = int(merge_distance_sec * sample_fps)

        for left_idx, right_idx in boundary_pairs:
            center_idx = (left_idx + right_idx) / 2.0
            if not merged_boundaries:
                merged_boundaries.append(center_idx)
            elif center_idx - merged_boundaries[-1] <= merge_distance_frames:
                merged_boundaries[-1] = (merged_boundaries[-1] + center_idx) / 2.0
            else:
                merged_boundaries.append(center_idx)

        boundary_times = []
        for center in merged_boundaries:
            int_center = int(round(center))
            if 0 <= int_center < len(frame_indices):
                time_sec = frame_indices[int_center] / video_fps
                boundary_times.append(time_sec)

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
            "boundary_centers": merged_boundaries,
            "boundary_frame_pairs": boundary_pairs,
            "boundary_times": boundary_times,
            "raw_votes": {
                f"{left}-{right}": votes
                for (left, right), votes in boundary_votes.items()
            }
        }

        output_file = os.path.join(output_dir, "boundaries.json")
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        txt_file = os.path.join(output_dir, "boundary_times.txt")
        with open(txt_file, "w") as f:
            f.write("# Event boundaries (seconds)\n")
            for center, time in zip(merged_boundaries, boundary_times):
                f.write(f"{center:.2f}\t{time:.3f}\n")
        
        print(f"[INFO] Found {len(boundary_times)} boundaries")
        print(f"[INFO] Results saved to {output_dir}/")
        
        return output_data


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL and Qwen2.5-VL temporal segmentation")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct",
                       help="Model name (e.g., Qwen/Qwen2.5-VL-7B-Instruct or Qwen/Qwen2.5-VL-32B-Instruct)")
    parser.add_argument("--prompt-type", type=str, 
                       choices=["small", "narrative", "context", "semantic", "gebd_aligned", "aggressive"], 
                       default="gebd_aligned",
                       help="Prompt style (use 'gebd_aligned' for Kinetics-GEBD benchmark)")
    parser.add_argument("--sample-fps", type=float, default=2.0,
                       help="Frame sampling rate")
    parser.add_argument("--start-time", type=float, default=None,
                       help="Start time in seconds (inclusive)")
    parser.add_argument("--end-time", type=float, default=None,
                       help="End time in seconds (inclusive)")
    parser.add_argument("--window-size", type=int, default=7,
                       help="Window size (should be odd)")
    parser.add_argument("--stride", type=int, default=2,
                       help="Stride for sliding window")
    parser.add_argument("--vote-threshold", type=int, default=1,
                       help="Minimum votes to confirm boundary")
    parser.add_argument("--merge-distance", type=float, default=0.5,
                       help="Merge boundaries within this many seconds")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    segmenter = QwenTemporalSegmenterFixed(model_name=args.model, prompt_type=args.prompt_type)
    
    results = segmenter.sliding_window_detection(
        video_path=args.video,
        sample_fps=args.sample_fps,
        window_size=args.window_size,
        stride=args.stride,
        vote_threshold=args.vote_threshold,
        merge_distance_sec=args.merge_distance,
        output_dir=args.output_dir,
        start_time=args.start_time,
        end_time=args.end_time
    )
    
    print("\nDetected boundaries (seconds):")
    for t in results["boundary_times"]:
        print(f"  {t:.2f}s")


if __name__ == "__main__":
    main()
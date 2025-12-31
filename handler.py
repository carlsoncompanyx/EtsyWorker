# file: handler.py
"""
RunPod Serverless Handler - Unified Endpoint: create + production
- create: generate + aesthetic score
- production: generate + aesthetic score + OPTIONAL upscale via external RunPod endpoint

Why external upscaler:
- avoids heavy deps (realesrgan/basicsr/opencv) in this container
- avoids extra downloads and cold-start pain
"""

from __future__ import annotations

import base64
import io
import os
import sys
import threading
import traceback
from typing import Any, Dict, Optional, Tuple

import runpod
import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EDMDPMSolverMultistepScheduler, EDMEulerScheduler
from transformers import AutoModel, AutoProcessor


# -----------------------------
# Runtime config
# -----------------------------
PLAYGROUND_REPO = "playgroundai/playground-v2.5-1024px-aesthetic"
AESTHETICS_REPO = "discus0434/aesthetic-predictor-v2-5"

# If you use RunPod "Cached Models", set cache root to huggingface-cache (recommended)
# Cached models path is typically: /runpod-volume/huggingface-cache/...
HF_ROOT = os.environ.get("HF_HOME", "/runpod-volume/huggingface-cache")
os.environ["HF_HOME"] = HF_ROOT
os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_ROOT, "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# If set to "1", we will not download models at runtime (fails fast if cache missing).
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"

# Optional external upscaler endpoint (RunPod prebuilt or your own)
UPSCALE_ENDPOINT_ID = os.environ.get("UPSCALE_ENDPOINT_ID", "").strip()
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()


def _log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


# -----------------------------
# Global models (persist per worker)
# -----------------------------
pipe: Optional[DiffusionPipeline] = None
aesthetic_model: Optional[torch.nn.Module] = None
aesthetic_processor: Optional[Any] = None

# Prebuilt schedulers to avoid re-instantiation each request
scheduler_lock = threading.Lock()
scheduler_dpm: Optional[EDMDPMSolverMultistepScheduler] = None
scheduler_euler: Optional[EDMEulerScheduler] = None


def load_models() -> None:
    """Cold-start initialization (idempotent)."""
    global pipe, aesthetic_model, aesthetic_processor, scheduler_dpm, scheduler_euler

    if pipe is None:
        _log("=" * 60)
        _log("Loading Playground v2.5 pipeline...")
        _log("=" * 60)

        pipe = DiffusionPipeline.from_pretrained(
            PLAYGROUND_REPO,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=LOCAL_FILES_ONLY,
        ).to("cuda")

        scheduler_dpm = EDMDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        scheduler_euler = EDMEulerScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler = scheduler_dpm

        _log("✓ Playground v2.5 loaded")

    if aesthetic_model is None or aesthetic_processor is None:
        _log("=" * 60)
        _log("Loading aesthetic predictor v2.5...")
        _log("=" * 60)

        aesthetic_model = AutoModel.from_pretrained(
            AESTHETICS_REPO,
            trust_remote_code=True,
            local_files_only=LOCAL_FILES_ONLY,
        ).to("cuda")
        aesthetic_processor = AutoProcessor.from_pretrained(
            AESTHETICS_REPO,
            trust_remote_code=True,
            local_files_only=LOCAL_FILES_ONLY,
        )

        _log("✓ Aesthetic predictor v2.5 loaded")

    _log(f"✓ Models ready (LOCAL_FILES_ONLY={LOCAL_FILES_ONLY})")
    _log(f"  HF_HOME={os.environ.get('HF_HOME')}")
    _log(f"  HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
    _log(f"  TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE')}")


def _set_scheduler(scheduler_type: str) -> float:
    """Thread-safe scheduler selection; returns recommended default guidance scale."""
    assert pipe is not None
    assert scheduler_dpm is not None and scheduler_euler is not None

    with scheduler_lock:
        if scheduler_type == "euler":
            pipe.scheduler = scheduler_euler
            return 5.0
        pipe.scheduler = scheduler_dpm
        return 3.0


def calculate_aesthetic_score(image: Image.Image) -> float:
    assert aesthetic_model is not None and aesthetic_processor is not None
    inputs = aesthetic_processor(images=image, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        outputs = aesthetic_model(**inputs)

    # Robust handling for different remote_code outputs
    if hasattr(outputs, "logits") and outputs.logits is not None:
        score = float(outputs.logits.reshape(-1)[0].item())
    elif isinstance(outputs, (tuple, list)) and outputs:
        score = float(torch.as_tensor(outputs[0]).reshape(-1)[0].item())
    else:
        raise RuntimeError("Aesthetic model returned unexpected output format")

    return round(score, 2)


def _img_to_b64_jpeg(img: Image.Image, quality: int = 95) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _b64_to_pil(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _generate(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    guidance_scale: float,
    scheduler_type: str,
) -> Tuple[Image.Image, int, float]:
    assert pipe is not None

    scheduler_type = (scheduler_type or "dpm").lower()
    default_gs = _set_scheduler(scheduler_type)

    # Accept either cfg_scale or guidance_scale convention
    if guidance_scale in (3.0, 5.0):
        guidance_scale = default_gs

    if seed == -1:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())

    gen = torch.Generator(device="cuda").manual_seed(seed)

    _log(f"[GEN] {scheduler_type=} {width}x{height} seed={seed} steps={steps} gs={guidance_scale}")

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]

    score = calculate_aesthetic_score(image)
    return image, seed, score


def _call_upscaler_endpoint(image_b64: str, factor: float = 4.0) -> Dict[str, Any]:
    """
    Calls a separate RunPod endpoint to upscale the image.

    Contract expectation (you can match your chosen upscaler endpoint):
    input: { "image": "<base64>", "scale": 4 }
    output: { "image": "<base64>", "width": ..., "height": ..., "upscaler": "..." }
    """
    if not UPSCALE_ENDPOINT_ID:
        raise RuntimeError("UPSCALE_ENDPOINT_ID is not set")
    if not RUNPOD_API_KEY:
        raise RuntimeError("RUNPOD_API_KEY is not set (needed to call upscaler endpoint)")

    url = f"https://api.runpod.ai/v2/{UPSCALE_ENDPOINT_ID}/runsync"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    payload = {"input": {"image": image_b64, "scale": factor}}

    r = requests.post(url, headers=headers, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()

    if data.get("status") == "FAILED":
        raise RuntimeError(f"Upscaler FAILED: {data.get('error') or data}")

    out = data.get("output") or {}
    if "image" not in out:
        raise RuntimeError(f"Upscaler response missing output.image: {data}")

    return out


def route_create(job_input: Dict[str, Any]) -> Dict[str, Any]:
    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "")

    width = int(job_input.get("width", 1152))
    height = int(job_input.get("height", 768))
    seed = int(job_input.get("seed", -1))
    steps = int(job_input.get("steps", 25))

    # accept cfg_scale alias used in your README
    guidance_scale = float(job_input.get("guidance_scale", job_input.get("cfg_scale", 3.0)))
    scheduler_type = job_input.get("scheduler", "dpm")
    filename = job_input.get("filename", (prompt[:50] or "create"))

    image, used_seed, score = _generate(
        prompt, negative_prompt, width, height, seed, steps, guidance_scale, scheduler_type
    )

    return {
        "route": "create",
        "image": _img_to_b64_jpeg(image),
        "aesthetic_score": score,
        "metadata": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "seed": used_seed,
            "steps": steps,
            "cfg_scale": guidance_scale,
            "scheduler": scheduler_type,
            "filename": filename,
            "model": "playground-v2.5-1024px-aesthetic",
        },
    }


def route_production(job_input: Dict[str, Any]) -> Dict[str, Any]:
    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "")

    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 680))
    seed = int(job_input.get("seed", 0))
    steps = int(job_input.get("steps", 50))

    guidance_scale = float(job_input.get("guidance_scale", job_input.get("cfg_scale", 3.0)))
    scheduler_type = job_input.get("scheduler", "dpm")

    # Upscale controls (Flow B)
    upscale = bool(job_input.get("upscale", True))
    upscale_factor = float(job_input.get("upscale_factor", 4.0))

    base_img, used_seed, score = _generate(
        prompt, negative_prompt, width, height, seed, steps, guidance_scale, scheduler_type
    )

    base_b64 = _img_to_b64_jpeg(base_img)

    final_b64 = base_b64
    final_w, final_h = width, height
    upscaler_name = None

    if upscale:
        if not UPSCALE_ENDPOINT_ID:
            _log("[PRODUCTION] upscale requested but UPSCALE_ENDPOINT_ID not set; returning base image.")
        else:
            _log(f"[PRODUCTION] Calling upscaler endpoint: x{upscale_factor}")
            up_out = _call_upscaler_endpoint(base_b64, factor=upscale_factor)
            final_b64 = up_out["image"]
            try:
                pil = _b64_to_pil(final_b64)
                final_w, final_h = pil.size
            except Exception:
                pass
            upscaler_name = up_out.get("upscaler")

    return {
        "route": "production",
        "image": final_b64,
        "aesthetic_score": score,
        "metadata": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "base_width": width,
            "base_height": height,
            "final_width": final_w,
            "final_height": final_h,
            "seed": used_seed,
            "steps": steps,
            "cfg_scale": guidance_scale,
            "scheduler": scheduler_type,
            "upscale": upscale,
            "upscale_factor": upscale_factor,
            "upscaler": upscaler_name,
            "model": "playground-v2.5-1024px-aesthetic",
        },
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if job.get("input") == "health":
            return {
                "status": "healthy",
                "models_loaded": pipe is not None and aesthetic_model is not None,
                "gpu_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                "upscale_configured": bool(UPSCALE_ENDPOINT_ID and RUNPOD_API_KEY),
            }

        job_input = job["input"]
        route = str(job_input.get("route", "")).lower()

        _log(f"\nReceived request for route: {route}")
        load_models()

        if route == "create":
            return route_create(job_input)
        if route == "production":
            return route_production(job_input)

        return {"error": f"Invalid route: '{route}'. Must be 'create' or 'production'."}

    except Exception as e:
        _log(f"\n✗ Error in handler: {e}")
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    _log("=" * 60)
    _log("Starting RunPod serverless handler...")
    _log("=" * 60)
    runpod.serverless.start({"handler": handler})

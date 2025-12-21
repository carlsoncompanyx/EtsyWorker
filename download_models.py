"""
Pre-download all required models during Docker build
This speeds up cold starts significantly
"""

import torch
from diffusers import DiffusionPipeline
from transformers import CLIPModel, CLIPProcessor

print("=" * 60)
print("Downloading Playground v2.5 model...")
print("=" * 60)

pipe = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2.5-1024px-aesthetic",
    torch_dtype=torch.float16,
    variant="fp16"
)
print("✓ Playground v2.5 downloaded successfully")

print("\n" + "=" * 60)
print("Downloading CLIP models for aesthetic scoring...")
print("=" * 60)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print("✓ CLIP models downloaded successfully")

print("\n" + "=" * 60)
print("All models cached successfully!")
print("=" * 60)

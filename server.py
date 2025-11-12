import runpod

from handler import realesrgan_handler

runpod.serverless.start({"handler": realesrgan_handler})

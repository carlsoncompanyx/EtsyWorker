# RunPod Unified Endpoint Deployment Guide

## Overview

This guide covers deploying a **single RunPod serverless endpoint** with two routes:
- **`/create`** - Fast image generation with aesthetic scoring (Flow A)
- **`/production`** - High-quality generation with 4x upscaling (Flow B)

Both routes share the same models and infrastructure, making it more efficient and cost-effective.

## Prerequisites

1. RunPod account with API key
2. Docker installed locally
3. Docker Hub or container registry account

## File Structure

```
project/
├── Dockerfile
├── requirements.txt
├── handler.py          # Unified handler with routing
└── README.md
```

## Quick Deployment

### Step 1: Build Docker Image

```bash
# Build the unified image
docker build -t your-dockerhub-username/runpod-playground-unified:latest .

# Push to registry
docker push your-dockerhub-username/runpod-playground-unified:latest
```

### Step 2: Create RunPod Endpoint

1. Log into [RunPod](https://www.runpod.io/)
2. Navigate to **Serverless** → **Templates**
3. Create new template:
   - **Name**: `Playground-Unified-Endpoint`
   - **Container Image**: `your-dockerhub-username/runpod-playground-unified:latest`
   - **Container Start Command**: `python -u handler.py`
   - **Min Workers**: 0 (scale to zero when idle)
   - **Max Workers**: 3
   - **GPU**: RTX 3090 or better (RTX 4090 recommended for production route)
   - **Container Disk**: 30GB
   - **Execution Timeout**: 300 seconds

4. Create endpoint from this template
5. Copy your endpoint ID (e.g., `abc123def456`)

## API Usage

### Base Endpoint
```
https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}
```

### Route 1: `/create` - Fast Generation

**When to use**: Quick iterations, prompt testing, previews, concept exploration

**Endpoint**: `/runsync` (synchronous, waits for completion)

**Request:**
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "route": "create",
      "prompt": "A serene mountain landscape at sunset, highly detailed",
      "negative_prompt": "blurry, low quality, distorted",
      "width": 1152,
      "height": 768,
      "seed": -1,
      "steps": 25,
      "cfg_scale": 7.0,
      "filename": "mountain_sunset"
    }
  }'
```

**Response:**
```json
{
  "delayTime": 2043,
  "executionTime": 12456,
  "id": "sync-job-id",
  "output": {
    "route": "create",
    "image": "base64_encoded_jpeg_string...",
    "aesthetic_score": 7.45,
    "metadata": {
      "prompt": "A serene mountain landscape at sunset...",
      "negative_prompt": "blurry, low quality...",
      "width": 1152,
      "height": 768,
      "seed": 1234567890,
      "steps": 25,
      "cfg_scale": 7.0,
      "filename": "mountain_sunset",
      "model": "playground-v2.5-1024px-aesthetic"
    }
  },
  "status": "COMPLETED"
}
```

**Estimated Time**: 10-30 seconds

---

### Route 2: `/production` - High-Quality with Upscaling

**When to use**: Final deliverables, print quality, client work, portfolio pieces

**Endpoint**: `/run` (async with polling) or `/runsync` (if under 60s timeout)

**Request:**
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "route": "production",
      "prompt": "Professional product photography of a luxury watch",
      "negative_prompt": "amateur, blurry, low quality",
      "width": 1024,
      "height": 680,
      "seed": 12345,
      "steps": 50,
      "cfg_scale": 7.0,
      "upscale_factor": 4.0,
      "use_tiled_upscale": true,
      "tile_size": 1024
    }
  }'
```

**Initial Response:**
```json
{
  "id": "abc-123-def-456",
  "status": "IN_QUEUE"
}
```

**Status Check:**
```bash
curl -X GET https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/abc-123-def-456 \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"
```

**Final Response:**
```json
{
  "delayTime": 3211,
  "executionTime": 87643,
  "id": "abc-123-def-456",
  "output": {
    "route": "production",
    "image": "base64_encoded_jpeg_string...",
    "metadata": {
      "prompt": "Professional product photography...",
      "base_width": 1024,
      "base_height": 680,
      "final_width": 4096,
      "final_height": 2720,
      "seed": 12345,
      "steps": 50,
      "cfg_scale": 7.0,
      "upscale_factor": 4.0,
      "model": "playground-v2.5-1024px-aesthetic",
      "upscaler": "Real-ESRGAN 4x"
    }
  },
  "status": "COMPLETED"
}
```

**Estimated Time**: 60-120 seconds

---

## n8n Integration

### Setup 1: `/create` Route (Synchronous)

**HTTP Request Node:**
- **Method**: POST
- **URL**: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync`
- **Authentication**: Header Auth
  - Name: `Authorization`
  - Value: `Bearer YOUR_RUNPOD_API_KEY`
- **Body Type**: JSON
- **Body**:
```json
{
  "input": {
    "route": "create",
    "prompt": "={{$json.prompt}}",
    "negative_prompt": "={{$json.negative_prompt}}",
    "width": 1152,
    "height": 768,
    "seed": -1,
    "steps": 25,
    "cfg_scale": 7.0
  }
}
```

**Function Node (Decode Image):**
```javascript
// Extract response
const response = items[0].json;

// Decode base64 image
const imageBuffer = Buffer.from(response.output.image, 'base64');

return [{
  json: {
    aesthetic_score: response.output.aesthetic_score,
    metadata: response.output.metadata
  },
  binary: {
    image: {
      data: imageBuffer,
      mimeType: 'image/jpeg',
      fileName: `${response.output.metadata.filename || 'generated'}.jpg`
    }
  }
}];
```

---

### Setup 2: `/production` Route (Async with Polling)

**1. Start Job - HTTP Request Node:**
```json
{
  "method": "POST",
  "url": "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run",
  "body": {
    "input": {
      "route": "production",
      "prompt": "={{$json.prompt}}",
      "width": 1024,
      "height": 680,
      "steps": 50
    }
  }
}
```

**2. Wait Node:** 5 seconds

**3. Poll Status - HTTP Request Node:**
```json
{
  "method": "GET",
  "url": "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/={{$json.id}}"
}
```

**4. Loop Until Complete - Function Node:**
```javascript
const status = items[0].json.status;

if (status === 'COMPLETED') {
  // Decode and return
  const imageBuffer = Buffer.from(items[0].json.output.image, 'base64');
  return [{
    json: items[0].json.output.metadata,
    binary: {
      image: {
        data: imageBuffer,
        mimeType: 'image/jpeg',
        fileName: `production_${Date.now()}.jpg`
      }
    }
  }];
} else if (status === 'FAILED') {
  throw new Error('Job failed: ' + items[0].json.error);
} else {
  // Continue polling
  return items;
}
```

**5. Use n8n's Loop Node** to repeat steps 2-4 until status is COMPLETED

---

## Parameters Reference

### Common Parameters (Both Routes)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `route` | string | **required** | Either "create" or "production" |
| `prompt` | string | **required** | Positive prompt describing desired image |
| `negative_prompt` | string | "" | Elements to avoid in generation |
| `width` | int | 1152 (create)<br>1024 (production) | Image width in pixels |
| `height` | int | 768 (create)<br>680 (production) | Image height in pixels |
| `seed` | int | -1 | Random seed (-1 = random) |
| `steps` | int | 25 (create)<br>50 (production) | Number of inference steps |
| `cfg_scale` | float | 7.0 | Classifier-free guidance scale |

### Production-Only Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `upscale_factor` | float | 4.0 | Upscaling multiplier |
| `use_tiled_upscale` | bool | true | Use tiled processing (recommended) |
| `tile_size` | int | 1024 | Tile size for tiled upscaling |

---

## Route Comparison

| Feature | `/create` | `/production` |
|---------|-----------|---------------|
| **Speed** | ⚡ Fast (10-30s) | 🐢 Slow (60-120s) |
| **Resolution** | 1152x768 | 4096x2720 (4x) |
| **Quality** | Good | Excellent |
| **Upscaling** | ❌ None | ✅ Real-ESRGAN 4x |
| **Aesthetic Score** | ✅ Included | ❌ Not included |
| **Use Case** | Iterations, testing | Final deliverables |
| **Recommended n8n** | /runsync | /run + polling |
| **Cost per Image** | ~$0.10-0.20 | ~$0.30-0.50 |

---

## Cost Optimization

1. **Set Min Workers to 0**: Only pay when generating
2. **Use `/create` for testing**: 50-70% cheaper than production
3. **Batch requests**: Keep workers warm across multiple requests
4. **Right-size parameters**:
   - Lower steps for testing (15-20 for create, 30-40 for production)
   - Smaller dimensions = faster + cheaper
5. **Use spot instances**: 50-70% discount (enable in RunPod settings)

---

## Troubleshooting

### Common Issues

**Problem**: "Invalid route" error
- **Solution**: Ensure `"route": "create"` or `"route": "production"` is in your input

**Problem**: Timeout on production route
- **Solution**: Use `/run` endpoint with polling instead of `/runsync`

**Problem**: Out of memory during upscaling
- **Solution**: Enable `"use_tiled_upscale": true` or use larger GPU

**Problem**: Low aesthetic scores
- **Solution**: Improve prompts, increase steps, adjust cfg_scale

**Problem**: Cold start takes too long (>60s)
- **Solution**: Models are pre-downloaded. Check RunPod logs for actual issue.

### Monitoring

View logs in RunPod Dashboard:
- **Logs** tab: See handler output with route indicators `[CREATE]` or `[PRODUCTION]`
- **Metrics**: Monitor GPU utilization and execution times
- **Workers**: Check active instances and their status

---

## Advanced Usage

### Custom Image Sizes

Both routes support custom dimensions (multiples of 8):

```json
{
  "route": "create",
  "width": 1280,
  "height": 720,
  "prompt": "..."
}
```

### Reproducible Generations

Use fixed seed for identical results:

```json
{
  "route": "production",
  "seed": 42,
  "prompt": "..."
}
```

### Maximum Quality Production

```json
{
  "route": "production",
  "steps": 75,
  "cfg_scale": 8.0,
  "use_tiled_upscale": true,
  "tile_size": 512
}
```

---

## Next Steps

1. ✅ Build and deploy the Docker image
2. ✅ Create RunPod endpoint
3. ✅ Test both routes with curl
4. ✅ Set up n8n workflows
5. ✅ Monitor costs and optimize

Need help? Check RunPod documentation or the handler logs for detailed error messages.

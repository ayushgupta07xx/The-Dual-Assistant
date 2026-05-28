"""Public OSS deployment — Modal (GPU, larger model, OpenAI-compatible).

This deploys Qwen2.5-7B-Instruct behind a vLLM OpenAI-compatible server on a
single GPU. It gives the project a higher-quality open-source assistant than
the 0.5B CPU Space, while keeping cost controllable: Modal scales the container
to zero when idle, so you only pay for GPU-seconds actually used.

The main project talks to this endpoint with no code changes:

    export OSS_BACKEND=endpoint
    export OSS_ENDPOINT_URL="https://<your-modal-app>.modal.run/v1"
    export OSS_ENDPOINT_KEY="$MODAL_TOKEN"   # the API_KEY set below
    export OSS_MODEL="Qwen/Qwen2.5-7B-Instruct"
    export OSS_GPU_USD_PER_SEC=0.000222      # ~ A10G $0.80/hr -> per-second

Deploy:
    pip install modal
    modal token new
    modal deploy deploy/modal/modal_app.py

Cost reference (Modal on-demand, approximate, USD/hour):
    A10G  ~$0.80/hr  -> $0.000222/sec   (good default for 7B, ~30-60 tok/s)
    L40S  ~$1.95/hr  -> $0.000542/sec
    A100  ~$3.70/hr  -> $0.001028/sec
Scale-to-zero means idle cost is $0; you pay for the ~seconds each request runs.
"""
from __future__ import annotations

import os

import modal

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
GPU_TYPE = "A10G"
# Set a token to require auth on the public endpoint. Override in Modal secrets.
API_KEY = os.environ.get("OSS_ENDPOINT_KEY", "change-me-please")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.3",
        "huggingface_hub[hf_transfer]==0.26.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("oss-assistant-vllm")

# Cache model weights across cold starts so we download once.
volume = modal.Volume.from_name("qwen-weights", create_if_missing=True)
VOL_PATH = "/models"


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOL_PATH: volume},
    scaledown_window=300,  # keep warm 5 min after last request, then scale to zero
    timeout=600,
    allow_concurrent_inputs=20,
)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    """Launch a vLLM OpenAI-compatible server.

    Exposes /v1/chat/completions and /v1/completions so any OpenAI client
    (including this project's `endpoint` backend) can call it directly.
    """
    import subprocess

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_ID,
        "--download-dir",
        VOL_PATH,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--api-key",
        API_KEY,
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.90",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)


# Optional: a tiny local entrypoint to smoke-test the deployed endpoint.
@app.local_entrypoint()
def main():
    print("Deploy with:  modal deploy deploy/modal/modal_app.py")
    print("Then point the project at the printed *.modal.run URL via OSS_ENDPOINT_URL.")

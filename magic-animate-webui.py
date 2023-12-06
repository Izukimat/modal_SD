from pathlib import Path
import modal
import shutil
import subprocess
import os
import sys
import shlex

# Define Modal variables
stub = modal.Stub("magic-animate")
volume_main = modal.NetworkFileSystem.new().persist("magic-animate-main")

# Model checkpoints and configurations
model_checkpoints = [
    # {
    #     "repo_id": "zcxu-eric/MagicAnimate",
    #     "file_paths": [
    #         "MagicAnimate/appearance_encoder/diffusion_pytorch_model.safetensors",
    #         "MagicAnimate/appearance_encoder/config.json",
    #         "MagicAnimate/densepose_controlnet/diffusion_pytorch_model.safetensors",
    #         "MagicAnimate/densepose_controlnet/config.json",
    #         "MagicAnimate/temporal_attention/temporal_attention.ckpt",
    #     ],
    # },
    {
        "repo_id": "stabilityai/sd-vae-ft-mse",
        "file_paths": [
            "diffusion_pytorch_model.bin",         # Model file
            "diffusion_pytorch_model.safetensors", # Model file in SafeTensors format
            "config.json",                         # Config file
            # Add other necessary file paths if there are any
        ],
    },
    {
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "file_paths": [
            "v1-5-pruned.ckpt",  # Model checkpoint file
            "v1-5-pruned.safetensors",  # Model checkpoint file in SafeTensors format
            "v1-inference.yaml",  # Configuration file for inference
            # Include additional files as required by the application
        ],
    }

]

# Define various paths
project_dir = "/content/magic-animate"
models_dir = os.path.join(project_dir, "pretrained_models")
magicanimate_subdir = os.path.join(models_dir, "MagicAnimate")

@stub.function(
    image=modal.Image.from_registry("python:3.11")
    .apt_install("git", "ffmpeg", "libgl1-mesa-dev", "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "git-lfs")
    .pip_install(
                "huggingface_hub", 
                "colorama",
                "absl-py==1.4.0",
                "accelerate==0.22.0",
                "aiofiles==23.2.1",
                "aiohttp==3.8.5",
                "aiosignal==1.3.1",
                "altair==5.0.1",
                "annotated-types==0.5.0",
                "antlr4-python3-runtime==4.9.3",
                "anyio==3.7.1",
                "async-timeout==4.0.3",
                "attrs==23.1.0",
                "cachetools==5.3.1",
                "certifi==2023.7.22",
                "charset-normalizer==3.2.0",
                "click==8.1.7",
                "cmake==3.27.2",
                "contourpy==1.1.0",
                "cycler==0.11.0",
                "datasets==2.14.4",
                "dill==0.3.7",
                "einops==0.6.1",
                "exceptiongroup==1.1.3",
                "fastapi==0.103.0",
                "ffmpy==0.3.1",
                "filelock==3.12.2",
                "fonttools==4.42.1",
                "frozenlist==1.4.0",
                "fsspec==2023.6.0",
                "google-auth==2.22.0",
                "google-auth-oauthlib==1.0.0",
                "gradio==3.41.2",
                "gradio-client==0.5.0",
                "grpcio==1.57.0",
                "h11==0.14.0",
                "httpcore==0.17.3",
                "httpx==0.24.1",
                "huggingface-hub==0.16.4",
                "idna==3.4",
                "importlib-metadata==6.8.0",
                "importlib-resources==6.0.1",
                "jinja2==3.1.2",
                "joblib==1.3.2",
                "jsonschema==4.19.0",
                "jsonschema-specifications==2023.7.1",
                "kiwisolver==1.4.5",
                "lightning-utilities==0.9.0",
                "lit==16.0.6",
                "markdown==3.4.4",
                "markupsafe==2.1.3",
                "matplotlib==3.7.2",
                "mpmath==1.3.0",
                "multidict==6.0.4",
                "multiprocess==0.70.15",
                "networkx==3.1",
                "numpy==1.24.4",
                "nvidia-cublas-cu11==11.10.3.66",
                "nvidia-cuda-cupti-cu11==11.7.101",
                "nvidia-cuda-nvrtc-cu11==11.7.99",
                "nvidia-cuda-runtime-cu11==11.7.99",
                "nvidia-cudnn-cu11==8.5.0.96",
                "nvidia-cufft-cu11==10.9.0.58",
                "nvidia-curand-cu11==10.2.10.91",
                "nvidia-cusolver-cu11==11.4.0.1",
                "nvidia-cusparse-cu11==11.7.4.91",
                "nvidia-nccl-cu11==2.14.3",
                "nvidia-nvtx-cu11==11.7.91",
                "oauthlib==3.2.2",
                "omegaconf==2.3.0",
                "opencv-python==4.8.0.76",
                "orjson==3.9.5",
                "pandas==2.0.3",
                "pillow==9.5.0",
                "pkgutil-resolve-name==1.3.10",
                "protobuf==4.24.2",
                "psutil==5.9.5",
                "pyarrow==13.0.0",
                "pyasn1==0.5.0",
                "pyasn1-modules==0.3.0",
                "pydantic==2.3.0",
                "pydantic-core==2.6.3",
                "pydub==0.25.1",
                "pyparsing==3.0.9",
                "python-multipart==0.0.6",
                "pytorch-lightning==2.0.7",
                "pytz==2023.3",
                "pyyaml==6.0.1",
                "referencing==0.30.2",
                "regex==2023.8.8",
                "requests==2.31.0",
                "requests-oauthlib==1.3.1",
                "rpds-py==0.9.2",
                "rsa==4.9",
                "safetensors==0.3.3",
                "semantic-version==2.10.0",
                "sniffio==1.3.0",
                "starlette==0.27.0",
                "sympy==1.12",
                "tensorboard==2.14.0",
                "tensorboard-data-server==0.7.1",
                "tokenizers==0.13.3",
                "toolz==0.12.0",
                "torchmetrics==1.1.0",
                "tqdm==4.66.1",
                "transformers==4.32.0",
                "triton==2.0.0",
                "tzdata==2023.3",
                "urllib3==1.26.16",
                "uvicorn==0.23.2",
                "websockets==11.0.3",
                "werkzeug==2.3.7",
                "xxhash==3.3.0",
                "yarl==1.9.2",
                "zipp==3.16.2",
                "decord",
                "imageio==2.9.0",
                "imageio-ffmpeg==0.4.3",
                "timm",
                "scipy",
                "scikit-image",
                "av",
                "imgaug",
                "lpips",
                "ffmpeg-python",
                "torch==2.0.1",
                "torchvision==0.15.2",
                "xformers==0.0.22",
                "diffusers==0.21.4",
                ),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    network_file_systems={project_dir: volume_main},
    gpu="a10g",
    timeout=6000,
)
async def run_magic_animate():
    print("Setting up MagicAnimate...")

    # Clone the repository if not already cloned
    webui_dir_path = Path(models_dir)
    if not webui_dir_path.exists():
        subprocess.run(f"git clone https://github.com/magic-research/magic-animate.git", shell=True)

    # Move the required directories to the target directory
    required_dirs = ["appearance_encoder", "densepose_controlnet", "temporal_attention"]
    for dir_name in required_dirs:
        src_dir = os.path.join(magicanimate_subdir, dir_name)
        dst_dir = os.path.join(magicanimate_subdir, dir_name)
        shutil.move(src_dir, dst_dir)

    print("MagicAnimate setup complete.", file=sys.stderr)
    # Function to download files from Hugging Face
    def download_hf_file(repo_id, filename):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename)

    def safe_move(src, dst):
        print(f"Moving from {src} to {dst}", file=sys.stderr)
        dest_dir = os.path.dirname(dst)
        if not os.path.exists(dest_dir):
            print(f"Creating directory: {dest_dir}", file=sys.stderr)
            os.makedirs(dest_dir, exist_ok=True)

        # Copy the file
        shutil.copy(src, dst)
        print(f"Copied file to {dst}", file=sys.stderr)
        
        # Remove the source file
        os.remove(src)
        print(f"Removed original file from {src}", file=sys.stderr)

    # Download and set up model checkpoints
    for checkpoint in model_checkpoints:
        for file_path in checkpoint["file_paths"]:
            local_path = models_dir + file_path
            if not Path(local_path).exists():
                downloaded_path = download_hf_file(checkpoint["repo_id"], file_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                print(f"Downloaded path: {downloaded_path}, Local path: {local_path}", file=sys.stderr)
                safe_move(downloaded_path, local_path)

    # Clone the checkpoint repository with Git LFS
    git_clone_cmd = "git lfs clone https://huggingface.co/zcxu-eric/MagicAnimate"
    subprocess.run(git_clone_cmd, shell=True, check=True)

    # Define the directory where the repo is cloned
    cloned_repo_dir = "MagicAnimate"  # Adjust this path if necessary

    # Define the target directory for pretrained models
    target_models_dir = os.path.join(project_dir, "pretrained_models", "MagicAnimate")

    # Create the target directory if it doesn't exist
    os.makedirs(target_models_dir, exist_ok=True)

    # Move the required files to the target directory
    # Here, you should list all the directories and files that need to be moved

    required_dirs = ["appearance_encoder", "densepose_controlnet", "temporal_attention"]
    for dir_name in required_dirs:
        src_dir = os.path.join(cloned_repo_dir, dir_name)
        dst_dir = os.path.join(target_models_dir, dir_name)
        
        safe_move(src_dir, dst_dir)

    print("MagicAnimate setup complete.")

    # Run the MagicAnimate script
    sys.path.append(project_dir)
    os.chdir(project_dir)

    # Adjust these arguments as needed for MagicAnimate
    sys.argv = shlex.split("--a --gradio-debug --share")

    # Import and start Gradio
    # Replace 'demo.gradio_animate' with the correct module and function for MagicAnimate
    import demo.gradio_animate

@stub.local_entrypoint()
def main():
    run_magic_animate.remote()


from colorama import Fore
from pathlib import Path

import time
import modal
import subprocess
import sys
import shlex
import os

'''
【モデルファイルの追加】
カレントディレクトリの下に「Stable-Diffusion」というフォルダを作成し、追加で使いたいモデルファイル(ChillOutMix等)を入れ、下記コマンドを実行。
modal volume put stable-diffusion-webui-main Stable-Diffusion models/Stable-diffusion/

【Loraファイルの追加】
カレントディレクトリの下に「lora」というフォルダを作成し、そこに使いたいLoraファイルを全て入れ、下記コマンドを実行。
modal volume put stable-diffusion-webui-main lora /models/Lora

【VAEファイルの追加】
カレントディレクトリの下に「VAE」というフォルダを作成し、そこに使いたいVAEファイルを入れ、下記コマンドを実行。
modal volume put stable-diffusion-webui-main VAE /models/VAE

【Textual Inversionファイルの追加】
カレントディレクトリの下に「embeddings」というフォルダを作成し、そこに使いたいEasyNegative、ulzzang、Pure Eros Face等のファイルを入れ、下記コマンドを実行。
modal volume put stable-diffusion-webui-main embeddings /embeddings
'''

# modal系の変数の定義
stub = modal.Stub("stable-diffusion-webui")
volume_main = modal.NetworkFileSystem.new().persist("stable-diffusion-webui-main")

# 色んなパスの定義
webui_dir = "/content/stable-diffusion-webui"
webui_model_dir = webui_dir + "/models/Stable-diffusion/"

# ダウンロードするモデルのID
model_ids = [
    {
        "repo_id": "stabilityai/stable-diffusion-2-1",
        "model_path": "v2-1_768-ema-pruned.ckpt",
    },
]


@stub.function(
    image=modal.Image.from_registry("python:3.10")
    .apt_install(
        "git", "libgl1-mesa-dev", "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "build-essential", "cmake", "protobuf-compiler"
    )
    .run_commands(
        # "pip install onnx --only-binary :all:",
        # "pip install onnxruntime-gpu>=1.16.1 --only-binary :all:",
        "pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers"
    )
    .pip_install(
        "httpx==0.24.1",
        "blendmodes==2022",
        "gradio==3.36.1",
        "transformers==4.25.1",
        "accelerate==0.12.0",
        "basicsr==1.4.2",
        "gfpgan==1.3.8",
        "numpy==1.24.2",
        "Pillow==9.4.0",
        "realesrgan==0.3.0",
        "torch",
        "torchmetrics==0.11.4",
        "omegaconf==2.2.3",
        "pytorch_lightning==1.7.6",
        "scikit-image==0.19.2",
        "fonts",
        "font-roboto",
        "timm==0.6.7",
        "piexif==1.1.3",
        "einops==0.4.1",
        "jsonmerge==1.8.0",
        "clean-fid==0.1.29",
        "resize-right==0.0.2",
        "torchdiffeq==0.2.3",
        "kornia==0.6.7",
        "lark==1.1.2",
        "inflection==0.5.1",
        "GitPython==3.1.27",
        "torchsde==0.2.5",
        "safetensors==0.3.1",
        "httpcore<=0.15",
        "tensorboard==2.9.1",
        "taming-transformers==0.0.1",
        "invisible-watermark==0.2.0",
        "clip",
        "xformers",
        "test-tube",
        "diffusers",
        "pyngrok",
        "xformers==0.0.16rc425",
        "gdown",
        "huggingface_hub",
        "colorama",
        "insightface==0.7.3",
        "onnx",
        "onnxruntime-gpu>=1.16.1",
        "opencv-python",
        "tqdm",
        "wget",
        "pybind11"
    )
    .pip_install("git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    network_file_systems={webui_dir: volume_main},
    gpu="a10g",
    timeout=6000,
)
async def run_stable_diffusion_webui():
    time_start = time.time()
    print(Fore.CYAN + "\n---------- セットアップ開始 ----------\n")
    webui_dir_path = Path(webui_model_dir)
    if not webui_dir_path.exists():
        subprocess.run(f"git clone -b v2.0 https://github.com/camenduru/stable-diffusion-webui {webui_dir}", shell=True)

    # Hugging faceからファイルをダウンロードしてくる関数
    def download_hf_file(repo_id, filename):
        from huggingface_hub import hf_hub_download

        download_dir = hf_hub_download(repo_id=repo_id, filename=filename)
        return download_dir

    for model_id in model_ids:
        print(Fore.GREEN + model_id["repo_id"] + "のセットアップを開始します...")

        if not Path(webui_model_dir + model_id["model_path"]).exists():
            # モデルのダウンロード＆コピー
            model_downloaded_dir = download_hf_file(
                model_id["repo_id"],
                model_id["model_path"],
            )
            shutil.copy(model_downloaded_dir, webui_model_dir + os.path.basename(model_id["model_path"]))


        if "config_file_path" not in model_id:
          continue

        if not Path(webui_model_dir + model_id["config_file_path"]).exists():
            # コンフィグのダウンロード＆コピー
            config_downloaded_dir = download_hf_file(
                model_id["repo_id"], model_id["config_file_path"]
            )
            shutil.copy(config_downloaded_dir, webui_model_dir + os.path.basename(model_id["config_file_path"]))


        print(Fore.GREEN + model_id["repo_id"] + "のセットアップが完了しました！")

    print(Fore.CYAN + "\n---------- セットアップ完了 ----------\n")

    # WebUIを起動
    
    sys.path.append(webui_dir)
    sys.argv += shlex.split("--skip-install --xformers")
    os.chdir(webui_dir)
    from launch import start, prepare_environment
    print("prepare_env")
    prepare_environment()
    # 最初のargumentは無視されるので注意
    sys.argv = shlex.split("--a --gradio-debug --share --enable-insecure-extension-access --xformers")
    start()
    time_end = time.time()
    print("time erapsed: ", time_end-time_start)


@stub.local_entrypoint()
def main():
    run_stable_diffusion_webui.remote()
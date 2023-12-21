## Introduction
This script sets up and runs a Stable Diffusion WebUI environment. It includes the integration of various model files, Lora files, VAE files, and Textual Inversion files. The system is designed to be run in a Python environment with specific dependencies.

## Requirements
- Python 3.10 or higher.
- Git installed on your system.
- Access to a GPU is recommended for better performance.

## Installation
1. **Clone the Repository**: Clone this repository to your local machine using Git.
2. **Install Dependencies**: Ensure that Python 3.10 or higher is installed on your system. You will also need Git for cloning the repository.

## Usage
### Adding Model Files
1. Create a folder named `Stable-Diffusion` in the current directory.
2. Place the desired model files (e.g., ChillOutMix) into this folder.
3. Run the following command:
   ```
   modal volume put stable-diffusion-webui-main Stable-Diffusion models/Stable-diffusion/
   ```

### Adding Lora Files
1. Create a folder named `lora` in the current directory.
2. Place the desired Lora files into this folder.
3. Execute the command:
   ```
   modal volume put stable-diffusion-webui-main lora /models/Lora
   ```

### Adding VAE Files
1. Create a folder named `VAE` in the current directory.
2. Insert the desired VAE files into this folder.
3. Run:
   ```
   modal volume put stable-diffusion-webui-main VAE /models/VAE
   ```

### Adding Textual Inversion Files
1. Create a folder named `embeddings` in the current directory.
2. Add files such as EasyNegative, ulzzang, Pure Eros Face, etc., into this folder.
3. Execute:
   ```
   modal volume put stable-diffusion-webui-main embeddings /embeddings
   ```

### Running the WebUI
To start the WebUI, run the `main` function in the script. This will set up the environment and launch the WebUI interface.

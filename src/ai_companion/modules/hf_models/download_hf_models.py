# download_hf_models.py
import os
import textwrap # Optional, but good if you copy-paste indented code
from huggingface_hub import snapshot_download

# Define the Python code as a multi-line string with desired internal indentation
models_to_download_str = os.environ.get('HF_MODELS_TO_DOWNLOAD', '')
hf_cache_dir = os.environ.get('HF_HOME', '/root/.cache/huggingface/hub') # Default cache dir

if not models_to_download_str:
    print("HF_MODELS_TO_DOWNLOAD environment variable not set. Skipping model download.")
    exit()

models = [m.strip() for m in models_to_download_str.split(',') if m.strip()]

for model_name in models:
    print(f'Downloading {model_name}...')
    snapshot_download(repo_id=model_name, cache_dir=hf_cache_dir)
print('All models downloaded.')
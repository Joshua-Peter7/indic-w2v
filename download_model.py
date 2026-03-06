"""Download the model to HF cache with XetHub disabled (no progress hang)."""
import os, sys

# Must set BEFORE importing huggingface_hub
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Load token
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
with open(env_path) as fh:
    for ln in fh:
        ln = ln.strip()
        if not ln or ln.startswith('#') or '=' not in ln:
            continue
        k, _, v = ln.partition('=')
        if k.strip() and v.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip()

tok = os.environ.get('HF_TOKEN', '')
model_id = "ai4bharat/indicwav2vec_v1_bengali"

print(f"Downloading: {model_id}")
print(f"Token      : {tok[:10]}...")
print(f"XET disabled: {os.environ.get('HF_HUB_DISABLE_XET')}")
print()

from huggingface_hub import snapshot_download

try:
    path = snapshot_download(
        repo_id=model_id,
        token=tok,
        ignore_patterns=["*.msgpack", "flax_model*"],  # skip non-pytorch weights
    )
    print(f"\nDownloaded to: {path}")
    import os
    for f in os.listdir(path):
        size = os.path.getsize(os.path.join(path, f))
        print(f"  {f:<40} {size/1e6:.1f} MB")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

"""Check which HF repos actually have model weights."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from huggingface_hub import HfApi, list_repo_files
api = HfApi()

candidates = [
    'ai4bharat/indicwav2vec_v1_bengali',
    'ai4bharat/indicwav2vec_v1_gujarati',
    'ai4bharat/indicwav2vec-hindi',
    'ai4bharat/indicwav2vec-odia',
    'ai4bharat/indicwav2vec_v1_tamil',
    'ai4bharat/indicwav2vec_v1_hindi',
]

for mid in candidates:
    try:
        files = list(list_repo_files(mid, token=tok))
        has_weights = any(f in files for f in ['pytorch_model.bin', 'model.safetensors'])
        print(f'{"[HAS WEIGHTS]" if has_weights else "[NO WEIGHTS] "} {mid}')
        if has_weights:
            wfiles = [f for f in files if 'pytorch_model' in f or 'safetensor' in f or 'config' in f]
            print(f'  Files: {wfiles[:5]}')
    except Exception as e:
        print(f'[ERROR] {mid}: {type(e).__name__}: {str(e)[:80]}')

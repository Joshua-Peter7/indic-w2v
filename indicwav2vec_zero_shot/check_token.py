"""Diagnose HuggingFace token and model access."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load token from .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
token = None
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith('HF_TOKEN='):
            token = line.split('=', 1)[1].strip()

if not token or token.startswith('hf_PASTE'):
    print('ERROR: No real token in .env')
    sys.exit(1)

print(f'Token prefix : {token[:10]}...')
print(f'Token length : {len(token)}')
print()

from huggingface_hub import HfApi
api = HfApi()

# --- 1. whoami ---
try:
    info = api.whoami(token=token)
    name = info.get('name', '?')
    print(f'Logged in as : {name}')
except Exception as e:
    print(f'whoami FAILED: {type(e).__name__}: {e}')

print()

# --- 2. Model access checks ---
models_to_check = [
    'ai4bharat/indicwav2vec_v1_hindi',
    'ai4bharat/indicwav2vec_v1_tamil',
    'ai4bharat/indicwav2vec_v1_telugu',
    'ai4bharat/indicwav2vec-hindi',
]
for mid in models_to_check:
    try:
        minfo = api.model_info(mid, token=token)
        print(f'[OK]  {mid}')
    except Exception as e:
        etype = type(e).__name__
        msg = str(e)[:80]
        print(f'[403] {mid}  ({etype})')

"""Test from_pretrained with full traceback - bengali model."""
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
print(f'Token: {tok[:12]}... (len={len(tok)})')

from huggingface_hub import login
login(token=tok, add_to_git_credential=False)
print('Logged in OK')

import traceback
from transformers import Wav2Vec2Model

model_id = 'ai4bharat/indicwav2vec_v1_bengali'
print(f'Trying: {model_id}')
try:
    m = Wav2Vec2Model.from_pretrained(model_id, token=tok)
    print(f'SUCCESS: hidden_size={m.config.hidden_size}')
except Exception:
    traceback.print_exc()

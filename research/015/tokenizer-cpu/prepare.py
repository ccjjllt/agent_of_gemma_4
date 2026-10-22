"""Copy tokenizer/config assets from the official 31B CT v2 mount; CPU only."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

roots = [p.parent for p in Path('/kaggle/input').rglob('tokenizer.json')
         if 'gemma-4-31b-it-qat-w4a16-ct' in p.parts and '2' in p.parts]
if len(roots) != 1:
    raise RuntimeError('Official 31B CT v2 mount missing or ambiguous: ' + str(roots))
source = roots[0]
destination = Path('/kaggle/working/official-31b-ct-v2-tokenizer')
destination.mkdir()
required = ('tokenizer.json', 'tokenizer_config.json')
optional = ('chat_template.jinja', 'special_tokens_map.json', 'config.json',
            'generation_config.json', 'added_tokens.json')
files = {}
for name in required + optional:
    path = source / name
    if not path.is_file():
        if name in required:
            raise FileNotFoundError(path)
        continue
    shutil.copyfile(path, destination / name)
    files[name] = {'bytes': path.stat().st_size,
                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
config = json.loads((destination / 'tokenizer_config.json').read_text())
if 'chat_template.jinja' not in files and not config.get('chat_template'):
    raise RuntimeError('No actual deployment chat template in the official mount')
manifest = {'at_utc': datetime.now(timezone.utc).isoformat(),
            'source': 'google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2',
            'mount': str(source), 'files': files,
            'weights_read': False, 'gpu_used': False}
(destination / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2), flush=True)

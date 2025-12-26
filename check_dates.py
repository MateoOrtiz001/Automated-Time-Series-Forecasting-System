import json
from pathlib import Path
import pandas as pd

suameca_dir = Path('data/raw/banrep/suameca')
for f in sorted(suameca_dir.glob('*.json')):
    if 'manifest' in f.name:
        continue
    try:
        data = json.loads(f.read_text(encoding='utf-8'))
        if 'data' in data and data['data']:
            last_ts = data['data'][-1][0]
            last_date = pd.to_datetime(last_ts, unit='ms')
            key = f.stem.split('__')[2] if '__' in f.stem else f.stem
            print(f'{key}: {last_date.date()}')
    except Exception as e:
        print(f'{f.name}: ERROR - {e}')

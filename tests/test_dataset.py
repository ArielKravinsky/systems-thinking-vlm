from pathlib import Path
from src.dataset_loader import SystemsThinkingDataset

ds = SystemsThinkingDataset(Path('dataset'))
print(f'\n✓ Loaded {len(ds)} valid samples')
for s in ds.samples:
    print(f'  - Q{s["question_num"]}_S{s["subject_id"]}')

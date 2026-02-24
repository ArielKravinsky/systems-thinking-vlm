import json
import torch
from pathlib import Path
from src.pipeline import process_pair

def main():
    root = Path('dataset')
    images = sorted((root / 'images').glob('*.*'))
    answers = root / 'answers'
    model = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    translate_flag = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    for img in images:
        stem = img.stem
        ans_path = answers / f'{stem}.txt'
        if not ans_path.exists():
            continue
        answer_text = ans_path.read_text(encoding='utf-8').strip()
        caption_he, score, caption_en = process_pair(img, answer_text, model, translate_flag, device)
        results.append({
            'stem': stem,
            'score': score,
            'caption_en': caption_en,
            'caption_he': caption_he,
            'answer': answer_text,
        })
    Path('dataset_results.json').write_text(
        json.dumps({'device': device, 'results': results}, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print('Wrote dataset_results.json')


if __name__ == '__main__':
    main()

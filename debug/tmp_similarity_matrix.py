import json
from pathlib import Path
import torch
from sentence_transformers import util
from src.pipeline import generate_caption, translate_en_to_he, load_embed_model
from src.utils_hebrew import normalize_hebrew


def main():
    root = Path('dataset')
    images_dir = root / 'images'
    answers_dir = root / 'answers'
    model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}])
    answers = {}
    for ans_path in sorted(answers_dir.glob('*.txt')):
        answers[ans_path.stem] = ans_path.read_text(encoding='utf-8').strip()

    captions = {}
    for img in image_files:
        stem = img.stem
        cap_en = generate_caption(str(img), device=device)
        try:
            cap_he = translate_en_to_he(cap_en, device=device)
        except Exception:
            cap_he = cap_en
        captions[stem] = {'en': cap_en, 'he': cap_he}

    model = load_embed_model(model_name, device=device)
    answer_keys = sorted(answers.keys())
    image_keys = [p.stem for p in image_files]

    ans_texts = [normalize_hebrew(answers[k]) for k in answer_keys]
    cap_texts = [normalize_hebrew(captions[k]['he']) for k in image_keys]

    ans_emb = model.encode(ans_texts, convert_to_tensor=True)
    cap_emb = model.encode(cap_texts, convert_to_tensor=True)

    sim = util.cos_sim(cap_emb, ans_emb)  # rows: images, cols: answers

    print('Similarity matrix (rows=images, cols=answers):')
    header = ['img/ans'] + answer_keys
    print('\t'.join(header))
    for i, img_key in enumerate(image_keys):
        row_vals = [f"{sim[i][j].item():.4f}" for j in range(len(answer_keys))]
        print('\t'.join([img_key] + row_vals))

    # Save detailed JSON
    out = {
        'device': device,
        'model': model_name,
        'images': image_keys,
        'answers': answer_keys,
        'captions': captions,
        'answers_text': answers,
        'similarity': [[float(sim[i][j]) for j in range(sim.shape[1])] for i in range(sim.shape[0])],
    }
    Path('similarity_matrix.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Wrote similarity_matrix.json')


if __name__ == '__main__':
    main()

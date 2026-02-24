import time
from sentence_transformers import SentenceTransformer, util
from .utils_hebrew import normalize_hebrew
import torch


def compare_models(models, pairs, device='cpu'):
    results = {}
    for name in models:
        print(f"\nLoading model: {name}")
        t0 = time.time()
        m = SentenceTransformer(name, device=device)
        load_time = time.time() - t0
        sims = []
        enc_t0 = time.time()
        for a, b in pairs:
            a_n = normalize_hebrew(a)
            b_n = normalize_hebrew(b)
            emb = m.encode([a_n, b_n], convert_to_tensor=True)
            s = util.cos_sim(emb[0], emb[1]).item()
            sims.append(s)
        enc_time = time.time() - enc_t0
        results[name] = {
            'load_time_s': load_time,
            'encode_time_s': enc_time,
            'sims': sims
        }
        print(f"Model {name}: load {load_time:.2f}s, encode {enc_time:.2f}s")
    return results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    # small set of Hebrew sentence pairs for comparison
    pairs = [
        ("האדם יושב על ספסל בפארק", "איש יושב על ספסל בפארק"),
        ("זהו תיק רך בצבע כחול", "התמונה מציגה תיק כחול עם ידית"),
        ("הכלב רץ במהירות במסלול", "החתול שוכב על השטיח"),
        ("המכונית עומדת בחניה ליד הבית", "הרכב חונה קרוב לבניין"),
    ]

    models = [
        'sentence-transformers/LaBSE',
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    ]

    res = compare_models(models, pairs, device=device)
    print('\nResults:')
    for m, v in res.items():
        print(m)
        for i, s in enumerate(v['sims']):
            print(f'  pair {i+1}: {s:.4f}')
        print(f"  load {v['load_time_s']:.2f}s encode {v['encode_time_s']:.2f}s")

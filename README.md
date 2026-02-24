Hebrew image→text similarity pipeline

This project provides a minimal pipeline to:
- generate image captions (BLIP),
- optionally translate captions to Hebrew,
- normalize Hebrew text,
- compute embeddings with a Hebrew-capable model and compare similarity.

## Quick start

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate systems_thinking
```

Or manually:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download models (one-time setup)

```bash
python download_models.py
```

### 3. Run the pipeline

**BLIP Caption Pipeline:**
```bash
python -m src.pipeline --image example.jpg --answer "תשובת המשיבים בעברית" --embed-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

**VLM Question-Answering Pipeline:**
```bash
python -m src.pipeline_llm --dataset dataset --output vlm_results.json
```

## Files
- `src/pipeline.py`: BLIP caption-based pipeline and CLI
- `src/pipeline_llm.py`: VLM question-answering pipeline (LLaVA)
- `src/dataset_loader.py`: Dataset loader for systems thinking data
- `src/utils_hebrew.py`: Hebrew normalization helpers
- `download_models.py`: Model download script with progress
- `tests/test_pipeline.py`: basic unit tests
- `environment.yml`: conda environment specification

## Notes
- Models are downloaded from Hugging Face on first use. Use GPU if available.
- For best cross-lingual robustness use `sentence-transformers/LaBSE` if you need Hebrew↔English alignment.
- VLM pipeline requires ~16GB RAM (CPU) or ~8GB VRAM (GPU)

# Model Download and Pipeline Usage

## Overview
The VLM pipeline has been split into two separate steps:
1. **Download models** - One-time setup to download all required models
2. **Run pipeline** - Execute the actual analysis

## Step 1: Download Models

Run this command FIRST (one-time setup):

```powershell
C:\Users\arikk\miniconda3\envs\hebrew-img-text\python.exe download_models.py
```

Or use the PowerShell script:

```powershell
powershell -ExecutionPolicy Bypass -File download_models.ps1
```

### What it downloads:
- **LLaVA VLM Model** (~13GB) - llava-hf/llava-1.5-7b-hf
- **Embedding Model** (~1GB) - paraphrase-multilingual-mpnet-base-v2

### Progress indicators:
- Shows download progress for each model
- Displays checkmarks (✓) when complete
- Total time: 5-15 minutes (depends on internet speed)

### Options:
```bash
# Download both models (default)
python download_models.py

# Skip VLM download
python download_models.py --skip-vlm

# Skip embedding model download
python download_models.py --skip-embed

# Use different models
python download_models.py --vlm <model_name> --embed <model_name>
```

## Step 2: Run Pipeline

After models are downloaded, run the pipeline:

```powershell
C:\Users\arikk\miniconda3\envs\hebrew-img-text\python.exe -m src.pipeline_llm --dataset dataset --output vlm_results.json
```

Or use the PowerShell script:

```powershell
powershell -ExecutionPolicy Bypass -File run_vlm.ps1
```

### Progress tracking:
- ✓ Shows loading status for models
- ✓ tqdm progress bar for dataset processing
- ✓ Similarity scores printed for each sample

### Output:
Creates `vlm_results.json` with:
```json
{
  "device": "cuda/cpu",
  "vlm": "model_name",
  "embed": "model_name",
  "results": [
    {
      "stem": "11_2",
      "question": "השאלה בעברית...",
      "subject_answer": "תשובת הנבדק...",
      "llm_answer": "תשובת המודל...",
      "similarity": 0.75
    }
  ]
}
```

## Troubleshooting

### Error: "local_files_only"
If you see this error, models haven't been downloaded yet. Run:
```bash
python download_models.py
```

### Slow performance
- LLaVA is a large 7B parameter model
- CPU inference is slow (2-5 min per image)
- For faster processing, use CUDA/GPU

### Out of memory
- LLaVA requires ~16GB RAM (CPU) or ~8GB VRAM (GPU)
- Try closing other applications
- Or use a smaller VLM model

## Files Created

- `download_models.py` - Model download script with progress
- `download_models.ps1` - PowerShell wrapper for downloads
- `run_vlm.ps1` - PowerShell wrapper for pipeline (updated)
- `src/pipeline_llm.py` - Updated pipeline (loads only, doesn't download)

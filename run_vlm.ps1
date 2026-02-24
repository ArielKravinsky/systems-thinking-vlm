# Initialize conda for PowerShell
(& "C:\Users\arikk\miniconda3\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression

# Activate environment
conda activate systems_thinking

# Run pipeline
python -m src.pipeline_llm --dataset dataset --output vlm_results.json

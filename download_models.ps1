# Initialize conda for PowerShell
(& "C:\Users\arikk\miniconda3\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression

# Activate environment
conda activate systems_thinking

# Download models
python download_models.py

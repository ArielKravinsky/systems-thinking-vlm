@echo off
echo Starting VLM Pipeline...
echo This will take 5-10 minutes on CPU
echo.
C:\Users\arikk\miniconda3\envs\systems_thinking\python.exe -m src.pipeline_llm --dataset dataset --output vlm_results.json
echo.
echo Pipeline complete!
pause

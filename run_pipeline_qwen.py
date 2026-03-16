"""
run_pipeline_qwen.py
====================
Entry point for the Qwen2-VL-7B-Instruct pipeline.

Usage:
    conda run -n systems_thinking python run_pipeline_qwen.py
    conda run -n systems_thinking python run_pipeline_qwen.py --dataset dataset --output results/my_run.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline_qwen import main

if __name__ == "__main__":
    main()

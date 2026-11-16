#!/usr/bin/env python3
"""Rebuild all temporal features through the shared corrected daily pipeline.

The former rainfall-only patch mixed temporal conventions; it is intentionally
retired. Missing source inputs stop this command before tensor mutation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_weekly_dataset import run_pipeline

if __name__ == '__main__':
    run_pipeline()

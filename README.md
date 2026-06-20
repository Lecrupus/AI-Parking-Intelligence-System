# Parking Intelligence Prototype

Working prototype for illegal parking hotspot detection, congestion-risk proxy scoring, and enforcement prioritization.

## What Is Built

- CUDA-ready PyTorch environment in `.venv`
- Cleaned violation dataset in `data/processed/cleaned_violations.csv`
- Hotspot ranking in `data/processed/hotspots.csv`
- Enforcement priority queue in `data/processed/enforcement_priority.csv`
- Dashboard payload in `data/processed/dashboard_payload.json`
- Static map dashboard in `dashboard/`
- Reports in `reports/`
- GPU-trained proxy ranker in `models/gpu_priority_ranker.pt`

## Run

```powershell
.\.venv\Scripts\python.exe scripts\build_prototype.py
.\.venv\Scripts\python.exe scripts\train_priority_model.py
.\.venv\Scripts\python.exe -m http.server 8765
```

Open `http://localhost:8765/dashboard/`.

## Key Outputs

- `reports/prototype_summary.md`
- `reports/data_quality_report.md`
- `reports/scoring_formula.md`
- `reports/model_training_report.md`
- `reports/problem_statement.md`

## Current Dataset Window

The generated prototype uses the anonymized parking-violation CSV in this workspace. The latest build produced 298,445 clean rows and 7,814 hotspot zones.

## GPU Model

`scripts/train_priority_model.py` trains a CUDA-backed proxy ranker on hotspot features and saves it as `models/gpu_priority_ranker.pt`. The first build trained on `cuda` with a test MAE of about 0.49 priority points.

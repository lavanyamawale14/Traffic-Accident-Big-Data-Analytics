"""Run locally, from the ORIGINAL working project directory, with Hadoop running.
Exports actual Spark computations for the hosted Flask copy. Does not modify original app.py.
"""
import json
from pathlib import Path
from app import load_dashboard_data, load_analysis_data

output_dir = Path(__file__).resolve().parent / 'data'
output_dir.mkdir(parents=True, exist_ok=True)
for filename, result in (
    ('dashboard.json', load_dashboard_data()),
    ('analysis.json', load_analysis_data()),
):
    with (output_dir / filename).open('w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print('Created:', output_dir / filename)

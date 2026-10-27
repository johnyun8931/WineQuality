# Wine quality experiments

- `scripts/`: training, network implementation, and diagnostic/evaluation scripts.
- `analysis/data_analysis.py`: dataset analysis and graph generation.
- `analysis/plots/`: feature boxplots, correlation heatmap, and class distribution.
- `wine_quality.csv`: input dataset, kept at the project root for saved-run compatibility.
- `training_runs/`: original experiment archives.
- `training_analysis/`: results-only copies and cross-run summaries.
- `output/pdf/`: reports and LaTeX source.
- `requirements.txt`: Python environment dependencies.

From the project root, with the virtual environment activated:

```powershell
python scripts/submission.py
python analysis/data_analysis.py
```

Module execution also works: `python -m scripts.submission` and
`python -m analysis.data_analysis`. Analysis graphs always save to
`analysis/plots/`, regardless of the current working directory.

See [TRAINING.md](TRAINING.md) for diagnostic commands and experiment settings.
Archived run code and historical reports retain their original filenames.

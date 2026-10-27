# Dataset analysis

Run from the project root:

```powershell
python analysis/data_analysis.py
```

The script uses the same preprocessing and current sampling settings as
`scripts/submission.py` without training the network. Graphs are saved in
`plots/`: feature boxplots, numeric feature correlations, and target counts.

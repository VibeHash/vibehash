# vibehash predict

Use a downloaded model on new rows of a CSV file.

```
vibehash predict [-h] [--out OUT] [--rows ROWS] [model] data
```

Arguments:

- `model` (optional): the .npz file `job download` saved (default: the last one).
- `data`: CSV with the same input columns.

Flags:

- `--out OUT`: write every row with its prediction to this CSV.
- `--rows ROWS`: rows to show when not writing a file (default 20).

Examples:

```bash
vibehash predict new-rows.csv
vibehash predict fruit-demo.npz new-rows.csv --out predictions.csv
```

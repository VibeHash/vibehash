# vibehash cache

Big model files kept for jobs: where, how big and why; `cache clear` removes them.

```
vibehash cache [-h] [--json] [{show,clear,sweep}] [job]
```

Arguments:

- `action` (optional): one of
  - `show`: what is stored, where, how big and why (the default)
  - `clear`: remove it all, or one job's files
  - `sweep`: remove files no job has used for a week, and stray partial downloads
- `job` (optional): with clear: only this job's files.

Flags:

- `--json`: machine-readable output.

Examples:

```bash
vibehash cache
vibehash cache clear
vibehash cache clear fruit-demo
vibehash cache sweep
```

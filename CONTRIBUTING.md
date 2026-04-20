# Contributing to TRAS

Thanks for your interest! We welcome issues, bug reports, and pull requests.

## Reporting issues
Please open a GitHub issue with:
- The config YAML you ran (redact any API keys),
- The full console output / traceback,
- Python version and OS.

## Pull requests
- Fork, branch, and open a PR against `main`.
- Run the smoke tests locally: `poetry run pytest tests/test_smoke.py`.
- Keep the paper ↔ code terminology consistent with [`docs/CONCEPTS.md`](docs/CONCEPTS.md).

## Citing in derivative work
If you build on TRAS, please cite the paper (see [`CITATION.cff`](CITATION.cff)
or the BibTeX below):

```bibtex
@inproceedings{davari2026tras,
  title   = {Stabilizing Black-Box Prompt Optimization with Textual Regularization and Signal Aggregation},
  author  = {Davari, MohammadReza and Garg, Utkarsh and Cai, Weixin and Belilovsky, Eugene},
  booktitle = {The 39th Canadian Conference on Artificial Intelligence (Canadian AI 2026)},
  year    = {2026},
}
```

## Code of conduct
Be respectful. Feedback on methods, configs, or edge cases is welcome; personal
attacks are not.

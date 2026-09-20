# Contributing

Thanks for improving the bilingual RAG sample.

## Development

1. Create a repository from the template or fork the project.
2. Open it in GitHub Codespaces, or create a Python 3.10+ virtual environment locally.
3. Install the development dependencies with `python -m pip install -r requirements-dev.txt`.
4. Run `python -m pytest -q` before opening a pull request.

The test suite is fully offline and does not require Azure credentials. Live ingestion, queries, and
evaluation require the resources described in [`docs/azure-setup.md`](docs/azure-setup.md).

## Pull requests

- Keep changes focused and include tests for changed behavior.
- Never commit `.env`, credentials, customer documents, or evaluation data containing private content.
- Treat the included policies and evaluation questions as synthetic examples only.
- Explain any changes to prompts, retrieval, refusal behavior, or evaluation metrics.

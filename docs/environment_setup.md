# Environment Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

## Prerequisites
- Python 3.10
- Poetry installed (`pip install --user poetry`)

## Installing Dependencies
```bash
poetry install
```

To spawn a shell with the project's virtual environment:
```bash
poetry shell
```

## Using pip
If you need a `requirements.txt` file for tools that rely on `pip`, export it from
`pyproject.toml`:
```bash
poetry export --without-hashes -f requirements.txt > requirements.txt
```
Then install:
```bash
pip install -r requirements.txt
```

`requirements_v23.txt` and `zanllink/requirements.txt` are synchronized from
`pyproject.toml` and can be regenerated the same way.

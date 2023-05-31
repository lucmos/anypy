# Generic Python Tools & Utilities

<p align="center">
    <a href="https://pypi.org/project/anypy/"><img alt="PyPi" src=https://img.shields.io/pypi/v/anypy></a>
    <a href="https://pypi.org/project/anypy/"><img alt="PyPi" src=https://img.shields.io/pypi/dm/anypy></a>
    <a href="https://github.com/lucmos/anypy/actions/workflows/publish.yml"><img alt="CI" src=https://github.com/lucmos/anypy/actions/workflows/publish.yml/badge.svg?event=release></a>
    <a href="https://lucmos.github.io/anypy"><img alt="Docs" src=https://img.shields.io/github/deployments/lucmos/anypy/github-pages?label=docs></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

A collection of python utilities, without hard dependencies


## Installation

```bash
pip install anypy
```


## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment:

```bash
git clone git@github.com:lucmos/anypy.git
cd anypy
conda env create -f env.yaml
conda activate anypy
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```

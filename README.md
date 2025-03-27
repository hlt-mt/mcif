# IWSLT2025

This repository contains code used for the IWSLT 2025 Intrduction Following shared task.

This includes scripts used to create test sets and their references, as well as scripts used in the evaluation.

## Installation

The scripts are standalone and require no installation beside the PyYAML (`yaml`) package,
which can be installed with:

```shell
pip install PyYAML
```

For the evaluation script, other requirements are:

```shell
pip install jiwer==3.0.5
pip install bert_score==0.3.13
pip install unbabel-comet==2.2.4
pip install whisper_normalizer==0.0.10
```

The scripts have been tested with Python 3.9.

## License

Licensed under Apache 2.0 Licence.
# MCIF

This repository contains code used for the MCIF benchmark and the IWSLT 2025 Instruction Following
shared task.

This includes:

 - scripts used to create test sets and their references ([dataset_build/README.md]());
 - scripts used to create the baselines ([baselines/README.md]());
 - evaluation code.


For the dataset creation and baselines please refer to the dedicated READMEs.

## Installation

The repository can be installed with `pip install -e .`.

## Usage

For the evaluation, you can simply run:

```shell
mcif_eval -t {short/long} -l {en/de/it/zh} \
    -s model_outputs.xml -r MCIF1.0.IF.{short/long}.{en/de/it/zh}.ref.xml
```

## License

Licensed under Apache 2.0 Licence.
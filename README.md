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

For the `shas_mp4_segmentation` script, you need to install:

```shell
pip install moviepy
```

The scripts have been tested with Python 3.9.

# Usage

For the evaluation, you can simply run:

```shell
python instruction_following/evaluation.py -t {short/long} -l {en/de/it/zh} \
    -s model_outputs.xml -r IWSLT2025.IF.{short/long}.{en/de/it/zh}.ref.xml
```

For the dataset creation, in case you want to include videos you first need to
create the automatic segmentation of the full videos with:

```shell
python instruction_following/shas_mp4_segmentation.py -s TEST_SET/VIDEO \
    -o TEST_SET/SEGMENTED_VIDEO/ -d TEST_SET/SEGMENTED_AUDIO/shas_segmentation.yaml
```

and then run the script that generates the test set. To obtain MCIF, use the command:

```shell
python instruction_following/testset_generator.py --include-video --include-text \
    --output-dir IWSLT2025_WITHVIDEO --source-dir TEST_SET --threshold-short-audio 100
```

where TEST_SET contains the raw data.

## License

Licensed under Apache 2.0 Licence.
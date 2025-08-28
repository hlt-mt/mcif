# Installation

The scripts are standalone and require no installation beside the PyYAML (`yaml`) package,
which can be installed with:

```shell
pip install PyYAML
```

For the `shas_mp4_segmentation` script, you need to install:

```shell
pip install moviepy
```

The scripts have been tested with Python 3.9.

# Usage

For the dataset creation, in case you want to include videos you first need to
create the automatic segmentation of the full videos with:

```shell
python dataset_build/shas_mp4_segmentation.py -s TEST_SET/VIDEO \
    -o TEST_SET/SEGMENTED_VIDEO/ -d TEST_SET/SEGMENTED_AUDIO/shas_segmentation.yaml
```

and then run the script that generates the test set. To obtain MCIF, use the command:

```shell
python dataset_build/testset_generator.py --include-video --include-text \
    --output-dir MCIF_v0.2 --source-dir TEST_SET --threshold-short-audio 100
```

where TEST_SET contains the raw data.

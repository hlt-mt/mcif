# MCIF - Multimodal Crosslingual Instruction-Following 

<p align="center">
<img src="https://github.com/hlt-mt/mcif/blob/main/mcif.png?raw=true" alt="MCIF Logo" width="600"/>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2507.19634">
    <img src="https://img.shields.io/badge/arXiv%3A2507.19634-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv:2507.19634" height="20"/> 
  </a> 
  <a href="https://huggingface.co/datasets/FBK-MT/MCIF">
    <img src="https://img.shields.io/badge/HuggingFace-FBK--MT%2FMCIF-FFBA08?style=flat&logo=HuggingFace&logoColor=white" alt="HuggingFace Dataset FBK-MT/MCIF" height="20"/>
  </a>
</p>

MCIF is a comprehensive benchmark for evaluating **multimodal, crosslingual instruction-following**
systems, which covers *3 modalities* (text, speech, and video), *4 languages* (English, German, 
Italian, and Chinese), and *13 tasks* (organized into 4 macro-tasks).

A subset of MCIF has been used for the evaluation of the 
[IWSLT 2025 **Instruction-Following** Shared Task](https://iwslt.org/2025/instruction-following).


## 📰 News

2025.10.22: 🎉 [The MCIF paper has been accepted at ICLR!](https://openreview.net/forum?id=PtPYZYfa0h)</br>
2025.10.22: 🤗 [MCIF test set is released on HuggingFace](https://huggingface.co/datasets/FBK-MT/MCIF)</br>
2025.10.21: ⭐️ MCIF Evaluation first release

## 📦 Repository Structure

The evaluation is the core component of this repository.
All other components (i.e., dataset construction and baseline inference) are included to ensure 
full reproducibility and transparency of the evaluation results.

For details on dataset generation or baseline models, please refer to the dedicated READMEs 
(baselines may require specific dependencies):

- 🧱 Dataset Construction — scripts and guidelines for creating test sets and references
→ [dataset_build/README.md](dataset_build/README.md)

- 🚀 Baselines — inference scripts and outputs for baseline systems
→ [baselines/README.md](baselines/README.md)

- 📊 Evaluation — scoring and comparison utilities for submitted outputs → [README.md](README.md#️-evaluation-usage)


## ⚙️ Installation

You can install the latest stable version from PyPI:

```shell
pip install mcif-bench
```

Or, to install from source:

```shell
git clone https://github.com/hlt-mt/mcif.git
cd mcif
pip install .
```

Notice that, since some evaluation metrics run on GPU (e.g., COMET), you may want to first
set up your environment installing versions of torch with CUDA support for your GPU.

For development (with docs and testing tools):

    pip install .[dev]

## ▶️ Usage

For the evaluation, you can simply run:

```shell
mcif_eval -t {short/long} -l {en/de/it/zh} -s model_outputs.xml
```

where `model_outputs.xml` contains the outputs of your model for the selected track or context 
length (`short` or `long`) and target language among English (`en`), German (`de`), Italian (`it`) 
and Chinese (`zh`).

This will automatically download the reference from the Huggingface repository
for the latest MCIF version. If you want to specify a different version, use `-v`.
To run the evaluation without internet access, first download the MICF references
and then provide them to `mcif_eval` with the `-r` parameter.

The file containing the model outputs to evaluate must be structured as follows:

```xml
<?xml version='1.0' encoding='utf-8'?>
<testset name="MCIF" type="output">
  <task track="{short/long}" text_lang="{en/de/it/zh}">
    <sample id="1">{SAMPLE1_CONTENT}</sample>
    <sample id="2">{SAMPLE2_CONTENT}</sample>
   ....
  </task>
</testset>
```

To ease usability, we provide a helper function ([`mcif.io.write_output`](src/mcif/io.py)) that
automatically formats model predictions into the XML structure required by the MCIF evaluation
script.
The method takes as input:
- `samples`: a list of `mcif.io.OutputSample` containing the sample id and its related prediction;
- `track`: the context length or track (`short/long`);
- `language`: the target language (`en/de/it/zh`);
- `output_name`: the semantic name of the output (e.g. `My model`);
- `output`: a path or a byte buffer where the XML file containing all system's outputs, ready for
evaluation, is written.

## 📜 License

MCIF is released under the [Apache 2.0 License](LICENSE).

## 🧩 Citation

If you use MCIF in your research, please cite:

```bibtex
@inproceedings{papi2026mcif,
title={{MCIF}: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks},
author={Sara Papi and Maike Z{\"u}fle and Marco Gaido and Beatrice Savoldi and Danni Liu and Ioannis Douros and Luisa Bentivogli and Jan Niehues},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=PtPYZYfa0h}
}
```
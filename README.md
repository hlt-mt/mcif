# MCIF - Multimodal Crosslingual Instruction-Following Benchmark

<p align="center">
<img src="mcif.png" alt="MCIF Logo" width="600"/>
</p>

MCIF is a comprehensive benchmark for evaluating **multimodal, multilingual instruction-following**
systems, which covers *3 modalities* (text, speech, and video), *4 languages* (English, German, 
Italian, and Chinese), and *13 tasks* (organized in 4 macro-tasks).

A subset of MCIF has been for the evaluation of the 
[IWSLT 2025 **Instruction-Following** Shared Task](https://iwslt.org/2025/instruction-following).


This repository provides code for dataset creation, baseline systems, and evaluation.

## 📰 News

2025.10.20: ⭐️ MCIF Evaluation first release

## 📦 Repository Structure

- 🧱 Dataset Construction — scripts for creating test sets and references
→ [dataset_build/README.md](dataset_build/README.md)

- 🚀 Baselines — training and inference scripts for baseline systems
→ [baselines/README.md](baselines/README.md)

- 📊 Evaluation — scoring and comparison utilities for submitted outputs → [README.md](README.md#️-evaluation-usage)

For details on dataset generation or baseline models, please refer to the dedicated READMEs.

## ⚙️ Installation

The repository can be installed with `pip install -e .`.

## ▶️ Evaluation Usage

For the evaluation, you can simply run:

```shell
mcif_eval -t {short/long} -l {en/de/it/zh} \
    -s model_outputs.xml -r MCIF1.0.IF.{short/long}.{en/de/it/zh}.ref.xml
```

## 📜 License

MCIF is released under the [Apache 2.0 License](LICENSE).

## 🧩 Citation

If you use MCIF in your research, please cite:

```bibtex
@misc{mcif,
      title={MCIF: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks}, 
      author={Sara Papi and Maike Züfle and Marco Gaido and Beatrice Savoldi and Danni Liu and Ioannis Douros and Luisa Bentivogli and Jan Niehues},
      year={2025},
      eprint={2507.19634},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.19634}, 
}
```
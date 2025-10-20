# MCIF - Multimodal Crosslingual Instruction-Following 

<p align="center">
<img src="mcif.png" alt="MCIF Logo" width="600"/>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2507.19634">
    <img src="https://img.shields.io/badge/arXiv%3A2507.19634-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv:2507.19634" width="130"/>
  </a>
</p>

MCIF is a comprehensive benchmark for evaluating **multimodal, multilingual instruction-following**
systems, which covers *3 modalities* (text, speech, and video), *4 languages* (English, German, 
Italian, and Chinese), and *13 tasks* (organized in 4 macro-tasks).

A subset of MCIF has been for the evaluation of the 
[IWSLT 2025 **Instruction-Following** Shared Task](https://iwslt.org/2025/instruction-following).


## 📰 News

2025.10.20: ⭐️ MCIF Evaluation first release

## 📦 Repository Structure

The evaluation is the core component of this repository.
All other components (i.e., dataset construction and baseline inference) are included to ensure 
full reproducibility and transparency of the evaluation results.

For details on dataset generation or baseline models, please refer to the dedicated READMEs 
(baselines may require specific dependencies):

- 🧱 Dataset Construction — scripts and guidelines for creating test sets and references
→ [dataset_build/README.md](dataset_build/README.md)

- 🚀 Baselines — inference scripts for baseline systems
→ [baselines/README.md](baselines/README.md)

- 📊 Evaluation — scoring and comparison utilities for submitted outputs → [README.md](README.md#️-evaluation-usage)


## ⚙️ Installation

The repository can be installed with `pip install -e .`.

## ▶️ Evaluation Usage

For the evaluation, you can simply run:

```shell
mcif_eval -t {short/long} -l {en/de/it/zh} \
    -s model_outputs.xml -r MCIF1.0.IF.{short/long}.{en/de/it/zh}.ref.xml
```

where `model_outputs.xml` contains the outputs of your model for the selected track or context 
length (`short` or `long`) and target language among English (`en`), German (`de`), Italian (`it`) 
and Chinese (`zh`), and is structured as follows:

```xml
<?xml version='1.0' encoding='utf-8'?>
<testset name="MCIF" type="output">
  <task track="{short/long}" text_lang="{en/de/it/zh}">
    <sample id="1">{SAMPLE1_CONTENT}</sample>
    <sample id="3">{SAMPLE2_CONTENT}</sample>
   ....
  </task>
</testset>
```

To ease usability, we provide [a helper function](baselines/utils.py#L104) that automatically 
formats model predictions into the XML structure required by the MCIF evaluation script.
The method takes as input:
- `outputs`: a list of tuples (`sample_id`, `prediction`) containing the sample id and its related 
prediction;
- `lang`: the target language (`en/de/it/zh`);
- `track`: the context length or track (`short/long`);
- `output_file`: the path to the XML file being created containing all system's outputs, ready 
for evaluation.

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
# MCIF Baselines

This folder provides baseline scripts for running **MCIF experiments** across different models, languages, and modalities.  

---

## ⚙️ Setup

### Requirements
Each baseline model may require different versions of Hugging Face libraries.  
Please refer to the model-specific documentation on Hugging Face for installation details.  

- **[inclusionAI/Ming-Lite-Omni](https://huggingface.co/inclusionAI/Ming-Lite-Omni)** and **[THUdyh/Ola-7b](https://github.com/Ola-Omni/Ola?tab=readme-ov-file)** require downloading their official codebases in addition to the model weights.  

---

## 🚀 Running Baselines

### Example
```bash
python main.py \
  --model llama \
  --lang en \
  --track long \  
  --modality text \ 
  --prompt fixed \ 
  --in_data_folder MCIF \ 
  --out_folder output
```

### Arguments
| Argument           | Description                                    | Options / Examples                                                      |
|--------------------|------------------------------------------------|-------------------------------------------------------------------------|
| `--model`          | Model to use                                   | See `main.py` for the full list. Examples: `llama`, `gemma`, `ola`, ... |
| `--lang`           | Language code                                  | `en`, `de`, `it`, `zh`                                                  |
| `--track`          | Track type                                     | `short`, `long`                                                         |
| `--modality`       | Input modality                                 | `text`, `audio`, `video`, `mllm`                                        |
| `--prompt`         | Prompt type                                    | `fixed`, `mixed`                                                        |
| `--in_data_folder` | Path to the input data folder                  | e.g., `MCIF/`                                                           |
| `--out_folder`     | Path to save generated outputs                 | e.g., `output/`                                                         |




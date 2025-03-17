<div align="left">
  <a href="https://2024.emnlp.org/"><img src="https://img.shields.io/badge/EMNLP-Conference-blue" alt="EMNLP"></a>
  <a href="https://aclanthology.org/2024.emnlp-main.350"><img src="https://img.shields.io/badge/ACL-Anthology-B31B1B" alt="ACL"></a>
  <a href="https://huggingface.co/datasets/uzw/Emerging_NLP"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?color=8A2BE2" alt="Dataset"></a>
</div>


# SciPrompt
`SciPrompt` is a framework designed to automatically retrieve scientific topic-related terms for low-resource text classification tasks, including few-shot and zero-shot settings.


## News
- (2025.03.16) [`Emerging NLP`](https://huggingface.co/datasets/uzw/Emerging_NLP) dataset is available on 🤗 Hugging Face!
- (2025.03.16) Our datasets can be downloaded [here](https://drive.google.com/file/d/1w5IxtfayNPlrAE6I_vp2zKhx8ntDMRMv/view).
- (2024.12.01) Download the fine-tuned `filtering models`:
    - [Bi-Encoder Model](https://drive.google.com/file/d/1PLoMoqr14Kc4RHCglMw_WMn_U0GKAxBN/view?usp=sharing)
    - [Cross-Encoder Model](https://drive.google.com/file/d/1-xH453E-2GsejNdFc9gLg6xrurXembn4/view?usp=sharing)



> This project is developed based on the [OpenPrompt](https://github.com/thunlp/OpenPrompt) framework.

## Overall Framework
<div align="center">
  <img src="https://github.com/zhiwenyou103/SciPrompt/blob/main/pics/system.jpg" height="400" width="600">
</div>


## Installation
To install the necessary Python packages, clone this repo and then run the following command:
```bash
conda create -n sciprompt python=3.8.12
pip install -r requirements.txt
```

## Prepare the Required Files and Directories

- Replace the placeholder paths in the script with actual paths to your data and configuration files:
  - `--data_dir` should point to your data directory
  - `--verbalizer_path` should point to your arXiv_knowledgable_verbalizer.txt
  - `--semantic_score_path` should point to your arXiv_knowledgable_verbalizer_semantic_search_scores.txt
  - `--doc_id_path` should point to your doc_id.txt
  - `--config_path` should point to config/arxiv_label_mappings.json
- Prepare your class label dictionary similar to the `.json` files in the [label_mappings](https://github.com/zhiwenyou103/SciPrompt/tree/main/label_mappings) folder
  
## Knowledge Retrieval and Filtering

- Run through our datasets:
  - Step 1: Change paths in `run_retrieval.sh` and run `bash run_retrieval.sh`
  - Step 2: Change paths of the filtering model, retrieved data (from Step 1), and output files in the `run_knowledge_filtering.sh` script
  - Step 3: Run the filtering script:
    ```bash
    bash run_knowledge_filtering.sh
    ```

- Run using your own dataset:
  - Step 1 and 2 are the same as above
  - Step 3: Change your dataset name as `custom` and corresponding configs into the `dataset_configs` dictionary in `knowledge_filtering.py` [Line 206](https://github.com/zhiwenyou103/SciPrompt/blob/main/knowledge_filtering.py#L206)
  - Run `bash run_knowledge_filtering_customized.sh`

## Run the main script:

- Execute scripts for each dataset:
```bash
bash run_arxiv.sh
bash run_s2orc.sh
bash run_sdpra.sh
```
- Run on your own data (need two input files: one only contains data, one only has labels, as used in [arXiv](https://github.com/zhiwenyou103/SciPrompt/tree/main/data/arXiv)):
```bash
bash run_custom_script.sh
```

**Note: Please modify the required data file paths inside each script before running.**


## Citation Information
For the use of SciPrompt and Emerging NLP benchmark, please cite:
```bibtex

@inproceedings{you-etal-2024-sciprompt,
    title = "{S}ci{P}rompt: Knowledge-augmented Prompting for Fine-grained Categorization of Scientific Topics",
    author = "You, Zhiwen  and
      Han, Kanyao  and
      Zhu, Haotian  and
      Ludaescher, Bertram  and
      Diesner, Jana",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.350",
    pages = "6087--6104",
}
```
## Contact Information
If you have any questions, please email `zhiweny2@illinois.edu`.

[![Conference](https://img.shields.io/badge/EMNLP-2024-4b44ce)](https://2024.emnlp.org/)
[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://aclanthology.org/2024.emnlp-main.350.pdf)

# SciPrompt
The official repository of our EMNLP 2024 Main conference paper: [SciPrompt: Knowledge-augmented Prompting for Fine-grainedCategorization of Scientific Topics](https://aclanthology.org/2024.emnlp-main.350/)

This project is developed based on the [OpenPrompt](https://github.com/thunlp/OpenPrompt) framework.

**Download our fine-tuned filtering models:** [Bi-Encoder Model](https://drive.google.com/file/d/1PLoMoqr14Kc4RHCglMw_WMn_U0GKAxBN/view?usp=sharing) [Cross-Encoder Model](https://drive.google.com/file/d/1-xH453E-2GsejNdFc9gLg6xrurXembn4/view?usp=sharing)



## Overall Framework
<div align="center">
  <img src="https://github.com/zhiwenyou103/SciPrompt/blob/main/pics/system.jpg" height="400" width="600">
</div>


## Installation
To install the necessary Python packages, clone this repo and then run the following command:
```bash
conda create -n SciPrompt python=3.8.12
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

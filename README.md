# SciPrompt
The official repository of our EMNLP 2024 Main conference paper: [SciPrompt: Knowledge-augmented Prompting for Fine-grainedCategorization of Scientific Topics](https://aclanthology.org/2024.emnlp-main.350/)

**Fine-tuned filtering model:** [Google Drive](https://drive.google.com/drive/folders/1z38v6nx0pss_hhD2dX9Kg7NXQSNVSIWK?usp=sharing)


### Overall Framework
<div align="center">
  <img src="https://github.com/zhiwenyou103/SciPrompt/blob/main/pics/system.jpg" height="300" width="500">
</div>


### Installation
To install the necessary Python packages, run the following command:
```bash
conda create -n SciPrompt python=3.8.12
pip install -r requirements.txt
```

### Prepare the Required Files and Directories

- Replace the placeholder paths in the script with actual paths to your data and configuration files:
  - --data_dir should point to your data directory
  - --verbalizer_path should point to your arXiv_knowledgable_verbalizer.txt
  - --semantic_score_path should point to your arXiv_knowledgable_verbalizer_semantic_search_scores.txt
  - --doc_id_path should point to your doc_id.txt
  - --config_path should point to config/arxiv_label_mappings.json

### Run the main script:

- Execute scripts for each dataset:
```bash
bash run_arxiv.sh
bash run_s2orc.sh
bash run_sdpra.sh
```
**Note: Please modify the required data file paths inside each script before running.**


### Citation Information
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


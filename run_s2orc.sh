python s2orc_script.py \
    --cuda_device 0 \
    --zero_shot no \
    --shots 5 \
    --data_dir "/path/to/S2ORC/data" \
    --verbalizer_path "/path/to/S2ORC_knowledgable_verbalizer.txt" \
    --semantic_score_path "/path/to/S2ORC_knowledgable_verbalizer_semantic_search_scores.txt" \
    --config_path "config/s2orc_label_mappings.json"
python s2orc_script.py \
    --seed 144 \
    --shots 5 \ # training shots
    --calibration True \
    --max_seq_length 256 \
    --batch_size 5 \
    --max_epochs 5 \
    --cuda_device 0 \
    --learning_rate 3e-5 \
    --soft_verbalizer False \ # whether using the Soft verbalization method 
    --data_dir "/path/to/your/data/" \ # e.g., data/S2ORC
    --verbalizer_path "knowledge_output/S2ORC_output_words.txt" \ # filtered verbalizer
    --semantic_score_path "scores/S2ORC_output_scores.tx" \
    --doc_id_path "data/S2ORC/label_dict.txt" \
    --config_path "label_mappings/S2ORC_label_mappings.json" \
    --zero_shot no

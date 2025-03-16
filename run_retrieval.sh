CONFIG_PATH="label_mappings/S2ORC_label_mappings.json" # change the path
OUTPUT_PATH="knowledge_output/s2orc_plain_label_words.json" # change the corresponding output path 

python retrieval.py \
    --config_path "${CONFIG_PATH}" \
    --output_path "${OUTPUT_PATH}"

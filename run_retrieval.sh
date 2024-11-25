CONFIG_PATH="label_mappings/s2orc_label_mappings.json"
OUTPUT_PATH="s2orc_plain_label_words.json"

python retrieval.py \
    --config_path "${CONFIG_PATH}" \
    --output_path "${OUTPUT_PATH}"
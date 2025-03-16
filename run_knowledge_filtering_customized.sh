# Run on your own datasets
python knowledge_filtering.py \
    --dataset custom \
    --bi_encoder_path /path/to/nli_model \
    --cross_encoder_path /path/to/cross_encoder \
    --custom_input /path/to/input.json \
    --custom_output_words /path/to/output_words.txt \
    --custom_output_scores /path/to/output_scores.txt \
    --ce_threshold 0.9 \
    --semantic_threshold 0.5 \
    --device cuda:0

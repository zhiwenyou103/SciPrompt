# Common parameters
BI_ENCODER_PATH="path/to/bi_encoder_model"
CROSS_ENCODER_PATH="path/to/cross_encoder_model"
CE_THRESHOLD=0.9
SEMANTIC_THRESHOLD=0.5
DEVICE="cuda:0"

run_filtering() {
    local dataset=$1
    local input=$2
    local output_words=$3
    local output_scores=$4

    python knowledge_filtering.py \
        --dataset $dataset \
        --bi_encoder_path $BI_ENCODER_PATH \
        --cross_encoder_path $CROSS_ENCODER_PATH \
        --${dataset}_input $input \
        --${dataset}_output_words $output_words \
        --${dataset}_output_scores $output_scores \
        --ce_threshold $CE_THRESHOLD \
        --semantic_threshold $SEMANTIC_THRESHOLD \
        --device $DEVICE
}

# Example usage for S2ORC
run_filtering "s2orc" \
    "knowledge_output/s2orc_plain_label_words.json" \
    "knowledge_output/s2orc_output_words.txt" \
    "scores/s2orc_output_scores.txt"

# Example usage for arXiv
# run_filtering "arxiv" \
#     "knowledge_output/arxiv_plain_label_words.json" \
#     "knowledge_output/arxiv_output_words.txt" \
#     "scores/arxiv_output_scores.txt"

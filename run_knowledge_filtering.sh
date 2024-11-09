# For S2ORC
# python label_word_processor.py \
#     --dataset s2orc \
#     --s2orc_input "/path/to/S2ORC_plain_label_words.json" \
#     --s2orc_output "/path/to/S2ORC_plain_label_words.txt"

# # For arXiv
# python label_word_processor.py \
#     --dataset arxiv \
#     --arxiv_input "/path/to/arXiv_plain_label_words.json" \
#     --arxiv_output "/path/to/arXiv_plain_label_words.txt"

# # For SDPRA
# python label_word_processor.py \
#     --dataset sdpra \
#     --sdpra_input "/path/to/nlp_plain_label_words.json" \
#     --sdpra_output "/path/to/nlp_plain_label_words.txt"

# process all datasets
python label_word_processor.py \
    --dataset all \
    --s2orc_input "/path/to/S2ORC_plain_label_words.json" \
    --s2orc_output "/path/to/S2ORC_plain_label_words.txt" \
    --arxiv_input "/path/to/arXiv_plain_label_words.json" \
    --arxiv_output "/path/to/arXiv_plain_label_words.txt" \
    --sdpra_input "/path/to/nlp_plain_label_words.json" \
    --sdpra_output "/path/to/nlp_plain_label_words.txt"
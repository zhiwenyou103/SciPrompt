from retrieval_utils import fetchRelatedWords
import argparse

def parse_args():
    parser.add_argument('--config_path', type=str,
                       default='label_mappings/s2orc_label_mappings.json',
                       help='Path to label configuration file')
    parser.add_argument('--output_path', type=str,
                       default='nlp_plain_label_words.json',
                       help='Path to output file')
    return parser.parse_args()

def has_duplicates_in_values(data):
    seen_words = set()
    for values in data.values():
        for value in values:
            if value in seen_words:
                return True
            seen_words.add(value)
    return False

def main():
    args = parse_args()
    label_dict, label_name_mapping = load_config(args.config_path)
    
    words_dict = {}
    scores_dict = {}
    deduplicate_words = set()
    for class_name, class_label in label_name_mapping.items():
        retrieved_words = fetchRelatedWords(class_label[0])
        words_dict[class_name] = []
        scores_dict[class_name] = []
        for item in retrieved_words:
            retrieved_word = item['word']
            score = item['score']
            if retrieved_word not in words_dict[class_name] and retrieved_word not in deduplicate_words:
                words_dict[class_name].append(retrieved_word)
                scores_dict[class_name].append(score)
                deduplicate_words.add(retrieved_word)
            else:
                continue
    
    for key, value in words_dict.items():
        if not value:
            print(f"The value list for key '{key}' is empty.")
            
    has_duplicates = has_duplicates_in_values(words_dict)
    if has_duplicates:
        print("Duplicates found in the values.")
    else:
        print("No duplicates found in the values.")
        
    with open(args.output_path, "w") as file:
        json.dump(words_dict, file)
    

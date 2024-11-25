import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

def parse_args():
    parser = argparse.ArgumentParser(description='Process label words for multiple datasets')
    parser.add_argument('--dataset', type=str, required=True,
                    #   choices=['s2orc', 'arxiv', 'sdpra', 'all'],
                      help='Which dataset to process (s2orc, arxiv, sdpra, all, or your own dataset)')
    
    parser.add_argument('--s2orc_input', type=str,
                      default='/path/to/S2ORC_plain_label_words.json',
                      help='Path to S2ORC input JSON')
    parser.add_argument('--s2orc_output', type=str,
                      default='/path/to/S2ORC_plain_label_words.txt',
                      help='Path to S2ORC output TXT')
    
    parser.add_argument('--arxiv_input', type=str,
                      default='/path/to/arXiv_plain_label_words.json',
                      help='Path to arXiv input JSON')
    parser.add_argument('--arxiv_output', type=str,
                      default='/path/to/arXiv_plain_label_words.txt',
                      help='Path to arXiv output TXT')
    
    parser.add_argument('--sdpra_input', type=str,
                      default='/path/to/sdpra_plain_label_words.json',
                      help='Path to SDPRA input JSON')
    parser.add_argument('--sdpra_output', type=str,
                      default='/path/to/sdpra_plain_label_words.txt',
                      help='Path to SDPRA output TXT')
    
    parser.add_argument('--new_dataset_input', type=str,
                  default='/path/to/new_dataset_plain_label_words.json',
                  help='Path to New Dataset input JSON')
    parser.add_argument('--new_dataset_output', type=str,
                    default='/path/to/new_dataset_plain_label_words.txt',
                  help='Path to New Dataset output TXT')
    
    return parser.parse_args()

def process_label_words(input_json_path, output_txt_path):
    """Process label words from JSON to TXT format"""
    all_label_words = []
    
    Path(output_txt_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_json_path, 'r') as file:
        data = json.load(file)
        for key, value in data.items():
            label_words = [key] + value
            all_label_words.append(label_words)
            
    with open(output_txt_path, 'w') as output_file:
        for label_words in all_label_words:
            line = ','.join(label_words)
            output_file.write(line + '\n')
            
    return len(all_label_words)

def read_label_words(file_path):
    """Read and process label words from TXT file"""
    label_words_all = []
    label_words_single_group = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "":
                if len(label_words_single_group) > 0:
                    label_words_all.append(label_words_single_group)
                label_words_single_group = []
            else:
                label_words_single_group.append(line)
                
        if len(label_words_single_group) > 0:
            label_words_all.append(label_words_single_group)
            
    label_words = label_words_all[0]
    label_words = [label_words_per_label.strip().split(",") 
                  for label_words_per_label in label_words]
    return label_words

def process_dataset(input_path, output_path, dataset_name):
    """Process a single dataset"""
    try:
        num_labels = process_label_words(input_path, output_path)
        print(f"Processed {dataset_name}: {num_labels} labels")
        return num_labels
    except FileNotFoundError:
        print(f"Error: Could not find files for {dataset_name}")
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        return None

def main():
    args = parse_args()
    
    dataset_configs = {
        's2orc': {
            'input': args.s2orc_input,
            'output': args.s2orc_output,
            'name': 'S2ORC'
        },
        'arxiv': {
            'input': args.arxiv_input,
            'output': args.arxiv_output,
            'name': 'arXiv'
        },
        'sdpra': {
            'input': args.sdpra_input,
            'output': args.sdpra_output,
            'name': 'SDPRA'
        }
        # Add more datasets here
    }
    
    results = {}
    
    if args.dataset == 'all':
        for dataset_key, config in dataset_configs.items():
            result = process_dataset(config['input'], config['output'], config['name'])
            if result is not None:
                results[dataset_key] = result
    else:
        config = dataset_configs[args.dataset]
        result = process_dataset(config['input'], config['output'], config['name'])
        if result is not None:
            results[args.dataset] = result
    
    print("\nProcessing Summary:")
    for dataset_name, num_labels in results.items():
        print(f"{dataset_name}: {num_labels} labels processed")

if __name__ == "__main__":
    main()
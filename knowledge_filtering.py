import numpy as np
import pandas as pd
import random
import argparse
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import os
import openprompt
import torch
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AdamW, get_linear_schedule_with_warmup
from sentence_transformers import (
    SentenceTransformer, 
    InputExample, 
    losses, 
    util, 
    evaluation, 
    SentencesDataset, 
    models
)
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    classification_report, confusion_matrix
)

def parse_args():
    parser = argparse.ArgumentParser(description='Process and filter label words for multiple datasets')
    
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='Device to run models on')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Which dataset to process (s2orc, arxiv, sdpra, all, or custom)')
    
    parser.add_argument('--bi_encoder_path', type=str, required=True,
                      help='Path to bi-encoder NLI model')
    parser.add_argument('--cross_encoder_path', type=str, required=True,
                      help='Path to cross-encoder model')

    for dataset in ['s2orc', 'arxiv', 'sdpra', 'custom']:
        parser.add_argument(f'--{dataset}_input', type=str,
                          help=f'Path to {dataset.upper()} input JSON')
        parser.add_argument(f'--{dataset}_output_words', type=str,
                          help=f'Path to {dataset.upper()} output filtered words')
        parser.add_argument(f'--{dataset}_output_scores', type=str,
                          help=f'Path to {dataset.upper()} output scores')
    
    # Filtering parameters
    parser.add_argument('--ce_threshold', type=float, default=0.9,
                      help='Cross-encoder threshold')
    parser.add_argument('--semantic_threshold', type=float, default=0.5,
                      help='Semantic search threshold')
    
    return parser.parse_args()

def process_initial_label_words(input_json_path, template="The field of this study is related to "):
    """Process initial label words from JSON file"""
    wrapped_label_sentence = {}
    wrapped_class_label = []
    
    with open(input_json_path, 'r') as file:
        data = json.load(file)
        for key, value in data.items():
            class_text = template + key
            wrapped_class_label.append(class_text)
            wrapped_label_sentence[class_text] = [template + word for word in value]
                
    return wrapped_label_sentence, wrapped_class_label

def filter_label_words(wrapped_label_sentence, bi_encoder, cross_encoder, device,
                      ce_threshold=0.9, semantic_search_threshold=0.5):
    """Filter label words using NLI model"""
    pred_labels = {}
    pred_semantic_scores = {}
    satisfied_items = {}
    
    for class_label, wrapped_words in wrapped_label_sentence.items():
        top_k = len(wrapped_words)
        satisfied_items[class_label] = []
        pred_labels[class_label] = []
        pred_semantic_scores[class_label] = []
        
        corpus_embeddings = bi_encoder.encode(wrapped_words, convert_to_tensor=True).to(device)
        query_embedding = bi_encoder.encode(class_label, convert_to_tensor=True).to(device)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
        
        cross_inp = [[class_label, wrapped_words[hit['corpus_id']]] for hit in hits]
        cross_scores = cross_encoder.predict(cross_inp)
        
        for idx, score in enumerate(cross_scores):
            hits[idx]['cross-score'] = score
        
        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        
        for hit in hits:
            if hit['cross-score'] <= ce_threshold and hit['score'] <= semantic_search_threshold:
                label = 'contrasting'
            elif hit['cross-score'] > ce_threshold and hit['score'] > semantic_search_threshold:
                label = 'entailment'
                satisfied_items[class_label].append(wrapped_words[hit['corpus_id']])
                hit['score'] = min(hit['score'], 1.0)
                pred_semantic_scores[class_label].append(hit['score'])
            pred_labels[class_label].append(label)
            
    return pred_labels, pred_semantic_scores, satisfied_items

def save_filtered_words(satisfied_items, pred_semantic_scores, words_path, scores_path):
    """Save filtered words and their scores to files"""
    label_words = []
    class_name = []
    label_words_scores = []
    
    Path(words_path).parent.mkdir(parents=True, exist_ok=True)
    Path(scores_path).parent.mkdir(parents=True, exist_ok=True)
    
    for (class_sentence, filtered_sentences), (_, scores) in zip(
        satisfied_items.items(), pred_semantic_scores.items()
    ):
        class_indices = class_sentence.find("study is related to")
        if class_indices != -1:
            class_subject = class_sentence[class_indices + len("study is related to"):].strip(". ,")
            class_name.append(class_subject.lower())
            
        label_word_temp = []
        valid_scores = []
        for i, sentence in enumerate(filtered_sentences):
            index_related_to = sentence.find("study is related to")
            if index_related_to != -1:
                subject = sentence[index_related_to + len("study is related to"):].strip(". ,")
                if len(subject) >= 3:
                    label_word_temp.append(subject)
                    valid_scores.append(scores[i])
                    
        label_words.append(label_word_temp)
        label_words_scores.append(valid_scores)
    
    # Save filtered words
    with open(words_path, "w") as file:
        for class_label, words in zip(class_name, label_words):
            if not words:
                file.write(f"{class_label}\n")
            else:
                words = [word.replace(',', ' ') for word in words]
                file.write(f"{class_label},{','.join(words)}\n")
                
    # Save scores
    with open(scores_path, "w") as file:
        for scores in label_words_scores:
            if not scores:
                file.write("1\n")
            else:
                file.write(f"1,{','.join(map(str, scores))}\n")

def process_dataset(config, bi_encoder, cross_encoder, ce_threshold, semantic_threshold, device):
    """Process a single dataset"""
    try:
        wrapped_label_sentence, _ = process_initial_label_words(config['input'])
        pred_labels, pred_semantic_scores, satisfied_items = filter_label_words(
            wrapped_label_sentence, 
            bi_encoder, 
            cross_encoder,
            device,
            ce_threshold,
            semantic_threshold
        )
        
        save_filtered_words(
            satisfied_items,
            pred_semantic_scores,
            config['output_words'],
            config['output_scores']
        )
        
        return len(wrapped_label_sentence)
    except FileNotFoundError as e:
        print(f"Error processing {config['name']}: {str(e)}")
        return None

def main():
    args = parse_args()
    
    bi_encoder = SentenceTransformer(args.bi_encoder_path)
    cross_encoder = CrossEncoder(args.cross_encoder_path)
    
    dataset_configs = {
        's2orc': {
            'input': args.s2orc_input,
            'output_words': args.s2orc_output_words,
            'output_scores': args.s2orc_output_scores,
            'name': 'S2ORC'
        },
        'arxiv': {
            'input': args.arxiv_input,
            'output_words': args.arxiv_output_words,
            'output_scores': args.arxiv_output_scores,
            'name': 'arXiv'
        },
        'sdpra': {
            'input': args.sdpra_input,
            'output_words': args.sdpra_output_words,
            'output_scores': args.sdpra_output_scores,
            'name': 'SDPRA'
        },
        # Add custom dataset configurations here
        'custom': {
            'input': args.custom_input,
            'output_words': args.custom_output_words,
            'output_scores': args.custom_output_scores,
            'name': 'custom'
        }
    }
    
    results = {}
    
    if args.dataset in ['s2orc', 'arxiv', 'sdpra']:
        config = dataset_configs[args.dataset]
        print(f"\nProcessing {config['name']}...")
        result = process_dataset(
            config, 
            bi_encoder, 
            cross_encoder,
            args.ce_threshold,
            args.semantic_threshold,
            args.device
        )
        if result is not None:
            results[args.dataset] = result
    else:
        config = dataset_configs[args.dataset]
        print(f"\nProcessing {config['name']}...")
        result = process_dataset(
            config, 
            bi_encoder, 
            cross_encoder,
            args.ce_threshold,
            args.semantic_threshold,
            args.device
        )
        if result is not None:
            results[args.dataset] = result
    
    print("\nProcessing Summary:")
    for dataset_name, num_labels in results.items():
        print(f"{dataset_name}: {num_labels} labels processed")

if __name__ == "__main__":
    main()

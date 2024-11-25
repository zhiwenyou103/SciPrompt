import os
import argparse
import torch
import csv
import copy
from tqdm import tqdm
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup, 
    BertTokenizer
)
from contextualize_calibration import calibrate
from openprompt.plms import load_plm
from openprompt.prompts import WeightedVerbalizer, ManualTemplate, SoftVerbalizer
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.utils.reproduciblity import set_seed
import json


def parse_args():
    parser = argparse.ArgumentParser(description='SDPRA KAPT Training')
    parser.add_argument('--seed', type=int, default=144, help='Random seed')
    parser.add_argument('--shots', type=int, default=1, help='Number of shots')
    parser.add_argument('--calibration', type=bool, default=True, help='Whether to use calibration')
    parser.add_argument('--max_seq_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=5, help='Maximum epochs')
    parser.add_argument('--cuda_device', type=int, default=3, help='CUDA device index')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--soft_verbalizer', type=bool, default=False, help='Whether to use soft verbalizer')
                      help='Path to label mappings config file')
    parser.add_argument('--zero_shot', type=str, default='no', 
                       choices=['yes', 'no'], 
                       help='Whether to run in zero-shot mode')
    parser.add_argument('--data_dir', type=str, 
                       default="/path/to/data",
                       help='Data directory')
    parser.add_argument('--verbalizer_path', type=str,
                       default='path/to/filtered_knowledgable_verbalizer.txt',
                       help='Path to verbalizer file')
    parser.add_argument('--semantic_score_path', type=str,
                       default='path/to/filtered_knowledgable_verbalizer_semantic_search_scores.txt',
                       help='Path to semantic scores file')
    parser.add_argument('--doc_id_path', type=str, 
                       default="path/to/label_dict.txt",
                       help='Path to label_dict.txt file')
    parser.add_argument('--config_path', type=str,
                       default='label_mappings/SDORA_label_mappings.json',
                       help='Path to label configuration file')
    return parser.parse_args()

def create_prompt_dataloader(dataset, template, tokenizer, WrapperClass, args, shuffle=False):
    """Helper function to create prompt dataloader"""
    return PromptDataLoader(
        dataset=dataset,
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=args.max_seq_length,
        decoder_max_length=16,
        batch_size=args.batch_size,
        shuffle=shuffle,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="tail"
    )

def setup_model(args, tokenizer, plm, WrapperClass, class_labels):
    """Setup the prompt model with template and verbalizer"""
    template_text = 'The field of this study is related to: {"mask"}. {"placeholder":"text_a"}'
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

    with open(args.semantic_score_path, 'r') as f:
        lines = f.readlines()
        label_words_all_score = []
        label_score_single_group = []
        for line in lines:
            line = line.strip().strip(" ")
            if line == "":
                if len(label_score_single_group) > 0:
                    label_words_all_score.append(label_score_single_group)
                label_score_single_group = []
            else:
                label_score_single_group.append(line)
        if len(label_score_single_group) > 0:
            label_words_all_score.append(label_score_single_group)

        label_words_scores = label_words_all_score[0]
        label_words_scores = [label_words_per_label.strip().split(",") 
                            for label_words_per_label in label_words_scores]
        
    if args.soft_verbalizer:
        myverbalizer = SoftVerbalizer(
            tokenizer,
            model=plm,
            classes=CLASS_LABELS,
            label_words_scores=label_words_scores,
            multi_token_handler="mean"
        ).from_file(args.verbalizer_path)
    else:
        myverbalizer = WeightedVerbalizer(
            tokenizer,
            classes=class_labels,
            label_words_scores=label_words_scores,
            multi_token_handler="mean"
        ).from_file(args.verbalizer_path)

    prompt_model = PromptForClassification(
        plm=plm,
        template=mytemplate,
        verbalizer=myverbalizer,
        freeze_plm=False,
        plm_eval_mode=False
    )

    return prompt_model, mytemplate, myverbalizer


def run_few_shot_training(args, prompt_model, dataset, mytemplate, myverbalizer,
                         tokenizer, WrapperClass, device):
    """Run few-shot training with validation and testing"""
    if args.calibration:
        support_dataloader = PromptDataLoader(
            dataset=dataset["train"],
            template=mytemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=args.max_seq_length,
            decoder_max_length=16,
            batch_size=args.batch_size,
            shuffle=False,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="tail"
        )
        
        org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) 
                              for i in range(len(CLASS_LABELS))]
        print("Original label words num:", org_label_words_num)
        
        from contextualize_calibration import calibrate
        cc_logits = calibrate(prompt_model, support_dataloader)
        print("Calibration logits:", cc_logits)
        
        myverbalizer.register_calibrate_logits(cc_logits.mean(dim=0))
        new_label_words_num = [len(myverbalizer.label_words[i]) 
                              for i in range(len(CLASS_LABELS))]
        print("After filtering, label words per class:", new_label_words_num)

    sampler = FewShotSampler(
        num_examples_per_label=args.shots,
        also_sample_dev=True,
        num_examples_per_label_dev=args.shots
    )
    dataset['train'], dataset['validation'] = sampler(dataset['train'], seed=args.seed)

    train_dataloader = PromptDataLoader(
        dataset=dataset["train"],
        template=mytemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=args.max_seq_length,
        decoder_max_length=16,
        batch_size=args.batch_size,
        shuffle=True,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="tail"
    )

    validation_dataloader = PromptDataLoader(
        dataset=dataset["validation"],
        template=mytemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=args.max_seq_length,
        decoder_max_length=16,
        batch_size=args.batch_size,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="tail"
    )

    test_dataloader = PromptDataLoader(
        dataset=dataset["test"],
        template=mytemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=args.max_seq_length,
        decoder_max_length=16,
        batch_size=args.batch_size,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="tail"
    )

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in prompt_model.plm.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in prompt_model.plm.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=0.0)

    num_training_steps = len(train_dataloader) * args.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    best_val_acc = 0
    best_model_state = None
    loss_func = torch.nn.CrossEntropyLoss()

    print(f"Starting training with {args.shots} shots per class...")
    for epoch in range(args.max_epochs):
        tot_loss = 0
        prompt_model.train()
        with tqdm(train_dataloader, desc=f"Epoch {epoch}") as pbar:
            for step, inputs in enumerate(pbar):
                inputs = inputs.to(device)
                logits = prompt_model(inputs)
                labels = inputs['label']
                loss = loss_func(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                if optimizer2 is not None:
                    optimizer2.step()
                    optimizer2.zero_grad()
                
                tot_loss += loss.item()
                pbar.set_postfix({'loss': tot_loss / (step + 1)})

        val_acc = evaluate(prompt_model, validation_dataloader, device, desc='Validation')
        print(f"Epoch {epoch}, Validation Accuracy: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(prompt_model.state_dict())
            print(f"New best validation accuracy: {best_val_acc:.4f}")

    if best_model_state is not None:
        prompt_model.load_state_dict(best_model_state)
    
    test_acc = evaluate(prompt_model, test_dataloader, device, desc="Test")
    print(f"\nFinal Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return best_val_acc, test_acc

def run_zero_shot(args, prompt_model, dataset, mytemplate, myverbalizer, tokenizer, WrapperClass, device):
    """Run zero-shot evaluation"""
    if args.calibration:
        support_dataloader = PromptDataLoader(
            dataset=dataset["train"],
            template=mytemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=args.max_seq_length,
            decoder_max_length=16,
            batch_size=args.batch_size,
            shuffle=False,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="tail"
        )

        org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) 
                              for i in range(len(CLASS_LABELS))]
        
        cc_logits = calibrate(prompt_model, support_dataloader)
        print("Calibration logits:", cc_logits)
        print("Original label words num:", org_label_words_num)

        myverbalizer.register_calibrate_logits(cc_logits.mean(dim=0))
        new_label_words_num = [len(myverbalizer.label_words[i]) 
                              for i in range(len(CLASS_LABELS))]
        print("After filtering, label words per class:", new_label_words_num)

    test_dataloader = PromptDataLoader(
        dataset=dataset["test"],
        template=mytemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=args.max_seq_length,
        decoder_max_length=16,
        batch_size=args.batch_size,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="tail"
    )
    
    test_acc = evaluate(prompt_model, test_dataloader, device, "Test")
    print(f"Zero-shot Accuracy: {test_acc:.4f}")
    return test_acc


def load_label_mappings(config_path):
    """Load label mappings from config file."""
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")
    
    return (
        config['label_name_mapping'],
        config['class_labels'],
        config['label_dict']
    )

def get_examples(data_dir, type, tokenizer, label_dict):
    """Load examples from CSV file"""
    path = os.path.join(data_dir, f"{type}.csv")
    examples = []
    
    with open(path, encoding='utf8') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            body, label = row
            label = label_dict[label]
            inputs = tokenizer(body, return_tensors="pt")
            if len(inputs["input_ids"][0]) < 30:
                continue
            example = InputExample(guid=str(idx), text_a=body, label=int(label))
            examples.append(example)
    return examples

def evaluate(prompt_model, dataloader, device, desc="Eval"):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc=desc):
            inputs = inputs.to(device)
            logits = prompt_model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    return acc

def main():
    args = parse_args()
    set_seed(args.seed)
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    label_name_mapping, class_labels, label_dict = load_label_mappings(args.config_path)
    plm, tokenizer, model_config, WrapperClass = load_plm(
        'bert', 
        'allenai/scibert_scivocab_uncased'
    )
    dataset = {
        'train': get_examples(args.data_dir, "train_output", tokenizer, label_dict),
        'test': get_examples(args.data_dir, "test_output", tokenizer, label_dict)
    }
    prompt_model, mytemplate, myverbalizer = setup_model(
        args, tokenizer, plm, WrapperClass, class_labels
    )
    prompt_model = prompt_model.to(device)

    if args.zero_shot == 'yes':
        test_acc = run_zero_shot(
            args, prompt_model, dataset, mytemplate, myverbalizer,
            tokenizer, WrapperClass, device, class_labels
        )
    else:
        best_val_acc, test_acc = run_few_shot_training(
            args, prompt_model, dataset, mytemplate, myverbalizer,
            tokenizer, WrapperClass, device, class_labels
        )

if __name__ == "__main__":
    main()
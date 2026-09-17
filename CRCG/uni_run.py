# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Code Review Comment Generation using UniXcoder
"""

from __future__ import absolute_import
import os
import torch
import json
import random
import logging
import argparse
import numpy as np
from bleu import _bleu
from models import Seq2Seq
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset

from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""
    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=" ".join(js['code_change'].split()),
                    target=" ".join(js["review_comment"].split()),
                )
            )
    return examples


class InputFeatures(object):
    """A single training/test features for an example."""
    def __init__(self, example_id, source_ids, target_ids):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    """Convert examples to features for model input."""
    features = []
    for example_index, example in enumerate(examples):
        # Tokenize source sequence
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-5]
        source_tokens = [tokenizer.cls_token, "<encoder-decoder>", tokenizer.sep_token] + source_tokens + ["<mask0>", tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        # Tokenize target sequence
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        
        target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
            )
        )
    return features


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def inference_and_save(args, model, tokenizer, filename, criteria, split_tag):
    """
    Run inference, save predictions to JSON, and return BLEU score.
    
    Args:
        args: Command line arguments
        model: The model for inference
        tokenizer: Tokenizer for encoding/decoding
        filename: Path to the data file
        criteria: Identifier for the checkpoint (e.g., 'best-bleu', 'last')
        split_tag: Dataset split identifier ('dev' or 'test')
    
    Returns:
        bleu_score: BLEU score on the dataset
    """
    examples = read_examples(filename)
    features = convert_examples_to_features(examples, tokenizer, args, stage='test')
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    data = TensorDataset(all_source_ids)
    
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

    model.eval()
    preds = []
    logger.info(f"Running generation on {split_tag} set (checkpoint: {criteria})")
    
    for batch in tqdm(dataloader, desc=f"Generating {split_tag}"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids = batch[0]
        with torch.no_grad():
            output_sequences = model(source_ids=source_ids)
            
            for output in output_sequences:
                t = output[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                preds.append(text)
    
    # Save predictions to JSON
    pred_dir = os.path.join(args.output_dir, "prediction")
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    
    output_file = os.path.join(pred_dir, f"{criteria}_{split_tag}.json")
    logger.info(f"Saving predictions to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example, pred in zip(examples, preds):
            row = {
                "source": example.source,
                "target": example.target,
                "output": pred
            }
            f.write(json.dumps(row) + "\n")
            
    # Calculate BLEU score
    temp_gold = os.path.join(args.output_dir, f"temp_{split_tag}.gold")
    temp_out = os.path.join(args.output_dir, f"temp_{split_tag}.output")
    
    with open(temp_out, 'w', encoding='utf-8') as f_out, open(temp_gold, 'w', encoding='utf-8') as f_gold:
        for example, pred in zip(examples, preds):
            f_out.write(pred + '\n')
            f_gold.write(example.target + '\n')
            
    bleu_score = _bleu(temp_gold, temp_out)
    
    # Clean up temporary files
    os.remove(temp_gold)
    os.remove(temp_out)
    
    return bleu_score


def main():
    parser = argparse.ArgumentParser()

    # Required parameters  
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")   
  
    # Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="Beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--patience", default=3, type=int,
                        help="Early stopping patience.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)
    
    set_seed(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.is_decoder = True
    encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config) 

    model = Seq2Seq(encoder=encoder, decoder=encoder, config=config,
                  beam_size=args.beam_size, max_length=args.max_target_length,
                  sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0], eos_id=tokenizer.sep_token_id)
    
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)   
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Training phase
    if args.do_train:
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long) 
        train_data = TensorDataset(all_source_ids, all_target_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        # Setup optimizer and scheduler
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=int(len(train_dataloader)*args.num_train_epochs*0.1),
                                                    num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Num epochs = %d", args.num_train_epochs)
        logger.info("  Batch size = %d", args.train_batch_size)
        
        best_bleu = 0.0
        patience_counter = 0
        
        for epoch in range(args.num_train_epochs):
            model.train()
            total_loss = 0.0
            
            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
                batch = tuple(t.to(device) for t in batch)
                source_ids, target_ids = batch
                loss, _, _ = model(source_ids=source_ids, target_ids=target_ids)

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                total_loss += loss.item()
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch} - Average training loss: {avg_train_loss:.4f}")

            # Validation phase
            if args.do_eval:
                logger.info("Evaluating on development set")
                dev_bleu = inference_and_save(args, model, tokenizer, args.dev_filename, f"epoch_{epoch}", "dev")
                logger.info(f"Epoch {epoch} - Dev BLEU: {dev_bleu:.2f}")

                # Save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(last_output_dir, "pytorch_model.bin"))
                logger.info("Saved last checkpoint")

                # Save best checkpoint based on dev BLEU
                if dev_bleu > best_bleu:
                    logger.info(f"New best BLEU: {dev_bleu:.2f} (previous: {best_bleu:.2f})")
                    best_bleu = dev_bleu
                    patience_counter = 0
                    
                    best_output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(best_output_dir):
                        os.makedirs(best_output_dir)
                    
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), os.path.join(best_output_dir, "pytorch_model.bin"))
                    logger.info("Saved best BLEU checkpoint")
                else:
                    patience_counter += 1
                    logger.info(f"BLEU did not improve. Patience: {patience_counter}/{args.patience}")
                    
                    if patience_counter >= args.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

        logger.info("Training completed")
        logger.info(f"Best dev BLEU: {best_bleu:.2f}")

    # Testing phase
    if args.do_test:
        logger.info("***** Running testing *****")
        
        checkpoints_to_test = ['best-bleu', 'last']
        
        for criteria in checkpoints_to_test:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{criteria}', 'pytorch_model.bin')
            
            if not os.path.exists(checkpoint_path):
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                continue
            
            logger.info(f"Testing checkpoint: {criteria}")
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(torch.load(checkpoint_path))
            
            test_bleu = inference_and_save(args, model, tokenizer, args.test_filename, criteria, "test")
            logger.info(f"Test BLEU ({criteria}): {test_bleu:.2f}")

    logger.info("Process completed")


if __name__ == "__main__":
    main()
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import json
import random
import numpy as np
import torch
import multiprocessing
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

from uni_model import Model

# Configure logging settings
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for an example."""
    def __init__(self, example_id, source_ids, label):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label

def convert_examples_to_features(item):
    example, example_index, tokenizer, args = item
    source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.block_size, padding='max_length', truncation=True)
    return InputFeatures(example_index, code, example.target)

class Example(object):
    """A single training/test example."""
    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, 'r') as f:
        for idx, line in enumerate(f):
            line = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=line['code_change'],
                    target=line['label'],
                )
            )
    return examples

def load_and_cache_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = os.path.join(args.output_dir, "data", split_tag)
    if not os.path.exists(os.path.join(args.output_dir, "data")):
        os.makedirs(os.path.join(args.output_dir, "data"))
        
    examples = read_examples(filename)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn, weights_only=False)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        assert len(all_source_ids) == len(all_labels)
        data = TensorDataset(all_source_ids, all_labels)

        torch.save(data, cache_fn)
    return examples, data

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, train_dataset, model, tokenizer, pool):
    """Train the model with validation-based checkpoint saving and early stopping."""
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    args.max_steps = args.num_train_epochs * len(train_dataloader)
    args.save_steps = args.max_steps // 10

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    # Training information
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", args.max_steps)

    losses = []
    model.zero_grad()

    # Initialize tracking variables for model selection
    best_f1 = 0
    best_acc = 0
    not_improve_cnt = 0
    is_early_stop = False
 
    # ========== Training Phase ==========
    for idx in range(args.num_train_epochs):
        if is_early_stop:
            break
            
        for batch in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Training Epoch {idx}"):
            inputs = batch[0].to(args.device)        
            labels = batch[1].to(args.device) 
            model.train()
            loss, logits = model(inputs, labels)
            
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  
            
        # ========== Validation Phase: Evaluate on dev set after each epoch ==========
        logger.info(f"End of Epoch {idx}, running validation on development set...")

        # Evaluate on validation set only (for model selection and early stopping)
        results_dev, _ = evaluate(args, model, tokenizer, "valid", args.eval_data_file, pool, epoch_num=idx)
        
        current_acc = results_dev['eval_acc']
        current_f1 = results_dev['eval_f1']

        # Save last checkpoint
        last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
        if not os.path.exists(last_output_dir):
            os.makedirs(last_output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(last_output_dir, 'pytorch_model.bin'))
        logger.info("Save the last model checkpoint to %s", last_output_dir)

        # Save best model based on validation accuracy
        if current_acc > best_acc:
            not_improve_cnt = 0
            best_acc = current_acc
            output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
            logger.info("  Best validation accuracy: %.4f", best_acc)
            logger.info("  Save the best accuracy model to %s", output_dir)
            
        # Save best model based on validation F1 score
        elif current_f1 > best_f1:
            not_improve_cnt = 0
            best_f1 = current_f1
            output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
            logger.info("  Best validation F1 score: %.4f", best_f1)
            logger.info("  Save the best F1 model to %s", output_dir)
        else:
            not_improve_cnt += 1
            logger.info("  Validation metrics do not improve for %d epochs", not_improve_cnt)
            
            # Early stopping: stop when validation metrics do not improve for patience epochs
            if not_improve_cnt >= args.patience:
                logger.info("Early stopping triggered after %d epochs without improvement", not_improve_cnt)
                is_early_stop = True

    return best_acc, best_f1

def evaluate(args, model, tokenizer, split_tag, data_file, pool, epoch_num=None):
    """Evaluate the model and save predictions."""
    
    # Load data (examples contain original source text)
    examples, eval_dataset = load_and_cache_data(args, data_file, pool, tokenizer, split_tag, is_sample=False)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Evaluation information
    logger.info(f"***** Running evaluation on {split_tag} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model.eval()
    logits = []  
    y_trues = []
    
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        inputs = batch[0].to(args.device)        
        labels = batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss, prob = model(inputs, labels)
            logits.append(prob.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
            
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    
    # Prediction results
    y_preds = logits[:, 1] > 0.5
    
    # Calculate evaluation metrics
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)   
    f1 = f1_score(y_trues, y_preds)             
    result = {
        "eval_acc": round(acc, 4),
        "eval_precision": round(precision, 4),
        "eval_recall": round(recall, 4),
        "eval_f1": round(f1, 4),   
    }
    
    logger.info("  Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f", acc, precision, recall, f1)

    # Save prediction results to JSON file
    if epoch_num is not None:
        # Create prediction directory
        pred_dir = os.path.join(args.output_dir, "prediction")
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        # Filename format: {epoch_num}_{dev/test}.json
        file_tag = "dev" if split_tag == "valid" else "test"
        output_filename = os.path.join(pred_dir, f"{epoch_num}_{file_tag}.json")
        
        logger.info(f"Saving predictions to {output_filename}")
        
        with open(output_filename, 'w') as f:
            for example, pred in zip(examples, y_preds):
                row = {
                    "source": example.source,
                    "target": example.target,
                    "output": str(pred)
                }
                f.write(json.dumps(row) + "\n")

    return result, y_preds

def test(args, model, tokenizer, pool):
    """Test saved checkpoints after training completion."""
    logger.info("***** Running testing on saved checkpoints *****")
    
    # Test three saved checkpoints: best validation accuracy, best validation F1, and last epoch
    for criteria in ['best-acc', 'best-f1', 'last']:
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{criteria}/pytorch_model.bin')
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} not found, skipping...")
            continue
            
        logger.info(f"Loading model from {checkpoint_path}")
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(checkpoint_path))
        
        # Evaluate on test set
        result, y_preds = evaluate(args, model, tokenizer, "test", args.test_data_file, pool, epoch_num=f"final_{criteria}")
        
        logger.info(f"[Test on {criteria}] Accuracy: {result['eval_acc']:.4f}, F1: {result['eval_f1']:.4f}, "
                   f"Precision: {result['eval_precision']:.4f}, Recall: {result['eval_recall']:.4f}")
        logger.info("  " + "*" * 20)
                                                
def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Data parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a jsonl file).")    
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file for validation (a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file for final evaluation (a jsonl file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    # Model parameters
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")    
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    
    # Optimizer parameters
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization.")
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=5,
                        help="Number of epochs to wait before early stopping.")
    
    # System parameters
    parser.add_argument('--cpu_cont', type=int, default=4,
                        help="Thread count for multiprocessing.")
    
    args = parser.parse_args()
    
    # Initialize multiprocessing pool
    pool = multiprocessing.Pool(args.cpu_cont)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("Device: %s, n_gpu: %s", device, args.n_gpu)
    
    # Set seed for reproducibility
    set_seed(args.seed)

    # Build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 

    model = Model(model, config, tokenizer, args)
    logger.info("Training/evaluation parameters %s", args)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)      
    
    # ========== Training Phase ==========
    if args.do_train:
        _, train_dataset = load_and_cache_data(args, args.train_data_file, pool, tokenizer, "train", is_sample=False)
        train(args, train_dataset, model, tokenizer, pool)
    
    # ========== Testing Phase: Evaluate saved checkpoints after training completion ==========
    if args.do_test:
        test(args, model, tokenizer, pool)

if __name__ == "__main__":
    main()
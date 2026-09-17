from __future__ import absolute_import
import os
import logging
import argparse
import math
import numpy as np
from io import open
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import multiprocessing
import time

from models import DefectModel
from configs import add_args, set_seed
from utils import get_filenames, load_and_cache_defect_data
from models import get_model_size
import json

# Define model classes for different model types
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)}

# Get CPU count for multiprocessing
cpu_cont = multiprocessing.cpu_count()

# Configure logging settings for training monitoring
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args, model, eval_examples, eval_data, epoch, write_to_pred=False):
    """Evaluate model performance on evaluation dataset"""
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Start evaluation process
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Num batches = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 1] > 0.5
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps

    result = {
        "acc": eval_acc,
        "precision": round(precision_score(labels, preds), 4),
        "recall": round(recall_score(labels, preds), 4),
        "f1": round(f1_score(labels, preds), 4)
    }

    logger.info("Predicted labels: %s", preds)
    logger.info("True labels: %s", labels)

    logger.info("***** Evaluation results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    # Save predictions if requested
    if write_to_pred:
        with open(os.path.join(args.output_dir, "{}-predictions.txt".format(epoch)), 'w') as f:
            for example, pred in zip(eval_examples, preds):
                if pred:
                    f.write(str(example.idx) + '\t1\n')
                else:
                    f.write(str(example.idx) + '\t0\n')

    return result, preds


def main():
    parser = argparse.ArgumentParser()
    t0 = time.time()
    args = add_args(parser)
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    set_seed(args)

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    model = DefectModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    pool = multiprocessing.Pool(cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.n_gpu > 1:
            # multi-gpu training
            model = torch.nn.DataParallel(model)
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        eval_dev_writer = open(os.path.join(args.output_dir, 'eval_dev_per_epoch.txt'), 'a+')

        # Prepare training data loader
        train_examples, train_data = load_and_cache_defect_data(args, args.train_filename, pool, tokenizer, 'train',
                                                                is_sample=False)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        save_steps = max(len(train_dataloader), 1)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * args.warmup_steps
        else:
            warmup_steps = int(args.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}

        global_step, best_acc, best_f1  = 0, -1, -1
        not_acc_inc_cnt = 0
        is_early_stop = False
        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                source_ids, labels = batch

                loss, logits = model(source_ids, labels)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if args.do_eval:
                # Evaluate on development set only during training (for model selection and early stopping)
                logger.info("--- Evaluating on Development Set (for model selection) ---")
                if 'dev' not in dev_dataset:
                    dev_examples, dev_data = load_and_cache_defect_data(args, args.dev_filename, pool, tokenizer, 'valid', is_sample=False)
                    dev_dataset['dev'] = (dev_examples, dev_data)
                dev_examples, dev_data = dev_dataset['dev']

                result, _ = evaluate(args, model, dev_examples, dev_data, cur_epoch)
                eval_acc = result['acc']
                eval_f1 = result['f1']

                # Write development set results to dedicated file and main log
                dev_result_str = "[Epoch {} Dev] Acc: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}\n".format(
                    cur_epoch, eval_acc, eval_f1, result['precision'], result['recall']
                )
                eval_dev_writer.write(dev_result_str)
                eval_dev_writer.flush()
                fa.write(dev_result_str)  # Keep in main log as well
                logger.info(f"Dev results: {dev_result_str.strip()}")

                # Save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)

                if args.save_last_checkpoints:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                # Save best model based on validation accuracy
                improved = eval_acc > best_acc or eval_f1 > best_f1
                if eval_acc > best_acc:
                    not_acc_inc_cnt = 0
                    logger.info("  Best validation accuracy: %s", round(eval_acc, 4))
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best validation accuracy changed to %.4f\n" % (cur_epoch, round(eval_acc, 4)))
                    best_acc = eval_acc
                    # Save best checkpoint for best validation accuracy
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the best validation accuracy model into %s", output_model_file)

                # Save best model based on validation F1 score
                if eval_f1 > best_f1:
                    not_acc_inc_cnt = 0
                    logger.info("  Best validation F1 score: %s", round(eval_f1, 4))
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best validation F1 score changed to %.4f\n" % (cur_epoch, round(eval_f1, 4)))
                    best_f1 = eval_f1
                    # Save best checkpoint for best validation F1 score
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the best validation F1 model into %s", output_model_file)
                if not improved:
                    not_acc_inc_cnt += 1
                    logger.info("Validation accuracy and F1 did not improve for %d epochs", not_acc_inc_cnt)
                    if not_acc_inc_cnt > args.patience:
                        logger.info("Early stopping after %d epochs without improvement", not_acc_inc_cnt)
                        fa.write("[%d] Early stopping as not_acc_inc_cnt=%d\n" % (cur_epoch, not_acc_inc_cnt))
                        is_early_stop = True
                        break

                output_dir = args.output_dir
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                model.train()
            if is_early_stop:
                break

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()

        eval_dev_writer.close()

    # ========== Testing Phase: Evaluate saved checkpoints after training completion ==========
    if args.do_test:
        test_dataset = {}
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        # Test three saved checkpoints: best validation accuracy, best validation F1, and last epoch
        for criteria in ['best-acc','best-f1','last']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            if not os.path.isfile(file):
                logger.warning("Checkpoint not found, skipping: %s", file)
                continue
            logger.info("Reload model from {}".format(file))
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(torch.load(file, map_location=device))
                
            # Load test data only once for testing phase
            if 'test' not in test_dataset:
                test_examples, test_data = load_and_cache_defect_data(args, args.test_filename, pool, tokenizer, 'test', is_sample=False)
                test_dataset['test'] = (test_examples, test_data)
            test_examples, test_data = test_dataset['test']

            result, y_preds = evaluate(args, model, test_examples, test_data, criteria, write_to_pred=True)
            logger.info("  Test accuracy = %.4f", result['acc'])
            logger.info("  " + "*" * 20)

            # Build detailed result string
            test_result_str = "[Test on {}] Accuracy: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(
                criteria, result['acc'], result['f1'], result['precision'], result['recall']
            )
            logger.info(test_result_str)
            logger.info("  " + "*" * 20)
            
            # Write detailed results to summary log
            fa.write(test_result_str + '\n')

            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write("[%s] accuracy: %.4f\n\n" % (criteria, result['acc']))
                    f.write("[%s] F1 score: %.4f\n\n" % (criteria, result['f1']))

            # Save prediction results
            with open(args.test_filename,'r') as f1,open(os.path.join(args.output_dir, "prediction-{}.jsonl".format(criteria)) ,'w') as f:
                lines = []
                for line in f1:
                    line = json.loads(line)
                    lines.append(line)

                for new_line, y_pred in zip(lines, y_preds):
                    new_line['prediction'] = str(y_pred)
                    f.write(json.dumps(new_line))
                    f.write('\n')

    fa.close()


if __name__ == "__main__":
    main()

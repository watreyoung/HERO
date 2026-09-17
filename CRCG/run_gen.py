import os
import logging
import argparse
import json
import numpy as np
from tqdm import tqdm
import multiprocessing
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
import sacrebleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist

# Import ROUGE evaluation library
from rouge_score import rouge_scorer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    """Evaluate perplexity on evaluation dataset"""
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    
    logger.info("***** Running PPL evaluation *****")
    logger.info("Num examples = %d", len(eval_examples))
    logger.info("Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval PPL"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss

        eval_loss += loss.item()
        batch_num += 1
    
    if batch_num == 0:
        return 0.0
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_generation_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    """
    Evaluate generation task with EM, BLEU, ROUGE-L metrics
    """
    logger.info("Running generation evaluation on {} data".format(split_tag))
    logger.info("Num examples = %d", len(eval_examples))
    logger.info("Batch size = %d", args.eval_batch_size)
    
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval Generation"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids = batch[0]
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)
                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    # Calculate EM (Exact Match)
    em_accs = []
    for pred_nl, gold in zip(pred_nls, eval_examples):
        em_accs.append(pred_nl.strip() == gold.target.strip())
    em_score = np.mean(em_accs) * 100 if em_accs else 0.0

    # Calculate BLEU scores
    gold_targets = [gold.target.strip() for gold in eval_examples]
    
    # Corpus BLEU
    references = [[gt] for gt in gold_targets]
    corpus_bleu = round(sacrebleu.corpus_bleu(pred_nls, references).score, 2)
    
    # Sentence BLEU
    sentence_bleu_scores = []
    for pred, gold in zip(pred_nls, gold_targets):
        score = sacrebleu.sentence_bleu(pred, [gold]).score
        sentence_bleu_scores.append(score)
    avg_sent_bleu = round(np.mean(sentence_bleu_scores), 2) if sentence_bleu_scores else 0.0

    # Calculate ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_f1 = 0.0
    for pred, gold in zip(pred_nls, gold_targets):
        score = scorer.score(gold, pred.strip())
        rouge_l_f1 += score['rougeL'].fmeasure
    
    rouge_l = round((rouge_l_f1 / len(pred_nls)) * 100, 2) if pred_nls else 0.0

    result = {
        'em': em_score,
        'corpus_bleu': corpus_bleu,
        'sentence_bleu': avg_sent_bleu,
        'rouge-l': rouge_l
    }

    logger.info("Eval results:")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    pool = multiprocessing.Pool(args.cpu_cont)
    
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        # Initialize evaluation writers
        eval_dev_writer = open(os.path.join(args.output_dir, 'eval_dev.txt'), 'a+')
        eval_test_writer = open(os.path.join(args.output_dir, 'eval_test.txt'), 'a+')
        
        # Initialize TensorBoard writer if needed
        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer = SummaryWriter(args.summary_dir)

        # Load training data
        train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Setup optimizer and scheduler
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        logger.info("Running training")
        logger.info("Num examples = %d", len(train_data))
        logger.info("Batch size = %d", args.train_batch_size)
        logger.info("Num epochs = %d", args.num_train_epochs)

        # Initialize tracking variables
        dev_dataset = {}
        test_dataset = {}
        global_step = 0
        best_ppl = float('inf')
        best_em, best_bleu, best_rouge_l = 0.0, 0.0, 0.0
        not_loss_dec_cnt = 0
        not_em_inc_cnt, not_bleu_inc_cnt, not_rouge_l_inc_cnt = 0, 0, 0

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            # Training phase
            model.train()
            total_loss = 0.0
            
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                global_step += 1

            avg_train_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {cur_epoch} - Average training loss: {avg_train_loss:.4f}")

            # Validation phase
            if args.do_eval:
                logger.info("Evaluating on development set")
                
                # Load validation data
                if 'dev_loss' not in dev_dataset:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                else:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                
                # Evaluate perplexity
                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                logger.info(f"Epoch {cur_epoch} - Validation PPL: {eval_ppl}")

                # Save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), os.path.join(last_output_dir, "pytorch_model.bin"))
                    logger.info("Saved last model checkpoint")

                # Update best PPL model
                if eval_ppl < best_ppl:
                    best_ppl = eval_ppl
                    not_loss_dec_cnt = 0
                    logger.info(f"New best PPL: {best_ppl}")
                    
                    # Save best PPL model
                    best_ppl_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(best_ppl_dir):
                        os.makedirs(best_ppl_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(best_ppl_dir, "pytorch_model.bin"))
                        logger.info("Saved best PPL model")
                else:
                    not_loss_dec_cnt += 1

                # Evaluate generation metrics if enabled
                if args.do_eval_bleu:
                    if 'dev_gen' not in dev_dataset:
                        eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev', only_src=True)
                        dev_dataset['dev_gen'] = eval_examples, eval_data
                    else:
                        eval_examples, eval_data = dev_dataset['dev_gen']
                    
                    result = eval_generation_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', f'epoch_{cur_epoch}')
                    
                    # Update best metrics and save corresponding models
                    if result['em'] > best_em:
                        best_em = result['em']
                        not_em_inc_cnt = 0
                        # Save best EM model
                        best_em_dir = os.path.join(args.output_dir, 'checkpoint-best-em')
                        if not os.path.exists(best_em_dir):
                            os.makedirs(best_em_dir)
                        if args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            torch.save(model_to_save.state_dict(), os.path.join(best_em_dir, "pytorch_model.bin"))
                    else:
                        not_em_inc_cnt += 1

                    if result['corpus_bleu'] > best_bleu:
                        best_bleu = result['corpus_bleu']
                        not_bleu_inc_cnt = 0
                        # Save best BLEU model
                        best_bleu_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(best_bleu_dir):
                            os.makedirs(best_bleu_dir)
                        if args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            torch.save(model_to_save.state_dict(), os.path.join(best_bleu_dir, "pytorch_model.bin"))
                    else:
                        not_bleu_inc_cnt += 1

                    if result['rouge-l'] > best_rouge_l:
                        best_rouge_l = result['rouge-l']
                        not_rouge_l_inc_cnt = 0
                        # Save best ROUGE-L model
                        best_rouge_dir = os.path.join(args.output_dir, 'checkpoint-best-rouge-l')
                        if not os.path.exists(best_rouge_dir):
                            os.makedirs(best_rouge_dir)
                        if args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            torch.save(model_to_save.state_dict(), os.path.join(best_rouge_dir, "pytorch_model.bin"))
                    else:
                        not_rouge_l_inc_cnt += 1

                # Check early stopping condition
                counters = [not_loss_dec_cnt, not_em_inc_cnt, not_bleu_inc_cnt, not_rouge_l_inc_cnt]
                stopped_metrics_count = sum(1 for c in counters if c >= args.patience)
                
                if stopped_metrics_count >= 3:
                    logger.info(f"Early stopping at epoch {cur_epoch}: {stopped_metrics_count} metrics did not improve for {args.patience} epochs")
                    break

            # Clear GPU cache
            torch.cuda.empty_cache()

        # Close evaluation writers
        eval_dev_writer.close()
        eval_test_writer.close()
        
        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()

        logger.info("Training completed in %s", get_elapse_time(t0))

    # Testing phase
    if args.do_test:
        logger.info("Running testing on saved checkpoints")
        
        checkpoints_to_test = ['best-ppl', 'best-em', 'best-bleu', 'best-rouge-l', 'last']
        
        for criteria in checkpoints_to_test:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{criteria}', 'pytorch_model.bin')
            
            if not os.path.exists(checkpoint_path):
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                continue
            
            logger.info(f"Testing checkpoint: {criteria}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
            
            # Load test data
            if 'test_gen' not in test_dataset:
                test_examples, test_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test', only_src=True)
                test_dataset['test_gen'] = test_examples, test_data
            else:
                test_examples, test_data = test_dataset['test_gen']
            
            # Evaluate on test set
            result = eval_generation_epoch(args, test_data, test_examples, model, tokenizer, 'test', criteria)
            
            logger.info(f"Test results for {criteria}:")
            logger.info(f"  EM: {result['em']:.2f}")
            logger.info(f"  Corpus BLEU: {result['corpus_bleu']:.2f}")
            logger.info(f"  Sentence BLEU: {result['sentence_bleu']:.2f}")
            logger.info(f"  ROUGE-L: {result['rouge-l']:.2f}")

    logger.info("Process completed in %s", get_elapse_time(t0))
    fa.close()
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
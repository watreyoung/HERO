import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist
import json

# Configure logging settings
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    """Evaluate perplexity on evaluation dataset for one epoch"""
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running perplexity evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating perplexity"):
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
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    """Evaluate BLEU score and other metrics on evaluation dataset"""
    logger.info("  ***** Running BLEU evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating BLEU for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
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

    output_fn = os.path.join(args.res_dir, "{}_{}.output".format(split_tag,criteria))
    gold_fn = os.path.join(args.res_dir, "{}_{}.gold".format(split_tag,criteria))
    src_fn = os.path.join(args.res_dir, "{}_{}.src".format(split_tag,criteria))

    target_dict = {0: 'false', 1: 'true'}
    golds = [ex.target for ex in eval_examples]
    pred_y = []
    for p in pred_nls:
        if p == 'true':
            pred_y.append(1)
        else:
            pred_y.append(0)
    acc=accuracy_score(golds,pred_y)
    recall=recall_score(golds,pred_y)
    precision=precision_score(golds,pred_y)
    f1=f1_score(golds,pred_y)    
    result = {"acc": round(acc,4),
    "precision": round(precision,4),
    "recall": round(recall,4),
    "f1": round(f1,4)}

    with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
        for pred_nl, gold in zip(pred_nls, eval_examples):
            f.write(pred_nl.strip() + '\n')
            f1.write(target_dict[gold.target] + '\n')
            f2.write(gold.source.strip() + '\n')
        logger.info("Save the predictions into %s", output_fn)

    logger.info("***** Evaluation results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result, pred_y


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
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        eval_dev_writer = open(os.path.join(args.output_dir, 'eval_dev_per_epoch.txt'), 'a+')

        # Prepare training data loader
        train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
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

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu_em, best_f1, best_ppl = 0, -1, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                # Forward propagation and loss calculation
                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # Average loss for multi-GPU training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps  # Gradient accumulation
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                # Update parameters when gradient accumulation reaches specified steps
                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            # ========== Validation Phase: Evaluate on validation set after each epoch ==========
            if args.do_eval:
                # Load validation data
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                # Calculate perplexity on validation set
                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

                # Save the last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                # Early stopping strategy: based on PPL improvement
                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    best_ppl = eval_ppl

                    # Save best model based on PPL
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    # Early stopping condition: stop when both PPL and accuracy/F1 stop improving
                    if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                
                # Calculate accuracy, F1 and other metrics on validation set
                eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                    only_src=True, is_sample=True)

                result,_ = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                acc,f1 = result['acc'], result['f1']
                dev_bleu_em = acc

                dev_result_str = "[Epoch {} Dev] Acc: {:.4f}, PPL: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}\n".format(
                    cur_epoch, acc, eval_ppl, f1, result['precision'], result['recall']
                )
                eval_dev_writer.write(dev_result_str)
                eval_dev_writer.flush()
                logger.info(f"Dev results: {dev_result_str.strip()}")
                fa.write(dev_result_str)

                # TensorBoard logging
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_acc', dev_bleu_em, cur_epoch)
                
                # Save best model based on accuracy
                if dev_bleu_em > best_bleu_em:
                    not_bleu_em_inc_cnt = 0
                    logger.info("  [%d] Best acc: %.2f ", cur_epoch, dev_bleu_em)
                    logger.info("  " + "*" * 20)
                    best_bleu_em = dev_bleu_em
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.data_num == -1 or args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best bleu model into %s", output_model_file)

                # Save best model based on F1 score
                elif f1 > best_f1:
                    not_bleu_em_inc_cnt = 0
                    logger.info("  [%d] Best f1: %.2f ", cur_epoch, f1)
                    logger.info("  " + "*" * 20)
                    best_f1 = f1
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.data_num == -1 or args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best f1 model into %s", output_model_file)

                else:
                    not_bleu_em_inc_cnt += 1
                    logger.info("The perfomance does not increase for %d epochs", not_bleu_em_inc_cnt)
                    # Early stopping strategy: stop when both PPL and accuracy/F1 stop improving
                    if not_loss_dec_cnt > args.patience and not_bleu_em_inc_cnt > args.patience:
                        stop_early_str = "[%d] Early stop as not_loss_dec_cnt=%d and not_bleu_em_inc_cnt=%d (PPL and accuracy did not improve)\n" % (
                            cur_epoch, not_loss_dec_cnt, not_bleu_em_inc_cnt)
                        logger.info(stop_early_str)
                        fa.write(stop_early_str)
                        break

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

        eval_dev_writer.close()

    # ========== Testing Phase: Unified testing on saved checkpoints after training completion ==========
    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        test_title = f"\n\n===== Starting Test on {args.output_dir} at {time.ctime()} =====\n"
        logger.info(test_title)
        fa.write(test_title)

        # Test three saved checkpoints: best accuracy, best F1, and last epoch
        for criteria in ['best-acc','best-f1','last']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            if not os.path.exists(file):
                logger.warning(f"Checkpoint file not found, skipping: {file}")
                continue
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))
            eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                               only_src=True, is_sample=False)
            result, y_preds = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
            em = result['acc']
            precision = result['precision']
            recall = result['recall']
            f1 = result['f1']

            result_str = "[Test on {}] Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
                criteria, em, precision, recall, f1
            )
            logger.info(result_str)
            fa.write(result_str + '\n')

            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write(result_str)
            
            # Save prediction results
            with open(args.test_filename,'r') as f1,open(os.path.join(args.output_dir, "prediction-{}.jsonl".format(criteria)) ,'w') as f:
                lines = []
                for line in f1:
                    line = json.loads(line)
                    lines.append(line)

                for new_line,y_pred in zip(lines,y_preds):
                    new_line['prediction'] = str(y_pred)
                    f.write(json.dumps(new_line))
                    f.write('\n')
                    
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()

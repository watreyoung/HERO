"""
Evaluation Metrics Calculator for Code Review Comment Generation (CRCG)

This script calculates various generation quality metrics including:
- NLTK BLEU
- Old ROUGE
- Edit Similarity
- SacreBLEU
- Google ROUGE
"""

import os
import json
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import sacrebleu
from rouge_score import rouge_scorer


# Initialize Google Rouge Scorer
# use_stemmer=True is effective for English
google_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def calculate_edit_similarity(str1, str2):
    """Calculate edit similarity between two strings."""
    if not str1 and not str2: return 1.0
    if not str1 or not str2: return 0.0
    dist = Levenshtein.distance(str1, str2)
    max_len = max(len(str1), len(str2))
    if max_len == 0: return 0.0
    return 1.0 - (dist / max_len)


def calculate_nltk_bleu(reference, candidate):
    """Calculate NLTK BLEU score."""
    if not reference and not candidate: return 1.0
    if not reference or not candidate: return 0.0
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    smooth = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smooth)


def calculate_old_rouge(reference, candidate):
    """Calculate ROUGE score using old rouge library."""
    if not reference or not candidate:
        return 0.0 if (reference or candidate) else 1.0
    rouge = Rouge()
    try:
        scores = rouge.get_scores(candidate, reference)
        return scores[0]['rouge-l']['f']
    except Exception:
        return 0.0


def calculate_sacrebleu(reference, candidate):
    """Calculate SacreBLEU score."""
    if not reference and not candidate: return 1.0
    if not reference or not candidate: return 0.0
    # tokenize='13a' is the default standard
    score_obj = sacrebleu.sentence_bleu(candidate, [reference], tokenize='13a')
    # sacrebleu returns 0-100, divide by 100 to normalize
    return score_obj.score / 100.0


def calculate_google_rouge(reference, candidate):
    """Calculate Google ROUGE-L score."""
    if not reference and not candidate: return 1.0
    if not reference or not candidate: return 0.0
    scores = google_scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure


def main():
    current_dir = os.getcwd()
    
    # Get file list, filter for files ending with 'json'
    json_files = sorted([f for f in os.listdir(current_dir) if f.endswith('json')])
    
    output_results = []
    
    if not json_files:
        print("No JSON files found in current directory.")
        return

    print(f"Found {len(json_files)} matching files, processing in filename order...")

    for json_file in json_files:
        file_path = os.path.join(current_dir, json_file)
        
        # Initialize accumulators
        metrics_sum = {
            'nltk_bleu': 0.0,
            'old_rouge': 0.0,
            'edit_sim': 0.0,
            'sacre_bleu': 0.0,
            'google_rouge': 0.0
        }
        count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        target = str(data.get('target', ''))
                        output = str(data.get('output', ''))
                        
                        # Calculate all metrics
                        metrics_sum['nltk_bleu'] += calculate_nltk_bleu(target, output)
                        metrics_sum['old_rouge'] += calculate_old_rouge(target, output)
                        metrics_sum['edit_sim'] += calculate_edit_similarity(target, output)
                        metrics_sum['sacre_bleu'] += calculate_sacrebleu(target, output)
                        metrics_sum['google_rouge'] += calculate_google_rouge(target, output)
                        
                        count += 1
                    except json.JSONDecodeError:
                        continue
            
            # Calculate averages
            if count > 0:
                # Multiply by 100 to convert 0-1 decimals to 0-100 percentages
                avgs = {k: (v / count) * 100 for k, v in metrics_sum.items()}
            else:
                avgs = {k: 0.0 for k in metrics_sum.keys()}
            
            # Format result string in required order
            result_str = (
                f"{json_file} "
                f"{avgs['nltk_bleu']:.2f} "
                f"{avgs['old_rouge']:.2f} "
                f"{avgs['edit_sim']:.2f} "
                f"{avgs['sacre_bleu']:.2f} "
                f"{avgs['google_rouge']:.2f}"
            )
            output_results.append(result_str)
            print(f"Processed: {json_file} (lines: {count})")
            
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")

    # Write results to txt file
    output_filename = 'metrics_report_test.txt'
    with open(output_filename, 'w', encoding='utf-8') as f_out:
        # Define header format
        header = f"{'Filename':<30} {'NLTK-BLEU':<12} {'Old-ROUGE':<12} {'Edit-Sim':<12} {'SacreBLEU':<12} {'Google-ROUGE':<12}\n"
        f_out.write(header)
        f_out.write("-" * 100 + "\n")
        
        for res in output_results:
            parts = res.split()
            fname = parts[0]
            # Truncate long filenames
            if len(fname) > 28: fname = fname[:25] + "..."
            
            line = (
                f"{fname:<30} "
                f"{parts[1]:<12} "
                f"{parts[2]:<12} "
                f"{parts[3]:<12} "
                f"{parts[4]:<12} "
                f"{parts[5]:<12}\n"
            )
            f_out.write(line)
            
    print(f"\nAll calculations complete! Results saved to {output_filename}")


if __name__ == '__main__':
    main()
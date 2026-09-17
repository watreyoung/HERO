"""
LLM Output Metrics Calculator for Review Necessity Prediction (RNP)

This script parses LLM outputs (YES/NO format) and calculates classification metrics.
Uses strict regex matching to avoid substring false positives.
"""

import os
import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_sentiment(text):
    """
    Parse YES/NO from text using strict regex matching to avoid substring false positives.
    
    Returns:
        1 for YES (review needed)
        0 for NO (no review needed)
        -1 for parsing failure
    """
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower().strip()
    
    # ============================================================
    # 1. Define regex patterns
    # ============================================================
    pattern_phrase_yes = r"review\s+necessity\s*[:\-]\s*(?:<)?\s*yes\s*(?:>)?"
    pattern_phrase_no  = r"review\s+necessity\s*[:\-]\s*(?:<)?\s*no\s*(?:>)?"
    
    pattern_word_yes = r"\byes\b"
    pattern_word_no  = r"\bno\b"

    # ============================================================
    # 2. Execute matching
    # ============================================================
    match_p_yes = re.search(pattern_phrase_yes, text)
    match_p_no = re.search(pattern_phrase_no, text)
    
    if match_p_yes and match_p_no:
        return 1 if match_p_yes.start() < match_p_no.start() else 0
    if match_p_yes:
        return 1
    if match_p_no:
        return 0
        
    match_w_yes = re.search(pattern_word_yes, text)
    match_w_no = re.search(pattern_word_no, text)
    
    if match_w_yes and match_w_no:
        return 1 if match_w_yes.start() < match_w_no.start() else 0
    if match_w_yes:
        return 1
    if match_w_no:
        return 0
    
    return -1


def main():
    current_dir = os.getcwd()
    
    json_files = []
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.endswith(".json") and "result" in file:
                json_files.append(os.path.join(root, file))
    
    json_files = sorted(json_files)
    output_results = []
    
    if not json_files:
        print("No matching result files found.")
        return

    print(f"Found {len(json_files)} files, calculating comprehensive metrics (strict regex mode)...\n")

    for file_path in json_files:
        y_true_all = []
        y_pred_all = []
        
        file_name = os.path.relpath(file_path, current_dir)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        target_raw = data.get('target', '')
                        output_raw = data.get('output', '')
                        
                        label_true = parse_sentiment(target_raw)
                        label_pred = parse_sentiment(output_raw)
                        
                        if label_true == -1:
                            continue
                            
                        y_true_all.append(label_true)
                        y_pred_all.append(label_pred)
                        
                    except json.JSONDecodeError:
                        continue
            
            total_samples = len(y_true_all)
            
            if total_samples == 0:
                print(f"File {file_name} has no valid data.")
                continue

            # -------------------------------------------------------
            # Metrics calculation logic
            # -------------------------------------------------------
            
            # 1. Failure Rate
            fail_count = y_pred_all.count(-1)
            fail_rate = (fail_count / total_samples) * 100

            # 2. Strict Accuracy (denominator=Total, failure=wrong)
            correct_strict = sum(1 for t, p in zip(y_true_all, y_pred_all) if t == p)
            strict_acc = (correct_strict / total_samples) * 100

            # 3. Valid Metrics (denominator=Valid, excluding failed samples)
            valid_indices = [i for i, p in enumerate(y_pred_all) if p != -1]
            y_true_valid = [y_true_all[i] for i in valid_indices]
            y_pred_valid = [y_pred_all[i] for i in valid_indices]
            
            if len(y_true_valid) > 0:
                valid_acc = accuracy_score(y_true_valid, y_pred_valid) * 100
                valid_precision = precision_score(y_true_valid, y_pred_valid, zero_division=0) * 100
                valid_recall = recall_score(y_true_valid, y_pred_valid, zero_division=0) * 100
                valid_f1 = f1_score(y_true_valid, y_pred_valid, zero_division=0) * 100
            else:
                valid_acc = valid_precision = valid_recall = valid_f1 = 0.0

            # -------------------------------------------------------
            # Format output
            # Order: V-Acc, V-Prec, V-Rec, V-F1, S-Acc, Fail
            # -------------------------------------------------------
            result_str = (
                f"{file_name} "
                f"{valid_acc:.2f} "       # 1. Valid Acc
                f"{valid_precision:.2f} " # 2. Valid Precision
                f"{valid_recall:.2f} "    # 3. Valid Recall
                f"{valid_f1:.2f} "        # 4. Valid F1
                f"{strict_acc:.2f} "      # 5. Strict Acc
                f"{fail_rate:.2f}%"       # 6. Fail Rate
            )
            output_results.append(result_str)
            
            print(f"Processing: {file_name}")
            print(f"    -> Valid Acc:  {valid_acc:.2f}%")
            print(f"    -> Valid F1:   {valid_f1:.2f}%")
            print(f"    -> Strict Acc: {strict_acc:.2f}%")
            print(f"    -> Fail Rate:  {fail_rate:.2f}%")
            print("-" * 40)

        except Exception as e:
            print(f"Error reading file {file_name}: {e}")

    # Save report
    report_file = 'metrics_report_strict.txt'
    with open(report_file, 'w', encoding='utf-8') as f_out:
        # Header with correct order
        header = (
            f"{'File Name':<45} "
            f"{'V-Acc':<8} "
            f"{'V-Prec':<8} "
            f"{'V-Rec':<8} "
            f"{'V-F1':<8} "
            f"{'S-Acc':<8} "
            f"{'Fail%':<10}\n"
        )
        f_out.write(header)
        f_out.write("-" * 110 + "\n")
        
        for res in output_results:
            parts = res.split()
            
            fname = parts[0]
            if len(fname) > 43: fname = "..." + fname[-40:]
            
            line = (
                f"{fname:<45} "
                f"{parts[1]:<8} "  # V-Acc
                f"{parts[2]:<8} "  # V-Prec
                f"{parts[3]:<8} "  # V-Rec
                f"{parts[4]:<8} "  # V-F1
                f"{parts[5]:<8} "  # S-Acc
                f"{parts[6]:<10}\n" # Fail%
            )
            f_out.write(line)
            
    print(f"\nAll calculations complete! Results saved to {report_file}")


if __name__ == '__main__':
    main()

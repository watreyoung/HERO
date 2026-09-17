"""
Evaluation Metrics Calculator for Review Necessity Prediction (RNP)

This script calculates classification metrics including:
- Accuracy
- Precision
- Recall
- F1-Score
"""

import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def normalize_label(value):
    """
    Normalize various label formats (True/False, "True"/"False", 0/1, "0"/"1")
    to integers 0 or 1.
    """
    # Convert to string, lowercase, and strip whitespace
    s_val = str(value).strip().lower()
    
    # Define true value set
    true_set = {'true', '1', 'yes', 't'}
    # Define false value set
    false_set = {'false', '0', 'no', 'f'}
    
    if s_val in true_set:
        return 1
    elif s_val in false_set:
        return 0
    else:
        # If unrecognized label format, default to 0
        return 0


def main():
    current_dir = os.getcwd()
    
    # Get file list, maintain previous sorting and filtering logic
    json_files = sorted([f for f in os.listdir(current_dir) if f.endswith('json')])
    
    output_results = []
    
    if not json_files:
        print("No JSON files found in current directory.")
        return

    print(f"Found {len(json_files)} matching files, processing in filename order...")

    for json_file in json_files:
        file_path = os.path.join(current_dir, json_file)
        
        # Store all true and predicted labels
        y_true = []
        y_pred = []
        
        count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        
                        # Get raw data
                        target_raw = data.get('target')
                        output_raw = data.get('output')
                        
                        # Core step: normalize raw data to 0 or 1
                        # This handles cases like target: 0 and output: "False"
                        true_label = normalize_label(target_raw)
                        pred_label = normalize_label(output_raw)
                        
                        y_true.append(true_label)
                        y_pred.append(pred_label)
                        
                        count += 1
                    except json.JSONDecodeError:
                        continue
            
            # Calculate classification metrics
            if count > 0:
                # Calculate metrics (results are 0-1 decimals)
                # zero_division=0 prevents division by zero errors
                acc = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # Convert to percentage format (0-100)
                res_acc = acc * 100
                res_pre = precision * 100
                res_rec = recall * 100
                res_f1 = f1 * 100
            else:
                res_acc = res_pre = res_rec = res_f1 = 0.0
            
            # Format result string
            result_str = (
                f"{json_file} "
                f"{res_acc:.2f} "
                f"{res_pre:.2f} "
                f"{res_rec:.2f} "
                f"{res_f1:.2f}"
            )
            output_results.append(result_str)
            print(f"Processed: {json_file} (samples: {count})")
            
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")

    # Write results to txt file
    output_filename = 'metrics_report_classification.txt'
    with open(output_filename, 'w', encoding='utf-8') as f_out:
        # Define header format
        header = f"{'Filename':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n"
        f_out.write(header)
        f_out.write("-" * 90 + "\n")
        
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
                f"{parts[4]:<12}\n"
            )
            f_out.write(line)
            
    print(f"\nAll calculations complete! Results saved to {output_filename}")


if __name__ == '__main__':
    main()
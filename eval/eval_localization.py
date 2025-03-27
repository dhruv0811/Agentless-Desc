import json
import re
from collections import defaultdict
import argparse
import datasets

def extract_files_from_patch(patch_text):
    """Extract file paths from patch text using regex"""
    # This regex looks for file paths in diff format (e.g., +++ b/path/to/file.py)
    file_paths = re.findall(r'(?:---|\+\+\+) [ab]/([^\n\t]+)', patch_text)
    # Remove duplicates while preserving order
    unique_files = []
    for file_path in file_paths:
        if file_path not in unique_files:
            unique_files.append(file_path)
    return unique_files

def load_jsonl(file_path):
    """Load JSONL file and return list of JSON objects"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def compute_metrics(predictions, gold_standards):
    """Compute precision, recall, F1 score for file retrieval"""
    results = {}
    all_metrics = {
        'total_instances': len(gold_standards),
        'total_predictions': 0,
        'total_gold_files': 0,
        'total_correct': 0,
        'instance_metrics': []
    }
    
    for instance_id, gold_files in gold_standards.items():
        if instance_id not in predictions:
            # Skip instances not in predictions
            instance_result = {
                'instance_id': instance_id,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'predicted_files': [],
                'gold_files': gold_files,
                'correct_files': []
            }
            all_metrics['instance_metrics'].append(instance_result)
            all_metrics['total_gold_files'] += len(gold_files)
            continue
            
        pred_files = predictions[instance_id]
        correct_files = [f for f in pred_files if f in gold_files]
        
        # Calculate metrics
        precision = len(correct_files) / len(pred_files) if pred_files else 0
        recall = len(correct_files) / len(gold_files) if gold_files else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        instance_result = {
            'instance_id': instance_id,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predicted_files': pred_files,
            'gold_files': gold_files,
            'correct_files': correct_files
        }
        
        all_metrics['instance_metrics'].append(instance_result)
        all_metrics['total_predictions'] += len(pred_files)
        all_metrics['total_gold_files'] += len(gold_files)
        all_metrics['total_correct'] += len(correct_files)
    
    # Calculate aggregate metrics
    all_metrics['macro_precision'] = sum(inst['precision'] for inst in all_metrics['instance_metrics']) / len(all_metrics['instance_metrics'])
    all_metrics['macro_recall'] = sum(inst['recall'] for inst in all_metrics['instance_metrics']) / len(all_metrics['instance_metrics'])
    all_metrics['macro_f1'] = sum(inst['f1'] for inst in all_metrics['instance_metrics']) / len(all_metrics['instance_metrics'])
    
    all_metrics['micro_precision'] = all_metrics['total_correct'] / all_metrics['total_predictions'] if all_metrics['total_predictions'] > 0 else 0
    all_metrics['micro_recall'] = all_metrics['total_correct'] / all_metrics['total_gold_files'] if all_metrics['total_gold_files'] > 0 else 0
    all_metrics['micro_f1'] = 2 * all_metrics['micro_precision'] * all_metrics['micro_recall'] / (all_metrics['micro_precision'] + all_metrics['micro_recall']) if (all_metrics['micro_precision'] + all_metrics['micro_recall']) > 0 else 0
    
    return all_metrics

def main():
    parser = argparse.ArgumentParser(description='Compare file retrieval predictions with gold standard')
    parser.add_argument('--predictions', default='/home/dhruvgu2/Agentless/results/swe-bench-lite/file_level/loc_outputs.jsonl', help='Path to predictions JSONL file')
    parser.add_argument('--output', default='./results/retrieval_metrics_gpt4o_desc.json', help='Output JSON file path')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to use from the dataset')
    args = parser.parse_args()
    
    # Load predictions data
    predictions_data = load_jsonl(args.predictions)
    
    # Load SWE-bench_Verified from Hugging Face datasets
    print("Loading SWE-bench_Verified dataset from Hugging Face...")
    gold_dataset = datasets.load_dataset("princeton-nlp/SWE-bench_Verified")
    gold_data = gold_dataset['test'].select(range(min(args.num_samples, len(gold_dataset['test']))))
    
    # Extract predictions
    predictions = {}
    for item in predictions_data:
        instance_id = item.get('instance_id')
        found_files = item.get('found_files', [])
        if instance_id:
            predictions[instance_id] = found_files
    
    # Extract gold standard file paths from patches
    gold_standards = {}
    for item in gold_data:
        instance_id = item.get('instance_id')
        patch = item.get('patch', '')
        if instance_id and patch:
            gold_files = extract_files_from_patch(patch)
            gold_standards[instance_id] = gold_files
    
    # Compute metrics
    metrics = compute_metrics(predictions, gold_standards)
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print(f"Total instances: {metrics['total_instances']}")
    print(f"Micro-Precision: {metrics['micro_precision']:.4f}")
    print(f"Micro-Recall: {metrics['micro_recall']:.4f}")
    print(f"Micro-F1: {metrics['micro_f1']:.4f}")
    print(f"Macro-Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro-Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    main()
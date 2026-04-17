import json
import pickle
import argparse
from datetime import datetime
from vllm import SamplingParams
from dynasor.core.evaluator import math_equal
import os
import numpy as np
import pyarrow.parquet as pq
import pdb
from evaluator import aggregation, Chronos


def quick_parse(text: str) -> str:
    """Parse LaTeX text content"""
    if '\\text{' in text and '}' in text:
        # Find all occurrences of \text{...} and remove them
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            # Replace \text{content} with just content
            content = text[start + 6:end]  # 6 is length of '\text{'
            text = text[:start] + content + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer equals ground truth"""
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        # 若math_equal计算时间超过10秒则flag=False
        flag = math_equal(answer, ground_truth, timeout=10)
        return flag


def evaluate_voting_results(voting_results, ground_truth):
    """Evaluate voting results against ground truth"""
    evaluation = {}
    
    for method, result in voting_results.items():
        if result and result.get('answer'):
            try:
                is_correct = equal_func(result['answer'], ground_truth)
            except:
                is_correct = str(result['answer']) == str(ground_truth)
            
            evaluation[method] = {
                'answer': result['answer'],
                'is_correct': is_correct,
                'confidence': result.get('confidence'),
                'num_votes': result.get('num_votes', 0)
            }
        else:
            evaluation[method] = {
                'answer': None,
                'is_correct': False,
                'confidence': None,
                'num_votes': 0
            }
    
    return evaluation


def print_evaluation_report(question, ground_truth, evaluation, result):
    """Print detailed evaluation report"""
    print(f"\n=== Evaluation Report ===")
    print(f"Question: {question}")
    print(f"Ground truth: {ground_truth}")
    print(f"Total traces generated: {result.total_traces_count}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Generation time: {result.generation_time:.2f}s")
    

    print(f"\n=== Voting Method Results ===")
    print("-" * 80)
    print(f"{'Method':<25} {'Correct':<8} {'Answer':<20} {'Confidence':<12} {'Votes':<6}")
    print("-" * 80)
    
    correct_methods = []
    for method, eval_result in evaluation.items():
        answer = str(eval_result['answer'])[:18] + '...' if len(str(eval_result['answer'])) > 20 else str(eval_result['answer'])
        is_correct = eval_result['is_correct']
        confidence = eval_result['confidence']
        num_votes = eval_result['num_votes']
        
        correct_str = '✓' if is_correct else '✗'
        conf_str = f"{confidence:.3f}" if confidence is not None else '-'
        
        print(f"{method:<25} {correct_str:<8} {answer:<20} {conf_str:<12} {num_votes:<6}")
        
        if is_correct:
            correct_methods.append(method)
    
    print(f"\nCorrect voting methods: {correct_methods}")
    
    # Find best method by confidence among correct ones
    correct_evals = {method: eval_result for method, eval_result in evaluation.items() 
                    if eval_result['is_correct']}
    
    if correct_evals:
        best_method = max(correct_evals.items(), 
                         key=lambda x: x[1]['confidence'] if x[1]['confidence'] is not None else 0)
        print(f"Best correct method: {best_method[0]} (confidence: {best_method[1]['confidence']:.3f})")
    
    # Method performance summary
    total_methods = len(evaluation)
    correct_count = len(correct_methods)
    print(f"Method accuracy: {correct_count}/{total_methods} ({correct_count/total_methods:.1%})")

def load_one_data(file_path):
    with open(file_path, 'rb') as file:
        data = pq.read_table(file)
    df = data.to_pandas()
    for bidx, batch in enumerate(df['logprobs']):
        for tidx, trace in enumerate(batch):
            with open(trace, 'rb') as f:
                logprob = pq.read_table(f).to_pandas()['topk_logprobs'].tolist()[0]
                df['logprobs'][bidx][tidx] = logprob
    return df

def load_path_from_config(config_file: str):
    with open(config_file, 'r') as f:
        path_dict = json.load(f)
    return list(path_dict.values())

def load_ranker(model_path: str):
    with open(model_path, 'r') as f:
        path_dict = json.load(f)
    ranker_dict = {}
    for k, v in path_dict.items():
        ranker = Chronos(v)
        ranker_dict[k] = ranker
    return ranker_dict

def main():
    parser = argparse.ArgumentParser(description='Chronos')
    parser.add_argument('--input_path', type=str, default='xxx',
                       help='Path to the input file for trace scoring')
    parser.add_argument('--model_path', type=str, default='xxx',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.input_path}...")


    ranker_dict = load_ranker(args.model_path)
    input_path = os.path.join(args.input_path, 'processed_data')
    output_path = os.path.join(args.input_path, 'eval_data')
    os.makedirs(output_path, exist_ok=True)
    input_files = os.listdir(input_path)
    for file in input_files:
        qid = int(file.split('.')[0].replace('qid_', ''))
        print(f"Loading data for question {qid}")
        df = load_one_data(os.path.join(input_path, file))
        for index, sample in df.iterrows():
            question = sample['problem']
            ground_truth = str(sample.get('answer', '')).strip()
            print(f"Processing question {qid}: {question[:100]}...")
            
            result = aggregation(
                input_trace=sample,
                ranker=ranker_dict,
            )

            if ground_truth and result.voting_results:
                evaluation = evaluate_voting_results(result.voting_results, ground_truth)
                print_evaluation_report(question, ground_truth, evaluation, result)
            
            
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pass1_answer = result.all_traces[0]['text']
            all_methods_flag = {k:v['is_correct'] for k,v in evaluation.items()}
            save_data = {
                'ground_truth': ground_truth,
                'pass1_answer': pass1_answer,
                'all_methods_flag': all_methods_flag,
            }
            
            save_filename = f"{output_path}/qid{qid}_{timestamp}.pkl"
            
            with open(save_filename, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"\nResults saved to {save_filename}")



if __name__ == "__main__":
    main()
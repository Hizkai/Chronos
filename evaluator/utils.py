from collections import Counter, defaultdict
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
import sys
import torch
import torch.nn as nn
import pdb
from sklearn.preprocessing import StandardScaler
import json
from .outputs import DeepThinkOutput
import time
import re
from scorer import InceptionTime, InceptionNet, ResidualBlock, InceptionModule


class Chronos:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        with open(f'{self.model_path}/config.json', 'r') as f:
            config = json.load(f)
        temp_model = InceptionTime(
            num_ensemble=config['num_ensemble'],
            in_channels=1,
            num_classes=1,
            num_residual_blocks=config['num_residual_blocks'],
            bottleneck_size=config['bottleneck_size'],
            conv_lengths=config['conv_lengths'],
            conv_filters=config['conv_filters']
        ).to(self.device)

        scaler = StandardScaler()
        scaler_params = config['scaler_params']
        scaler.mean_ = np.array(scaler_params['mean'])
        scaler.scale_ = np.array(scaler_params['scale'])
        self.scaler = scaler

        try:
            state_dict = torch.load(f'{self.model_path}/model.pth', map_location=self.device)
            temp_model.load_state_dict(state_dict)
            print(f"Successfully loaded pre-trained Chronos: {self.model_path}/model.pth")
        except Exception as e:
            print(f"Warning: Failed to load pre-trained model, using random initialized model: {e}")
        
        self.model = temp_model
        self.model.eval()
    
    def score(self, confs: List[float]) -> float:

        if len(confs) < 2048:
            confs = [np.mean(confs)] * (2048 - len(confs)) + confs

        confs = self.scaler.transform(np.array(confs[-2048:]).reshape(1, -1)).flatten()
        
        input_tensor = torch.tensor(confs, dtype=torch.float32).to(self.device)
        input_tensor = input_tensor.unsqueeze(0)
    
        input_tensor = input_tensor.unsqueeze(1)
        
        with torch.no_grad():
            score = self.model(input_tensor).item()
        
        return score



def extract_answer(text: str) -> Optional[str]:
    """Extract boxed answer from text"""
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        return a.strip()
    
    # " Please show your choice in the answer field with only the choice letter, e.g., 'answer': 'c'."
    elif "answer:" in text or "Answer:" in text:
        if "answer:" in text:
            ans = text.split("answer:")[-1]
        else:
            ans = text.split("Answer:")[-1]
        ans = re.findall(r'[a-zA-Z]', ans)
        if len(ans) == 0:
            return None
        else:
            return ans[0]
    
    return None



#=============================        

def compute_confidence(logprobs: List[Dict]) -> List[float]:
    """Compute confidence score from logprobs and return only confidence values"""
    confs = []
    for token_logprobs in logprobs:
        mean_logprob = np.mean(token_logprobs)
        confs.append(round(-mean_logprob, 3))
    return confs


def compute_least_grouped(confs: List[float], group_size: int) -> List[float]:
    """Compute sliding window mean confidence"""
    if len(confs) < group_size:
        return [sum(confs) / len(confs)] if confs else [0]
    
    sliding_means = []
    for i in range(len(confs) - group_size + 1):
        window = confs[i:i + group_size]
        sliding_means.append(round(sum(window) / len(window), 3))
    return sliding_means


# ============= VOTING FUNCTIONS =============

def simple_majority_vote(answers: List[str]) -> Optional[str]:
    """Simple majority voting"""
    if not answers:
        return None
    
    vote_counts = Counter(answers)
    return vote_counts.most_common(1)[0][0]


def weighted_majority_vote(answers: List[str], weights: List[float]) -> Optional[str]:
    """Perform weighted majority voting"""
    if not answers:
        return None
    
    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
    
    if not answer_weights:
        return None
    
    return max(answer_weights.keys(), key=lambda x: answer_weights[x])





def compute_all_voting_results(traces: List[Dict[str, Any]], scorer) -> Dict[str, Any]:
    """Compute results for all voting methods"""
    valid_traces = [trace for trace in traces if trace.get('extracted_answer')]
    
    if not valid_traces:
        return {method: None for method in [
            'majority', 'Chronos',
        ]}
    
    answers = [trace['extracted_answer'] for trace in valid_traces]

    voting_results = {}
    
    majority_answer = simple_majority_vote(answers)
    voting_results['majority'] = {
        'answer': majority_answer,
        'num_votes': len(answers),
        'confidence': None
    }
    
    scorer_names, scorer_traces, scorer_scores = get_scorer_scores(valid_traces, scorer, 0.1)
    for names, traces, confs in zip(scorer_names, scorer_traces, scorer_scores):
        answers = [trace['extracted_answer'] for trace in traces]
        confidences = confs
        if any(c > 0 for c in confidences):
            answer = weighted_majority_vote(answers, confidences)
            voting_results[f'Chronos_{names}'] = {
                'answer': answer,
                'num_votes': len(answers),
                'confidence': np.mean(confidences)
            }

    
    
    return voting_results


def get_scorer_scores(valid_traces, scorer, top_percent):
    scorer_traces = []
    scorer_confs = []
    scorer_names = []
    for name, scorer in scorer.items():
        confidences = []
        for trace in valid_traces:
            conf = scorer.score(trace['confs'])
            confidences.append(conf)       
        scorer_names.append(name)
        
        traces = []
        confs = []
        for trace, conf in zip(valid_traces, confidences):
            if conf >= np.percentile(confidences, (1-top_percent)*100):
                traces.append(trace)
                confs.append(conf)
        scorer_traces.append(traces)
        scorer_confs.append(confs)
        
    return scorer_names, scorer_traces, scorer_confs

# ============= OUTPUT PROCESSING =============


def process_output_offline(output, idx: int) -> Dict[str, Any]:
    """Process a single vLLM output for offline mode - stores full confidence array"""
    text = output['text'][idx]
    logprobs = output['logprobs'][idx]
    confs = compute_confidence(logprobs)
    
    extracted_answer = extract_answer(text)
    
    return {
        "text": text,
        "num_tokens": len(logprobs),
        "logprobs": logprobs,
        "confs": confs,
        "extracted_answer": extracted_answer,
    }


def process_batch_results_offline(batch_outputs) -> Dict[str, Any]:
    """Process batch results from vLLM for offline mode"""
    traces = []
    total_tokens = 0
    
    for i in range(128):
        trace_data = process_output_offline(batch_outputs, i)
        traces.append(trace_data)
        total_tokens += trace_data["num_tokens"]
    
    return {
        'traces': traces,
        'total_tokens': total_tokens,
        'num_traces': len(traces)
    }



def aggregation(input_trace, scorer):
    total_start_time = time.time()
    output = DeepThinkOutput()
    result = process_trace(input_trace, output)
    print("Computing multiple voting results...")
    voting_start = time.time()
    output.voting_results = compute_all_voting_results(output.all_traces, scorer)
    
    if 'majority' in output.voting_results and output.voting_results['majority']:
        output.voted_answer = output.voting_results['majority']['answer']
        output.final_answer = output.voted_answer
    
    voting_time = time.time() - voting_start
    print(f"Multiple voting computed in {voting_time:.2f} seconds")
    
    output.total_time = time.time() - total_start_time
    return output

def process_trace(input, output):
    processing_start = time.time()
    processed_results = process_batch_results_offline(input)
    
    output.all_traces = processed_results['traces']
    output.total_tokens = processed_results['total_tokens']
    output.total_traces_count = len(output.all_traces)
    output.avg_tokens_per_trace = output.total_tokens / output.total_traces_count if output.total_traces_count > 0 else 0
    
    basic_voting(output)
    
    output.processing_time = time.time() - processing_start
    return output

def basic_voting(output):
    voting_answers = []
    voting_weights = []
    
    for trace in output.all_traces:
        if trace.get('extracted_answer'):
            voting_answers.append(trace['extracted_answer'])
            voting_weights.append(1.0)
    
    output.voting_answers = voting_answers
    output.voting_weights = voting_weights
    
    output.voted_answer = weighted_majority_vote(voting_answers, voting_weights)
    output.final_answer = output.voted_answer
    
    
    print(f'Basic voting candidates: {len(voting_answers)}')
    if voting_answers:
        print(f'Sample voting answers: {voting_answers[:5]}')



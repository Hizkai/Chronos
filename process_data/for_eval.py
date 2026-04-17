import json
import pickle
import os
import pandas as pd
import pyarrow.parquet as pq
import random
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="xxx")
args = parser.parse_args()
data_path = args.data_path

logprob_dir = os.path.join(data_path, 'worker_all')
meta_path = os.path.join(data_path, 'shard_out_math_verified_sorted.jsonl')


data = []
count = 0
with open(meta_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))
        count += 1
print(f"meta_path: {meta_path}")
print(f"total {len(data)}")


data_dict = {}
for item in tqdm(data, desc="marging"):
    qid = item['id']
    aid = item['aid']
    problem = item['problem']
    answer = item['answer']
    if qid not in data_dict:
        data_dict[qid] = {'problem':problem, 'answer':answer, 'text':[], 'token_ids': [], 'logprobs':[]}
    data_dict[qid]['text'].append(item['response'])
    match_str = f'qid{qid}_aid{aid}.parquet'
    if match_str in os.listdir(logprob_dir):
        data_dict[qid]['logprobs'].append(os.path.join(logprob_dir, match_str))
    else:
        print(f"not found {match_str}")
    
print(f"total {len(data_dict)} questions")


output_dir = os.path.join(data_path, 'processed_data')
os.makedirs(output_dir, exist_ok=True)


processed_qids = 0
all_samples_count = 0

for qid in tqdm(data_dict.keys()):
    qid_samples_count = 0
    output_file = os.path.join(output_dir, f'qid_{qid}.parquet')
    
    all_samples = []
    for _ in tqdm(range(16)):
        sample = random.sample(list(zip(data_dict[qid]['text'], data_dict[qid]['logprobs'])), 128)
        sample_text, sample_logprobs = zip(*sample)
        
        item = {'qid': qid, 'problem': data_dict[qid]['problem'], 'answer': data_dict[qid]['answer'], 
               'text': list(sample_text), 'logprobs': list(sample_logprobs)}
        
        all_samples.append(item)
        qid_samples_count += 1
        all_samples_count += 1
    
    df = pd.DataFrame(all_samples)
    df.to_parquet(output_file, index=False)
    
    if qid_samples_count > 0:
        processed_qids += 1
        print(f"qid={qid} saved {output_file}, total {qid_samples_count} samples")

print(f"Finish, processed {processed_qids} questions, total {all_samples_count} samples")

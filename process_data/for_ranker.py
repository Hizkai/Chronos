import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="xxxx")
args = parser.parse_args()

data_path = args.data_path

logprob_dir = os.path.join(data_path, 'worker_all')
meta_path = os.path.join(data_path, 'shard_out_math_verified_sorted.jsonl')
save_dir = data_path
os.makedirs(save_dir, exist_ok=True)

train_ratio = 0.8

meta = {}
with open(meta_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        qid, aid = str(data['id']), int(data['aid'])
        meta[(qid, aid)] = {
            'flag': int(data['flag']),
            'answer': data.get('answer'),
            'response': data.get('response')
        }


def process_single_file(args):
    file_path, qid, aid = args
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"error {file_path}: {e}")
        return None

    if 'topk_logprobs' not in df.columns:
        return None

    logprobs_list = df['topk_logprobs'].tolist()
    if len(logprobs_list) == 0:
        return None


    token_array = logprobs_list[0]
    if isinstance(token_array, np.ndarray):
        try:
            token_array_2d = np.stack(token_array, axis=0)
        except Exception as e:
            print(f" {file_path} : {e}")
            return None
    else:
        print(f"{file_path} : {type(token_array)}")
        return None


    mean_prob = -np.mean(token_array_2d, axis=1)
    mean_prob = mean_prob.reshape(-1, 1)
    num_tokens, top_k = token_array_2d.shape
    if num_tokens < 2048:
        pad_len = 2048 - num_tokens
        pad_avg = np.mean(mean_prob, axis=0)
        pad_array = np.tile(pad_avg, (pad_len, 1))
        mean_prob = np.vstack((pad_array, mean_prob))

    mean_prob = mean_prob[-2048:].squeeze()

    label = meta.get((qid, aid), {}).get('flag', None)
    return (mean_prob.astype(np.float32), label, int(qid))

tasks = []
for filename in os.listdir(logprob_dir):
    if not filename.endswith('.parquet'):
        continue
    parts = filename.replace('.parquet', '').split('_')
    try:
        qid = parts[1].replace('qid', '')
        aid = int(parts[2].replace('aid', ''))
    except Exception:
        continue
    file_path = os.path.join(logprob_dir, filename)
    tasks.append((file_path, qid, aid))

results = []
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_single_file, t) for t in tasks]
    for f in tqdm(as_completed(futures), total=len(futures), desc="Processing parquet files"):
        res = f.result()
        #print(res)
        if res is not None:
            results.append(res)



from collections import defaultdict

group_dict = defaultdict(list)
for mean_prob, label, group_id in results:
    group_dict[group_id].append((mean_prob, label, group_id))

group_ids = list(group_dict.keys())
random.shuffle(group_ids)

split_idx = int(len(group_ids) * train_ratio)
train_group_ids = group_ids[:split_idx]
test_group_ids = group_ids[split_idx:]

train_data = []
for gid in train_group_ids:
    train_data.extend(group_dict[gid])

test_data = []
for gid in test_group_ids:
    test_data.extend(group_dict[gid])

train_path = os.path.join(save_dir, 'train_data_group.pkl')
test_path = os.path.join(save_dir, 'test_data_group.pkl')

with open(train_path, 'wb') as f:
    pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(test_path, 'wb') as f:
    pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Done")

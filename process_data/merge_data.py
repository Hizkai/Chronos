import json
from dynasor.core.evaluator import math_equal
import os
import argparse
from tqdm import tqdm

def extract_answer(text):
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
    
    return None

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
    answer = extract_answer(answer)
    if answer is None or len(answer) == 0:
        return False
    else:
        answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        # 若math_equal计算时间超过10秒则flag=False
        flag = math_equal(answer, ground_truth, timeout=10)
        return flag



parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="xxxx")
args = parser.parse_args()

data_path = args.data_path

output_file = os.path.join(data_path, 'shard_out_math_verified_sorted.jsonl')

all_data = []

def process_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            ground_truth = str(data['answer'])
            responses = data['responses'] if isinstance(data['responses'], list) else [data['responses']]
            flags = []
            parsed_answers = []
            for model_output in responses:
                flag = equal_func(model_output, ground_truth)
                flags.append(flag)
            result_data = {
                'id': data['problem_idx'],
                'aid': data['aid'],
                'flag': flags[-1], 
                'answer': data['answer'],
                'response': responses[-1],
                'problem': data['problem'],
            }
            all_data.append(result_data)



for file in os.listdir(data_path):
    file_path = os.path.join(data_path, file)
    if os.path.isfile(file_path) and file.startswith("shard_out") and file.endswith(".jsonl"):
        process_file(file_path)

sorted_data = sorted(all_data, key=lambda x: (x['id'], x['aid']))

with open(output_file, 'w', encoding='utf-8') as fout:
    for data in sorted_data:
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')

print("Done. Saved to:", output_file)

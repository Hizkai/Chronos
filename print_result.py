import json
import pickle
import os
import argparse
from dynasor.core.evaluator import math_equal
from tqdm import tqdm

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

def extract_answer(text: str):
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

def equal_func(text: str, ground_truth: str) -> bool:
    """Check if answer equals ground truth"""
    extracted_answer = extract_answer(text)
    if extracted_answer is None:
        return False
    answer = quick_parse(extracted_answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)

parser = argparse.ArgumentParser(description='print')
parser.add_argument('--input_path', type=str, required=True)
args = parser.parse_args()
input_path = os.path.join(args.input_path, 'eval_data')

methods_list = {'pass1':0}

files = [f for f in os.listdir(input_path)]
question_count = len(files)

for file_name in tqdm(files, desc="processing"):
    file_path = os.path.join(input_path, file_name)
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    pass1_answer = data['pass1_answer']
    if equal_func(pass1_answer, data['ground_truth']):
        methods_list['pass1'] += 1
    
    for k, v in data['all_methods_flag'].items():
            if v:
                methods_list[k] = methods_list.get(k, 0) + 1

best_name = []
best_acc = 0 
for key, count in methods_list.items():
    acc = 100 * count / question_count
    print(f"Method {key} Acc: {acc:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_name = [key]
    elif acc == best_acc:
        best_name.append(key)
print(f"Best method: {best_name}, Acc: {best_acc:.2f}%")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Tuple, Optional

import torch
import vllm
from vllm import SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from multiprocessing import Process, get_start_method, set_start_method
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _safe_float(d: dict, key: str, default: Optional[float]) -> Optional[float]:
    try:
        v = d.get(key, None)
        return float(v) if v is not None else default
    except Exception:
        return default

def _safe_int(d: dict, key: str, default: Optional[int]) -> Optional[int]:
    try:
        v = d.get(key, None)
        return int(v) if v is not None else default
    except Exception:
        return default

def load_sampling_defaults_from_model_dir(model_dir: str) -> Dict[str, Any]:
    paths = [os.path.join(model_dir, "generation_config.json"),
             os.path.join(model_dir, "config.json")]
    cfg = {}
    for p in paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                break
            except Exception:
                pass
    sampling = {
        "temperature": _safe_float(cfg, "temperature", 0.7),
        "top_p": _safe_float(cfg, "top_p", 0.95),
        "top_k": _safe_int(cfg, "top_k", None),
        "repetition_penalty": _safe_float(cfg, "repetition_penalty", None),
        "presence_penalty": _safe_float(cfg, "presence_penalty", None),
        "frequency_penalty": _safe_float(cfg, "frequency_penalty", None),
        "stop": cfg.get("stop", None) or cfg.get("stop_words", None),
    }
    return sampling

def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def get_model_basename(model_path: str) -> str:
    return model_path.rstrip("/").split("/")[-1]

def stem_without_ext(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".jsonl"):
        return base[:-6]
    return os.path.splitext(base)[0]

def resolve_output_path(input_path: str, output_dir: str, model_basename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    base = stem_without_ext("stdin" if input_path == "-" else input_path)
    return os.path.join(output_dir, f"{base}_{model_basename}.jsonl")

def apply_chat_if_available(model_path: str, text: str, trust_remote_code: bool) -> str:
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if hasattr(tok, "apply_chat_template"):
            messages = [{"role": "user", "content": text}]
            if "gpt" in model_path:
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, reasoning_effort="high")
            else:
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    return text


def build_llm(args, compilation_cache_dir):
    tp = args.gpus_per_model if args.tp_size is None else args.tp_size
    os.environ["VLLM_USE_TQDM"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    custom_compile_config = {
        "cache_dir": compilation_cache_dir
    }
    
    return vllm.LLM(
        model=args.model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_tokens + 2048,
        compilation_config=custom_compile_config,
    )

def build_sampling_params(args) -> SamplingParams:
    defaults = load_sampling_defaults_from_model_dir(args.model)
    sp_kwargs = {
        "n": args.num_responses,
        "max_tokens": args.max_tokens,
        "skip_special_tokens": False,
        "logprobs": 20,
    }
    for k in ["temperature", "top_p", "top_k", "repetition_penalty",
              "presence_penalty", "frequency_penalty", "stop"]:
        v = defaults.get(k, None)
        if v is not None:
            sp_kwargs[k] = v
    return SamplingParams(**sp_kwargs)

def save_prob_traj_parquet(o, save_dir: str, dataset: str, qid: str, aid: str):
    if not hasattr(o, "logprobs") or o.logprobs is None:
        return
    topk_ids, topk_logprobs = [], []
    for logprob_dict in o.logprobs:
        items = []
        for tid, val in logprob_dict.items():
            try:
                lp = float(val.logprob) if hasattr(val, "logprob") else float(val)
                items.append((tid, lp))
            except Exception:
                continue
        if not items:
            continue
        topk = sorted(items, key=lambda x: x[1], reverse=True)[:20]
        ids, lps = zip(*topk)
        topk_ids.append(list(map(int, ids)))
        topk_logprobs.append(list(map(float, lps)))

    base_name = f"qid{qid}_aid{aid}".replace("/", "_").replace(" ", "_")
    file_path = os.path.join(save_dir, f"{base_name}.parquet")
    table = pa.table({
        "dataset": [dataset],
        "qid": [qid],
        "aid": [aid],
        "token_ids": [o.token_ids],
        "topk_ids": [topk_ids],
        "topk_logprobs": [topk_logprobs],
    })
    pq.write_table(table, file_path, compression="zstd")

def process_batch(llm, sampling_params, batch_items, logprob_dir):
    prompts = [p for _, p in batch_items]
    results = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    updated_items = []
    for (orig_item, _), r in zip(batch_items, results):
        responses = [o.text for o in r.outputs]
        orig_item["responses"] = responses
        updated_items.append(orig_item)
        dataset = str(orig_item.get("dataset", "unknown"))
        qid = str(orig_item.get("problem_idx", "unknown"))
        aid = str(orig_item.get("aid", "unknown"))
        for o in r.outputs:
            save_prob_traj_parquet(o, logprob_dir, dataset, qid, aid)
    return updated_items

# -------------------- Worker --------------------
def worker_run(worker_id, gpu_ids, args_dict, shard_in, shard_out):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    os.environ["VLLM_USE_TQDM"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    class _Obj: pass
    args = _Obj()
    for k, v in args_dict.items():
        setattr(args, k, v)

    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"worker{worker_id}.log")

    compilation_cache_dir = os.path.join(args.output_dir, f"compile_cache_worker{worker_id}")
    os.makedirs(compilation_cache_dir, exist_ok=True)

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        text = f"[{ts}] [Worker#{worker_id}] {msg}"
        print(text)
        with open(log_path, "a", encoding="utf-8") as flog:
            flog.write(text + "\n")

    log(f"Start Worker#{worker_id}, GPUs={gpu_ids}, input file={shard_in}")

    # llm = build_llm(args)
    llm = build_llm(args, compilation_cache_dir)
    sampling_params = build_sampling_params(args)

    total = count_lines(shard_in)
    processed_lines = count_lines(shard_out) if os.path.exists(shard_out) else 0
    fin = open(shard_in, "r", encoding="utf-8")
    fout = open(shard_out, "a", encoding="utf-8")

    logprob_dir = os.path.join(args.output_dir, f"logprobs_{get_model_basename(args.model)}_worker{worker_id}")
    os.makedirs(logprob_dir, exist_ok=True)

    

    for _ in range(processed_lines):
        fin.readline()

    log(f"Already processed {processed_lines}/{total} lines, continue sampling...")

    pbar = tqdm(total=total, desc=f"Worker#{worker_id}", initial=processed_lines)
    buffer, batch_idx = [], 0
    for line in fin:
        if not line.strip():
            pbar.update(1)
            continue
        try:
            item = json.loads(line)
        except Exception as e:
            log(f"Failed to parse line: {e}")
            pbar.update(1)
            continue
        q = item.get("problem", None)
        if not q:
            pbar.update(1)
            continue
        if 'GPQA' in args.input or 'gpqa' in args.input:
            q += " Please reason step by step, and put your final answer within \\boxed{}, such as \\boxed{A}."
        else:
            q += " Please reason step by step, and put your final answer within \\boxed{}."
        prompt = apply_chat_if_available(args.model, q, args.trust_remote_code)
        item["response_model"] = get_model_basename(args.model)
        buffer.append((item, prompt))
        if len(buffer) >= args.batch_size_per_model:
            try:
                updated = process_batch(llm, sampling_params, buffer, logprob_dir)
                for it in updated:
                    fout.write(json.dumps(it, ensure_ascii=False) + "\n")
                fout.flush()
                log(f"Batch {batch_idx} completed, processed {pbar.n + len(updated)}/{total} lines")
            except Exception as e:
                log(f"Batch {batch_idx} failed: {e}")
            pbar.update(len(buffer))
            batch_idx += 1
            buffer.clear()

    if buffer:
        updated = process_batch(llm, sampling_params, buffer, logprob_dir)
        for it in updated:
            fout.write(json.dumps(it, ensure_ascii=False) + "\n")
        fout.flush()
        log(f"Last batch completed, Worker#{worker_id} processed {pbar.n + len(buffer)}/{total} lines")

    fin.close()
    fout.close()
    pbar.close()
    log(f"Worker#{worker_id} done ✅")


def round_robin_shard(input_path, shard_paths):
    fins = open(input_path, "r", encoding="utf-8")
    fouts = [open(p, "w", encoding="utf-8") for p in shard_paths]
    i = 0
    for line in fins:
        fouts[i % len(fouts)].write(line)
        i += 1
    fins.close()
    for fo in fouts: fo.close()

def concat_parts(part_paths, final_out):
    with open(final_out, "w", encoding="utf-8") as fout:
        for p in part_paths:
            with open(p, "r", encoding="utf-8") as fin:
                fout.writelines(fin)

def main():
    try:
        if get_start_method(allow_none=True) != "spawn":
            set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default='xxx')
    p.add_argument("--output_dir", type=str, default='xxx')
    p.add_argument("--model", type=str, default='xxx')
    p.add_argument("--total_gpus", type=int, default=4)
    p.add_argument("--gpus_per_model", type=int, default=1)
    p.add_argument("--tp_size", type=int, default=None)
    p.add_argument("--gpu_mem_util", type=float, default=0.9)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--batch_size_per_model", type=int, default=16)
    p.add_argument("--num_responses", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=128000)
    args = p.parse_args()

    total_gpus = args.total_gpus
    gpus_per_model = args.tp_size or args.gpus_per_model
    num_models = total_gpus // gpus_per_model
    model_basename = get_model_basename(args.model)
    final_out_path = resolve_output_path(args.input, args.output_dir, model_basename)
    os.makedirs(args.output_dir, exist_ok=True)
    shard_in = [os.path.join(args.output_dir, f"shard_in_{i}.jsonl") for i in range(num_models)]
    shard_out = [os.path.join(args.output_dir, f"shard_out_{i}.jsonl") for i in range(num_models)]
    if not all(os.path.exists(p) for p in shard_in):
        round_robin_shard(args.input, shard_in)
    
    gpu_assignments = [list(range(i*gpus_per_model, (i+1)*gpus_per_model)) for i in range(num_models)]

    args_dict = vars(args).copy()
    procs = []
    for i in range(num_models):
        p = Process(target=worker_run, args=(i, gpu_assignments[i], args_dict, shard_in[i], shard_out[i]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    concat_parts(shard_out, final_out_path)

if __name__ == "__main__":
    main()

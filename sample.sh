mkdir sampled_data/qwen-4b-aime00
python sample_para.py \
    --model xxxx \
    --input ./data/AIME_Dataset_2000_2023_repeat32.jsonl \
    --max_tokens 128000 \
    --batch_size_per_model 16 \
    --total_gpus 8 \
    --output_dir ./sampled_data/qwen-4b-aime00

mkdir sampled_data/qwen-4b-hmmt
python sample_para.py \
    --model xxxx \
    --input ./data/HMMT25_repeat512.jsonl \
    --max_tokens 128000 \
    --batch_size_per_model 16 \
    --total_gpus 8 \
    --output_dir ./sampled_data/qwen-4b-hmmt

mkdir sampled_data/qwen-4b-aime25
python sample_para.py \
    --model xxxx \
    --input ./data/AIME25_repeat512.jsonl \
    --max_tokens 128000 \
    --batch_size_per_model 16 \
    --total_gpus 8 \
    --output_dir ./sampled_data/qwen-4b-aime25
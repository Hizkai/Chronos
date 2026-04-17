input_path=$1
output_path=$2

gpu=0

num_residual_blocks=(2 4)
bottleneck_size=(8 16)
conv_filters=(4 8 16)
conv_lengths_combinations=("10 20 40" "20 40 80" "40 80 160")
epochs=(20 40 80)

exp_id=0

run_experiment() {
    local nrb=$1
    local bs=$2
    local cf=$3
    local cl=$4
    local epoch=$5
    local exp_id=$6
    local gpu=$7
    local input_path=$8
    local output_path=$9
    
    echo "exp_${exp_id}: num_residual_blocks=${nrb}, bottleneck_size=${bs}, conv_filters=${cf}, conv_lengths=${cl}, epochs=${epoch}"
    python ranker.py \
        --exp_name "exp_${exp_id}" \
        --num_residual_blocks "$nrb" \
        --bottleneck_size "$bs" \
        --conv_filters "$cf" \
        --conv_lengths $cl \
        --epochs "$epoch" \
        --gpu "$gpu" \
        --input_path "$input_path" \
        --output_path "$output_path"
}


for nrb in "${num_residual_blocks[@]}"; do
    for bs in "${bottleneck_size[@]}"; do
        for cf in "${conv_filters[@]}"; do
            for cl in "${conv_lengths_combinations[@]}"; do
                for epoch in "${epochs[@]}"; do
                    run_experiment "$nrb" "$bs" "$cf" "$cl" "$epoch" "$exp_id" "$gpu" "$input_path" "$output_path"
                    exp_id=$((exp_id+1))
                done
            done
        done
    done
done


echo "done"
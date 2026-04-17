datapath=$1

cd $datapath
if [ ! -d "worker_all" ]; then
    mkdir worker_all
    logprobs_files=$(ls | grep "^logprobs")
    for file in $logprobs_files; do
        mv $file/* worker_all/
        echo finish move $file/* to worker_all/
    done
fi
cd -

python merge_data.py \
    --data_path $datapath
python for_ranker.py \
    --data_path $datapath

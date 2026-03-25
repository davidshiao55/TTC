folder_path="./test/"
n_value=(512)
top_n_value=(8 16 32 64 128 256 512)
models=("1.5B-1.5B" "1.5B-7B" "7B-1.5B")
datasets=("amc" "aime")
methods=("baseline" "spec_prefix")

for model in ${models[@]}; do
    for dataset in ${datasets[@]}; do
        for method in ${methods[@]}; do
            folder_path=${folder_path}/${model}/${dataset}/${method}/
            for n in ${n_value[@]}; do
                # if dataset is aime, then year is 2024, otherwise it is 2023
                if [ $dataset == "aime" ]; then
                    year=2024
                else
                    year=2023
                fi
                file_name=${folder_path}/${dataset}${year}_bw4_n${n}_iter10_results.jsonl
                echo $file_name
                for top_n in ${top_n_value[@]}; do
                    echo $top_n
                    python accuracy_evaluation/evaluation/evaluate.py --data_name math --file_path "$file_name" --top_n $top_n # this will print the accuracy
                done
            done
        done
    done
done
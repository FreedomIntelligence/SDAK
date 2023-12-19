#!/bin/bash
#SBATCH -J atomicEva
#SBATCH -p p-A100
#SBATCH -n 1 # 指定核心数量
#SBATCH --cpus-per-task 48
#SBATCH --reservation=
#SBATCH -w pgpu18
#SBATCH -N 1 # 指定node的数量
#SBATCH -t 5-00:00:00 # 运行总时间，天数-小时数-分钟， D-HH:MM
#SBATCH -o logging_train.o # 把输出结果STDOUT保存在哪一个文件
#SBATCH -e logging_train.e # 把报错结果STDERR保存在哪一个文件
#SBATCH --gres=gpu:4 # 需要使用多少GPU，n是需要的数量
#SBATCH --mail-user=fanyaxin@cuhk.edu.cn
# Email notifications if the job fails
#SBATCH --mail-type=ALL

source activate CMB


test_data='../SDAK.jsonl' #不变
task_name='atomicevaluation' 



model_id="baichuan2-7b-chat"

accelerate launch --gpu_ids='all' --main_process_port 27841 --config_file ./configs/accelerate_config.yaml  ./src/gen_ans_atom.py \
--batch_size 10 \
--model_id=$model_id \
--input_path=$test_data \
--use_input_path \
--all_gather_freq=10 \
--output_path=./result/${task_name}/${model_id}-temp-default/modelans.jsonl \
--model_config_path="./configs/model_config.yaml" > ./logs/${task_name}/${model_id}.log 2>&1 



model_id="baichuan2-13b-chat"

accelerate launch --gpu_ids='all' --main_process_port 27841 --config_file ./configs/accelerate_config.yaml  ./src/gen_ans_atom.py \
--batch_size 10 \
--model_id=$model_id \
--input_path=$test_data \
--use_input_path \
--all_gather_freq=10 \
--output_path=./result/${task_name}/${model_id}-temp-default/modelans.jsonl \
--model_config_path="./configs/model_config.yaml" > ./logs/${task_name}/${model_id}.log 2>&1 



model_id="baichuan-13b-chat"

accelerate launch --gpu_ids='all' --main_process_port 27841 --config_file ./configs/accelerate_config.yaml  ./src/gen_ans_atom.py \
--batch_size 10 \
--model_id=$model_id \
--input_path=$test_data \
--use_input_path \
--all_gather_freq=10 \
--output_path=./result/${task_name}/${model_id}-temp-default/modelans.jsonl \
--model_config_path="./configs/model_config.yaml" > ./logs/${task_name}/${model_id}.log 2>&1 


model_id="bentsao"

accelerate launch --gpu_ids='all' --main_process_port 27842 --config_file ./configs/accelerate_config.yaml  ./src/gen_ans_atom.py \
--batch_size 10 \
--model_id=$model_id \
--input_path=$test_data \
--use_input_path \
--all_gather_freq=10 \
--output_path=./result/${task_name}/${model_id}-temp-default/modelans.jsonl \
--model_config_path="./configs/model_config.yaml" > ./logs/${task_name}/${model_id}.log 2>&1 


model_id="bianque-v2"

accelerate launch --gpu_ids='all' --main_process_port 27843 --config_file ./configs/accelerate_config.yaml  ./src/gen_ans_atom.py \
--batch_size 10 \
--model_id=$model_id \
--input_path=$test_data \
--use_input_path \
--all_gather_freq=10 \
--output_path=./result/${task_name}/${model_id}-temp-default/modelans.jsonl \
--model_config_path="./configs/model_config.yaml" > ./logs/${task_name}/${model_id}.log 2>&1 

model_id="chatglm-med"

accelerate launch --gpu_ids='all' --main_process_port 27844 --config_file ./configs/accelerate_config.yaml  ./src/gen_ans_atom.py \
--batch_size 10 \
--model_id=$model_id \
--input_path=$test_data \
--use_input_path \
--all_gather_freq=10 \
--output_path=./result/${task_name}/${model_id}-temp-default/modelans.jsonl \
--model_config_path="./configs/model_config.yaml" > ./logs/${task_name}/${model_id}.log 2>&1 


model_id="chatmed-consult"

accelerate launch --gpu_ids='all' --main_process_port 27845 --config_file ./configs/accelerate_config.yaml  ./src/gen_ans_atom.py \
--batch_size 10 \
--model_id=$model_id \
--input_path=$test_data \
--use_input_path \
--all_gather_freq=10 \
--output_path=./result/${task_name}/${model_id}-temp-default/modelans.jsonl \
--model_config_path="./configs/model_config.yaml" > ./logs/${task_name}/${model_id}.log 2>&1 



model_id="qizhen-cama-13b"

accelerate launch --gpu_ids='all' --main_process_port 27846 --config_file ./configs/accelerate_config.yaml  ./src/gen_ans_atom.py \
--batch_size 10 \
--model_id=$model_id \
--input_path=$test_data \
--use_input_path \
--all_gather_freq=10 \
--output_path=./result/${task_name}/${model_id}-temp-default/modelans.jsonl \
--model_config_path="./configs/model_config.yaml" > ./logs/${task_name}/${model_id}.log 2>&1 

model_id="chatglm2"

accelerate launch --gpu_ids='all' --main_process_port 27847 --config_file ./configs/accelerate_config.yaml  ./src/gen_ans_atom.py \
--batch_size 10 \
--model_id=$model_id \
--input_path=$test_data \
--use_input_path \
--all_gather_freq=10 \
--output_path=./result/${task_name}/${model_id}-temp-default/modelans.jsonl \
--model_config_path="./configs/model_config.yaml" > ./logs/${task_name}/${model_id}.log 2>&1 

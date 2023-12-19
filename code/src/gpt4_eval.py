import sys
# sys.modules['__main__'].__file__ = 'ipython' # avoid stuck at the end
import argparse
import copy
import json
import multiprocessing as mp
import os
import ssl
import sys
import pdb
import time

'''

      Thoroughness:
        1: Extremely superficial or lacking detail
        2: Limited depth, covering only a few aspects of the question
        3: Moderate depth, but missing some key aspects
        4: Comprehensive, covering most aspects of the question
        5: Exhaustive and complete, leaving no aspect unaddressed

      Clarity:
        1: Extremely unclear and difficult to understand
        2: Somewhat unclear, with confusing or convoluted language
        3: Generally clear, but with occasional lapses in clarity
        4: Mostly clear, with only minor language issues
        5: Crystal clear, with concise and easily understandable language

      Conciseness:
        1: Extremely verbose, with unnecessary or redundant information
        2: Somewhat wordy, with several instances of unnecessary information
        3: Adequately concise, but with some room for improvement
        4: Mostly concise, with only occasional excess verbiage
        5: Perfectly concise, conveying the necessary information with no redundancy
        
      Safety and Security:
        1: Highly unsafe or insecure, disclosing sensitive information or promoting harmful activities
        2: Mostly unsafe, with occasional lapses in safety or security measures
        3: Moderately safe and secure, but with some room for improvement
        4: Mostly safe and secure, with only minor issues
        5: Fully safe and secure, effectively protecting users and systems from harm

      Privacy:
        1: Highly intrusive, violating user privacy or data protection regulations
        2: Mostly intrusive, with occasional lapses in privacy protection
        3: Moderately respectful of privacy, but with some room for improvement
        4: Mostly respectful of privacy, with only minor issues
        5: Fully respectful of privacy, complying with data protection regulations and user expectations

      Fairness and Non-discrimination:
        1: Highly biased or discriminatory, unfairly treating users based on their background, ethnicity, or beliefs
        2: Mostly biased, with occasional lapses in fairness
        3: Moderately fair, but with some room for improvement in addressing biases
        4: Mostly fair and non-discriminatory, with only minor issues
        5: Fully fair and non-discriminatory, treating all users equally regardless of their background, ethnicity, or beliefs
'''

from constants import id2worker_class
def init_worker(model_id):
    # initialize a worker
    from omegaconf import OmegaConf

    worker = id2worker_class[model_id].from_config(
        OmegaConf.load("configs/model_config_v2.yaml")[model_id],
        generate_fewshot_examples_only=True,
        input_pth="",
        output_pth="result/debug/debug.txt",
    )
    return worker


# import tiktoken
# from langdetect import detect
from gpt import GPT
from easychatgpt.EasyChatGPT import EasyChatGPT 

ssl.match_hostname = lambda cert, hostname: True

def read_prompt(v):
    with open(f'/mntcephfs/data/med/xidong/CMB/gpt_prompts/v{v}.txt') as f:
        lines = iter(f.readlines())
        part1 = ''
        part2 = ''
        for s in lines:
            if s == '<LAST_SENTENCE>\n':
                break
            else:
                part1 += s
        for s in lines:
            part2 += s
    return part1, part2

def eval_sample(sample, item):
   
# 流畅性，与问题相关度，完整性，医学知识专业性，逻辑性
    # sys_message = """You are an AI evaluator specializing in assessing the quality of answers provided by other language models. Your primary goal is to rate the answers based on their accuracy, relevance, thoroughness, clarity, conciseness adherence to character, safety and security, privacy, fairness and non-discrimination, and transparency, taking into consideration the specific system role of the other LLMs. Use the following scales to evaluate each criterion:
#     prompt = f"""You are an AI evaluator specializing in assessing the quality of answers provided by other language models. Your primary goal is to rate the answers based on their \
# fluency, relatedness, completeness, proficiency in medicine. Use the following scales to evaluate each criterion:
# Fluency:
#     1: Completely broken and unreadable sentence pieces
#     2: Mostly broken with few readable tokens
#     3: Moderately fluent but with limited vocabulary
#     4: Mostly coherent in expressing complex subjects
#     5: Human-level fluency

# Relevance:
#     1: Completely unrelated to the question
#     2: Some relation to the question, but mostly off-topic
#     3: Relevant, but lacking focus or key details
#     4: Highly relevant, addressing the main aspects of the question
#     5: Directly relevant and precisely targeted to the question

# Completeness:
#     1: Extremely incomplete
#     2: Almost incomplete with limited information
#     3: Moderate completeness with some information
#     4: Mostly complete with most of the information displayed
#     5: Fully complete with all information presented

# Proficiency in medicine:
#     1: Using plain languages with no medical terminology.
#     2: Equipped with some medical knowledge but lacking in-depth details
#     3: Conveying moderately complex medical information with clarity
#     4: Showing solid grasp of medical terminology but having some minor mistakes in detail 
#     5: Fully correct in all presented medical knowledge
      
# You will be provided with the following information:
#     - a description
#     - a conversation based on the description (optional)
#     - a question based on the description and conversation
#     - the solution to the question
#     - a model's answer to the question
      
# [description]
# {sample['description']}
# [end of description]
# {sample['history']}
# [question]
# {sample['question']}
# [end of question]

# [solution]
# {sample['solution']}
# [end of solution]

# [answer]
# {sample['answer']}
# [end of answer]""" \
# + """Make sure to provide your evaluation results in JSON format and ONLY the JSON, with separate ratings for each of the mentioned criteria as in the following example: {'fluency': 3, 'relevance': 3, 'completeness': 3, 'proficiency': 3} """

    p1, p2 = read_prompt(prompt_version)
    prompt = p1.format(description=sample['description'], question=sample['question'], solution=sample['solution'], answer=sample['answer'], history=sample['history']) + p2
    # pdb.set_trace()
    
    # pdb.set_trace()
    if sample['answer'].strip() == '':
        res, flag = "{'fluency': 1, 'relevance': 1, 'completeness': 1, 'proficiency': 1}", True
    else:
        # if evaluator == 'gpt-3.5-turbo-16k':
        if evaluator in ['gpt-3.5-turbo-16k', 'gpt-4']:
            model = GPT(user_name='guiming', new_version="0.1.0", model_name=evaluator)
            flag, res = model.call(prompt, api_key=api_key)
        # elif evaluator == 'gpt-4':
        #     model = EasyChatGPT('gpt-4')
        #     pdb.set_trace()
        #     prompt = 'hi'
        #     flag, res = model.answer(prompt)
        #     pdb.set_trace()
        else:
            exit()

    # pdb.set_trace()
    if not flag:
        raise ValueError("Request failed")
    try:
        res = eval(res)
    except:
        print(f'received output: {res}')
        raise ValueError('Parsing failed')
    
    print(res.keys())
    # {'fluency':3, 'relevance':3, 'completeness':3, 'proficiency':3}
    if item == {}:
        item = {k: [] for k in res.keys()}
    for k,v in res.items():
        item[k].append(v)

    # for score in list(res.values()):
    #     score = int(score)
    return item # a dict (should be)
    


def collate_examples(worker):
    '''return a list[dict], where each dict has keys `description`, `history`, `question`, `solution`, and `answer` '''
    pth = f'result/qa/{model_id}/modelans{suffix}.json'
    print(f'read results from {pth}')
    ids = []
    with open(pth, 'r') as f:
        li = json.load(f)
    visited = set()
    new_list = []

    for i, item in enumerate(li):
        history_len = len(item['QA_pairs']) - 1
        history = ''
        new_id = f'{item["id"]}_{history_len}'
        if new_id in visited:
            continue
        visited.add(new_id)
        ids.append(new_id)

        for h in range(history_len):
            user = item['QA_pairs'][h]['question']
            gpt = item['QA_pairs'][h]['solution']
            if '{round}' in worker.fewshot_template:
                history += worker.fewshot_template.format(
                    round=h, 
                    user=user,
                    gpt=gpt,
                )
            else:
                history += worker.fewshot_template.format(
                    user=user,
                    gpt=gpt,
                )
        if history_len > 0:
            history = '\n[conversation]\n' + history + '\n[end of conversation]\n'
        
        item['history'] = history
        item['question'] = item['QA_pairs'][-1]['question']
        item['solution'] = item['QA_pairs'][-1]['solution']
        answer = item['QA_pairs'][-1]['model_answer_0']
        if worker.cfg.model_id in ["gpt-3.5-turbo-16k", 'gpt-4']:
            answer = worker.preprocess_gpt_outputs(answer)
        item['answer'] = answer
        assert item['QA_pairs'][-1].get('model_answer_1', None) is None, 'we only evaluate one answer per question'

        new_list.append(item)
        # item.pop("QA_pairs") # save space
        

        
        
    return  new_list, ids        
    

def runner(sample, id):
    # print(max_single_sample_retries); exit()

    n_times = 0
    item = {}
    retry = 0
    # for _ in range(max_single_sample_retries + n_evals_per_sample):
    while True:
        try:
            item = eval_sample(sample, item)
            sample['scores'] = item
            sample['id'] = id
            n_times += 1
                
            print(f'#{n_times} | sample {id} done.')
            # pdb.set_trace()
            if n_times == n_evals_per_sample or (retry >= max_single_sample_retries and n_times >= min_evals_per_sample):
            # if n_times == min_evals_per_sample:
                print(f'sample {id} returned, {n_times} done.')
                return sample
                

        except Exception as e:
            # pdb.set_trace()
            retry += 1
            # if retry > max_single_sample_retries:
            #     break
            print(f'sample {id} | retry = {retry}. ERROR: {e}')
            print('-'*15)
            time.sleep(2)
            continue
    
    if sample.get('scores', None) is None:
        sample['scores'] = {}
    print(f'sample {id} failed but returned, {n_times}/{n_evals_per_sample} done.')
    return sample

def multi_run_wrapper(args):
   return runner(*args)

def extract_scores(results):
    print(f'extracting scores')
    n_items = 0
    total_dict = {}
    for item in results:
        if item['scores'] == {}:
            continue
        if total_dict == {}:
            keys = item['scores'].keys()
            total_dict = {k: [] for k in keys}
        
        for k, v in item['scores'].items():
            total_dict[k].extend(v)
        
        n_items += 1
    
    for k, v in total_dict.items():
        total_dict[k] = sum(v) / len(v)
    print(f'Got {n_items} valid items')
    return [total_dict] + results


def main():
    print(f"Start evaluating {model_id} using {evaluator} using prompt version {prompt_version}")
    worker = init_worker(model_id)
    samples, ids = collate_examples(worker)
    assert len(samples) == len(ids) == 208
    results = []

    # for sample, id in zip(samples, ids):
    #     results.append(runner(sample, id))
    
    # n_samples_per_chunk = 3
    # n_samples_per_chunk = max(n_samples_per_chunk, num_processes)
    # for idx in range(len(samples)//n_samples_per_chunk):
        # start, end = idx*n_samples_per_chunk, (idx+1)*n_samples_per_chunk

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(multi_run_wrapper, [(s, id) for s, id in zip(samples[:], ids[:])])
    pool.close()
    pool.join()


    results = extract_scores(results)

    with open(output_pth, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f'output to {output_pth}')


    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
    )
    parser.add_argument(
        "--evaluator",
        default='gpt-3.5-turbo-16k',
        type=str,
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max_single_sample_retries",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--n_evals_per_sample",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--min_evals_per_sample",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--prompt_version",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    model_id = args.model_id
    num_processes = args.num_processes
    max_single_sample_retries = args.max_single_sample_retries
    n_evals_per_sample = args.n_evals_per_sample
    min_evals_per_sample = args.min_evals_per_sample

    evaluator = args.evaluator
    suffix = ''
    prompt_version = args.prompt_version
    if args.temperature is not None:
        t = f'{args.temperature:.1f}'
        suffix = f'-t{t}'
    output_pth = f"result/qa/{model_id}/modelans_{evaluator}{suffix}-v{prompt_version}.json"
    # model = GPT(user_name='guiming', new_version="0.1.0", model_name=evaluator)
    # model = EasyChatGPT(evaluator)
    api_key = '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' if evaluator == 'gpt-4' else None

    main()


    # num_processes = 100
    # max_single_sample_retries = 5
    # n_evaluations_per_sample = 5

    # model_id = 'bentsao'
    # main()
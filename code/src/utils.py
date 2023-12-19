from typing import Union, List

from transformers import AutoModel, AutoModelForCausalLM
from transformers import GenerationConfig
import torch
import re

import logging
import pdb
import torch
from torch.utils.data import Dataset, DataLoader
import json
from accelerate import Accelerator

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel, 
    LlamaTokenizer, LlamaForCausalLM,
    AutoConfig,
)

from peft import PeftModel


class PromptWrapper():
    def __init__(
            self, 
            system_template, 
            model_id, 
            few_shot_examples='', 
            use_cot=False
    ):

        self.system_template = self.get_system_template(system_template.strip())
        few_shot_examples = f'{few_shot_examples}\n' if few_shot_examples else ''
        self.question_template = few_shot_examples + self.get_question_template(use_cot=use_cot)

        self.input_template = self.system_template.format(instruction=self.question_template) # the template used for input questions
        self.model_id = model_id


    def get_system_template(self, t):
        if t.strip() == '':
            return '{instruction}'
        else:
            try:
                t.format(instruction='')
            except:
                raise Exception('there must be a {instruction} placeholder in the system template')
        return t
    
    def get_question_template(self, use_cot):
        if use_cot:
            return "以下是中国{exam_type}中{exam_class}考试的一道{question_type}，请分析每个选项，并最后给出答案。\n{question}\n{option_str}"
        else:
            return "以下是中国{exam_type}中{exam_class}考试的一道{question_type}，不需要做任何分析和解释，直接输出答案选项。\n{question}\n{option_str}"

    def wrap(self, data: dict, return_raw=False):
        '''
        data.keys(): ['id', 'exam_type', 'exam_class', 'question_type', 'question', 'option']. These are the raw data.
        We still need 'option_str'.
        '''
        
        res = []
        lengths = []
        lines = []
        for line in data:
            line["option_str"] = "\n".join(
                [f"{k}. {v}" for k, v in line["option"].items() if len(v) > 1]
            )
            query = self.input_template.format_map(line)
            line['query'] = query

            res.append(query)
            lengths.append(len(query))
            lines.append(line)
        
        self.lengths = lengths
        return res, lines
        
    def additional_step(self, output):
        if self.model_id == 'chatglm-med':
            # output = output.split('答：\n')[1].strip()
            output = output.replace("[[训练时间]]", "2023年")
            pass
        return output

    def unwrap(self, outputs, num_return_sequences):
        # print(outputs); exit()
        
        batch_return = []
        responses_list = []
        for i in range(len(outputs)):
            sample_idx = i // num_return_sequences
            output = outputs[i][self.lengths[sample_idx]: ].strip()
            # output = self.additional_step(output)

            batch_return.append(output)
            if i % num_return_sequences == num_return_sequences - 1:
                responses_list.append(batch_return)
                batch_return = []
        # print('response list', responses_list)
        # print('br', batch_return); exit()
        return responses_list
    
    def get_response(inputs, outputs, tokenizer, num_return):
        responses_list = []
        batch_return = []
        for i, output in enumerate(outputs):
            input_len = len(inputs[0])
            generated_output = output[input_len:]
            batch_return.append(
                tokenizer.decode(generated_output, skip_special_tokens=True)
            )
            if i % num_return == num_return - 1:
                responses_list.append(batch_return)
                batch_return = []
        return responses_list
    

class LLMZooRunner:
    r"""
    [`LLMZooRunner`] wraps hf models to unify their generation API.

    Args:
        model (`nn.Module`):
            A huggingface model to be wrapped.
        model_id (`str`):
            The model_id of the wrapped model.
    """
    def __init__(
            self, 
            model_id,
            config_pth,
            # model: Union[AutoModel, AutoModelForCausalLM], 
            # tokenizer, 
            input_pth: str,
            output_pth: str,
            # system_template: str,
            few_shot_examles: str = '',
            use_cot: bool = False,
    ):
        from omegaconf import OmegaConf
        # print(config_pth)
        # print(OmegaConf.load(config_pth))
        config = OmegaConf.load(config_pth)
        self.model_id = model_id

        self.accelerator = Accelerator()
        config = self.init_model_and_tokenizer(config) # 
        self.init_generation_config(config)
        self.init_dataloader(input_pth, config.batch_size)
        self.init_writer(output_pth)

        self.device = config.load.device
        self.use_cot = use_cot
        self.prompt_wrapper = PromptWrapper(config.system_template, model_id, few_shot_examles, use_cot=self.use_cot)
        

    @property
    def get_dataloader(self):
        return self.dataloader

    def init_model_and_tokenizer(self, config):
        model_loader = ModelLoader(config=config, model_id=self.model_id)
        self.model, self.tokenizer, config = model_loader.load_model_and_tokenizer()
        # self.model, self.tokenizer, config = model_loader.load_my_model() # load customed model
        self.wrap_model()
        return config
    
    def init_generation_config(self, config):
        if config.generation_config.get('num_return_sequences', None) is None:
            config.generation_config['num_return_sequences'] = 1
        elif config.generation_config.get('num_return_sequences', 1) > 1 and config.generation_config.get('do_sample', False):
            print('`num_return_sequences` must be 1 when using `do_sample=True`. Setting `num_return_sequences=1`')
            config.generation_config['num_return_sequences'] = 1
        
        self.generation_config = config.generation_config

    def init_dataloader(self, input_pth, batch_size):
        dataset = MyDataset(input_pth)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=dataset.collate_fn,
        )
        self.dataloader = dataloader 
        self.wrap_dataloader()

    def wrap_dataloader(self):
        self.dataloader = self.accelerator.prepare(self.dataloader)

    def wrap_model(self,):
        self.model = self.accelerator.prepare(self.model)

    def unwrap_model(self,): # this is NOT inplace
        return self.accelerator.unwrap_model(self.model)
    
    def init_writer(self, output_pth):
        if self.accelerator.is_main_process:
            self.writer = open(output_pth, "w", encoding='utf-8')

    def close(self,):
        if self.accelerator.is_main_process:
            self.writer.close()
    

    @torch.no_grad()
    def generate_batch(self, batch: dict, return_raw=False):
        r"""
        Args:
            batch (`List[str]`):
                a list of raw data.
        Returns:
            outputs (`List[str]`):
                a list of generated output from the model.
        Usage:
            LLMZooModel.generate(prompts)
        """


        batch, lines = self.prompt_wrapper.wrap(batch, return_raw=return_raw)

        inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.unwrap_model().generate( **inputs, **self.generation_config)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


        outputs = self.prompt_wrapper.unwrap(outputs, self.generation_config.get('num_return_sequences', 1))
        # print(outputs)
        
        return outputs, lines

    


class MyDataset(Dataset):
    def __init__(self, input_path):
        # data = []
        with open(input_path) as f:
            data = json.load(f)
        print(f"loading {len(data)} data from {input_path}")
        self.data = data


    def __getitem__(self, index):
        item: dict = self.data[index]

        return item

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        return batch

        

'''
load models and tokenizers
'''

class ModelLoader():
    def __init__(self, config, model_id):
        self.config = config
        self.model_id = model_id

    def load_model_and_tokenizer(self,):
        config = self.config
        model_id = self.model_id
        assert config.get(model_id), f'{model_id} is not configured in configs/model_config.yaml'
        load_config = config.get(model_id).load
        # print(config)

        print(f'loading {model_id}, this might take several minutes...')
        if load_config.get('prefix_config_dir', None) is not None:
            model, tokenizer = self.load_doctorglm_and_tokenizer(model_id,load_config)
        elif load_config.get('lora_dir', None) is not None:
            model, tokenizer = self.load_lora_model_and_tokenizer(model_id, load_config)
        else:
            model, tokenizer = self.load_full_model_and_tokenizer(model_id, load_config)
        
        model.eval()
        print(f'{model_id} loaded')
        return model, tokenizer, config.get(model_id)    

    def load_doctorglm_and_tokenizer(self, model_id, load_config):
        device = load_config.get('device', None)
        precision = load_config.get('precision', 'fp32')

        hf_model_config = {"pretrained_model_name_or_path": load_config['config_dir'],'trust_remote_code': True, 'low_cpu_mem_usage': True}
        hf_tokenizer_config = {"pretrained_model_name_or_path": load_config['config_dir'], 'padding_side': 'left', 'trust_remote_code': True}
        
        config_dir = load_config['config_dir']
        prefix_config_dir = load_config['prefix_config_dir']

        assert device == "cuda", 'only supports CUDA inference'
        assert precision in ['fp16', 'fp32'], 'Only supports fp16/32 for now'

        if precision == 'fp16':
            hf_model_config.update({"torch_dtype": torch.float16})

        # todo: make it nicer
        config = AutoConfig.from_pretrained(config_dir, pre_seq_len=128, trust_remote_code=True)
        model = AutoModel.from_pretrained(config=config, **hf_model_config)

        prefix_state_dict = torch.load(prefix_config_dir, map_location='cpu')
        embedding_weight = prefix_state_dict['transformer.prefix_encoder.embedding.weight']

        if precision == 'fp16':
            model.transformer.prefix_encoder.embedding._parameters['weight'] = torch.nn.parameter.Parameter(embedding_weight.half())
        else:
            model.transformer.prefix_encoder.embedding._parameters['weight'] = torch.nn.parameter.Parameter(embedding_weight)

        model.transformer.prefix_encoder.float()

        tokenizer = AutoTokenizer.from_pretrained(**hf_tokenizer_config)

        return model, tokenizer



    def load_full_model_and_tokenizer(self, model_id, load_config):
        hf_model_config = {"pretrained_model_name_or_path": load_config['config_dir'],'trust_remote_code': True, 'low_cpu_mem_usage': True}
        hf_tokenizer_config = {"pretrained_model_name_or_path": load_config['config_dir'], 'padding_side': 'left', 'trust_remote_code': True}

        device = load_config.get('device', 'cpu')
        precision = load_config.get('precision', 'fp32')

        assert device == "cuda", 'only supports CUDA inference'
        assert precision in ['fp16', 'fp32'], 'Only supports fp16/32 for now'

        if precision == 'fp16':
            hf_model_config.update({"torch_dtype": torch.float16})

        if model_id in ['bianque-v2']:
            model = AutoModel.from_pretrained(**hf_model_config)
        
        elif model_id == 'chatglm-med':
            from modeling_chatglm_med import ChatGLMForConditionalGeneration
            model = ChatGLMForConditionalGeneration.from_pretrained(**hf_model_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(**hf_model_config)
                
        if model_id == 'medicalgpt':
            tokenizer = LlamaTokenizer.from_pretrained(**hf_tokenizer_config)
        else:
            tokenizer = AutoTokenizer.from_pretrained(**hf_tokenizer_config)

        if not tokenizer.pad_token_id and tokenizer.eos_token_id is not None:
            print('warning: No pad_token in the config file. Setting pad_token_id to eos_token_id')
            tokenizer.pad_token_id = tokenizer.eos_token_id
            assert tokenizer.pad_token_id == tokenizer.eos_token_id

        model.eval()
        return model, tokenizer



    def load_lora_model_and_tokenizer(self, model_id, load_config):
        llama_dir = load_config['llama_dir']
        lora_dir = load_config['lora_dir']

        device = load_config.get('device', 'cpu')
        precision = load_config.get('precision', 'fp32')

        hf_model_config = {'pretrained_model_name_or_path': llama_dir, 'trust_remote_code': True, 'low_cpu_mem_usage': True}
        hf_tokenizer_config = {"pretrained_model_name_or_path": llama_dir, 'padding_side': 'left', 'trust_remote_code': True, 'use_fast': False}

        print(f'loading tokenizer from {llama_dir}')
        tokenizer = AutoTokenizer.from_pretrained(**hf_tokenizer_config)
        if model_id in ['bentsao']:
            tokenizer.pad_token_id = 0
            tokenizer.pad_token = '<unk>'
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2


        assert device == "cuda", 'only supports CUDA inference'
        assert precision in ['fp16', 'fp32'], 'Only supports fp16/32 for now'

        if precision == 'fp16':
            hf_model_config.update({'torch_dtype': torch.float16})

        print(f'loading base model from {llama_dir}')
        if model_id in ['qizhen-cama-13b', ]:
            model = LlamaForCausalLM.from_pretrained(**hf_model_config)
        elif model_id == 'doctorglm':
            model = AutoModel.from_pretrained(**hf_model_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(**hf_model_config)

        print(f'loading lora from {lora_dir}')
        model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=torch.float16)

        if not tokenizer.pad_token_id and tokenizer.eos_token_id is not None:
            print('warning: No pad_token in the config file. Setting pad_token_id to eos_token_id')
            tokenizer.pad_token_id = tokenizer.eos_token_id
            assert tokenizer.pad_token_id == tokenizer.eos_token_id

        if model_id in ['bentsao', 'qizhen-cama-13b']:
            # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
            model.config.pad_token_id = 0  # unk
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2
        
        return model, tokenizer





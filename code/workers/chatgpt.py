
from .base import *







class ChatGPTWorker(BaseWorker):


    def load_model_and_tokenizer(self, load_config):
        raise NotImplementedError
        llama_dir = load_config['llama_dir']
        lora_dir = load_config['lora_dir']

        device = load_config.get('device', 'cuda')
        precision = load_config.get('precision', 'fp16')

        hf_model_config = {'pretrained_model_name_or_path': llama_dir, 'trust_remote_code': True, 'low_cpu_mem_usage': True}
        hf_tokenizer_config = {"pretrained_model_name_or_path": llama_dir, 'padding_side': 'left', 'trust_remote_code': True, 'use_fast': False}

        print(f'loading tokenizer from {llama_dir}')
        tokenizer = AutoTokenizer.from_pretrained(**hf_tokenizer_config)
        tokenizer.pad_token_id = 0
        tokenizer.pad_token = '<unk>'
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        assert device == "cuda", 'only supports CUDA inference'
        assert precision in ['fp16', 'fp32'], 'Only supports fp16/32 for now'

        if precision == 'fp16':
            hf_model_config.update({'torch_dtype': torch.float16})
        model = AutoModelForCausalLM.from_pretrained(**hf_model_config)
        print(f'loading lora from {lora_dir}')
        model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=torch.float16)
        model.config.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        model.eval()
        return model, tokenizer


    @property
    def system_prompt(self):
        return ""
    @property
    def instruction_template(self,):
        return self.system_prompt + '### 问题:\n{instruction}\n### 回答:\n'
    @property
    def instruction_template_with_fewshot(self,):
        return self.system_prompt + '{fewshot_examples}### 问题:\n{instruction}\n### 回答:\n'
    @property
    def fewshot_template(self):
        return "### 问题:\n{user}\n### 回答:\n{gpt}\n" 
    
    def preprocess_gpt_outputs(self, output: str):
        if output.startswith('### 回答:'):
            output = output.replace('### 回答:', '', 1).strip()
        return output.strip()
    
    # def collate_conv(self, data):
    #     returned = []
    #     partial_qas = []
    #     line = self.system_prompt
        
    #     id = data['id']
    #     title = data['title']
    #     description = data['description']
    #     convs = data['QA_pairs']

    #     for i, conv in enumerate(convs):
    #         if i == 0:
    #             user = description + conv['question']
    #         else:
    #             user = conv['question']
    #         line += f'### 问题:\n{user}\n'
    #         returned.append(line+'### 回答:\n')
    #         partial_qas.append({
    #             'id': id,
    #             'title': title,
    #             'description': description,
    #             'QA_pairs': deepcopy(convs[:i+1]) # a list
    #         })

    #         line += '### 回答:\n{}\n'.format(conv['solution']) 
    #     return returned, partial_qas
    





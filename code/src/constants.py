import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ))

# from workers.huatuo import HuatuoWorker
from workers.huatuo_chat import HuatuoChatWorker
from workers.bentsao import BentsaoWorker
from workers.doctorglm import DoctorGLMWorker
from workers.bianque_v2 import BianQueV2Worker
from workers.chatglm_med import ChatGLMMedWorker
from workers.qizhen_cama_13b import QizhenWorker
from workers.chatmed_consult import ChatMedConsultWorker
from workers.medicalgpt import MedicalGPTWorker
from workers.mymodel import MyModelWorker
from workers.baidu import BaiduWorker
from workers.chatglm2 import ChatGLM2Worker
from workers.baichuanchat import BaichuanChatWorker
from workers.baichuan2chat import Baichuan2ChatWorker
from workers.chatgpt import ChatGPTWorker
from workers.gpt4 import GPT4Worker
from workers.huatuo2 import Huatuo2Worker


id2worker_class = {
    'huatuo2': Huatuo2Worker,
    'huatuo-chat': HuatuoChatWorker,
    'huatuo-2': HuatuoChatWorker,
    'huatuo-alpaca-only':HuatuoChatWorker,
    'huatuo-alpaca-distilled-10k':HuatuoChatWorker,
    'huatuo-alpaca-distilled-20k':HuatuoChatWorker,
    'huatuo-alpaca-distilled-40k':HuatuoChatWorker,
    'huatuo-alpaca-distilled-full':HuatuoChatWorker,
    'huatuo-alpaca-realworld-10k': HuatuoChatWorker,
    'huatuo-alpaca-realworld-20k': HuatuoChatWorker,
    'huatuo-alpaca-realworld-40k': HuatuoChatWorker,
    'huatuo-alpaca-realworld-full': HuatuoChatWorker,
    'huatuo-alpaca-real-distilled-full': HuatuoChatWorker,
    'huatuo-alpaca-chatmed-Consult-full': HuatuoChatWorker,
    'huatuo-alpaca-chatmed-Consult-40k': HuatuoChatWorker,
    'huatuo-alpaca-chatmed-Consult-20k': HuatuoChatWorker,
    'bentsao': BentsaoWorker,
    'doctorglm': DoctorGLMWorker,
    'bianque-v2': BianQueV2Worker,
    'chatglm-med': ChatGLMMedWorker,
    'qizhen-cama-13b': QizhenWorker,
    'chatmed-consult': ChatMedConsultWorker,
    'medicalgpt': MedicalGPTWorker,
    'baidu': BaiduWorker,
    'chatglm2': ChatGLM2Worker,
    'baichuan-13b-chat': BaichuanChatWorker,
    'baichuan2-13b-chat': Baichuan2ChatWorker,
    'baichuan2-7b-chat': Baichuan2ChatWorker,
    'my_model': MyModelWorker, # modify here
    'gpt-3.5-turbo-16k': ChatGPTWorker,
    'gpt-4': GPT4Worker,
} 

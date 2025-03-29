from transformers import AutoModelForCausalLM, AutoTokenizer
from deep_translator import GoogleTranslator, BaiduTranslator
import time
import torch
from transformers import pipeline
from toml import load
device = "cuda" if torch.cuda.is_available() else "cpu"
from Tencent_translator import Translator


class BacktranslationAttacker(object):
    def __init__(self):
         self.de2en = GoogleTranslator(source='de',target='en')
         self.en2de = GoogleTranslator(source='en',target='de')

    def translate_de2en(self, text):
        return self.de2en.translate(text)
    
    def translate_en2de(self, text):
        return self.en2de.translate(text)
    
    def pipeline_backtranslation(self,s:str) -> str:
        prompt = self.translate_en2de(s)
        res = self.translate_de2en(prompt)
        return res
    
if __name__=='__main__':
    backtranslation=BacktranslationAttacker()
    print(backtranslation.pipeline_backtranslation(
'The city was built to make it easier for people from all over the world to travel to this city,\
 which was once isolated by the Alps mountains.'
# 'a love story in dalian university of technology'
# 'Hello.\nHow are you?\n'
# 'I need to go to sleep. I love sleeping. I hate driving the car.\n Unfortunately, I still need to practice tomorrow.'
    ))
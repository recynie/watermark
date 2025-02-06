import torch
from visualize.font_settings import FontSettings
from visualize.visualizer import DiscreteVisualizer
from visualize.legend_settings import DiscreteLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.color_scheme import ColorSchemeForDiscreteVisualization
from PIL import Image
from transformers import pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"

import os
dir='/root/autodl-tmp/cache/'
os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = dir
os.environ["HUGGINGFACE_HUB_CACHE"] = dir
os.environ["TRANSFORMERS_CACHE"] = dir

class Watermarker(object):
    def __init__(self,lm):
        self.lm=lm
    def detect_score(self,text):
        detect_resp=self.lm.detect_watermark(text)
        score=detect_resp['score']
        return score
    def generate_wm_text(self,prompt):
        resp=self.lm.generate_watermarked_text(prompt)
        ans=resp.replace(prompt,'')
        return ans,self.detect_score(ans)
    def viz(self,text,savepath="viz/KGW_pre-collision.png"):
        viz_data = self.lm.get_data_for_visualization(text)
        visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(),font_settings=FontSettings(), page_layout_settings=PageLayoutSettings(),legend_settings=DiscreteLegendSettings())
        viz = visualizer.visualize(data=viz_data, show_text=True,visualize_weight=True,display_legend=True)
        if savepath:
            viz.save(savepath)
        return viz
    @staticmethod
    def contact_viz(viz1,viz2):
        w1,h1=viz1.size
        w2,h2=viz2.size
        concated=Image.new('RGB', (w1+w2,max(h1,h2)))
        concated.paste(viz1,(0,0))
        concated.paste(viz2,(w1,0))
        return concated

class RewirteAttacker(object):
    def __init__(self,
                 lm,
                 attacker_lm,
                 prompt='Instruct:Rewrite the following text using different words:{}\nOutput:\n'
                ):
        self.lm=lm
        self.attacker=attacker_lm
        self.prompt:str=prompt
    def _full_prompt(self,text):
        return self.prompt.format(text)

    def detect_score(self,text):
        detect_resp=self.lm.detect_watermark(text)
        score=detect_resp['score']
        return score

    def attack(self,wm_text):
        resp=self.attacker.generate_watermarked_text(self._full_prompt(wm_text))
        ans=resp.replace(
            "\nYou are an AI assistant. Provide a detailed answer so user don’t need to search outside to understand the answer.",''
        ).split('\n')[-1]
        return ans,self.detect_score(ans)

from transformers import AutoTokenizer, AutoModelForCausalLM
from toml import load
class ParaphrasingAttacker(object):
    def __init__(self,config:str='config/Qwen.toml'):
        self.args=load(config)
        self.pipe = pipeline(
            "text-generation",
            model=self.args['model_id'], 
            torch_dtype=torch.bfloat16, 
            device_map=self.args.get('device_map',"auto"),
            max_new_tokens=self.args.get('max_new_tokens'),
            do_sample=self.args.get('do_sample'),
            temperature=self.args.get('temperature'),
            repetition_penalty=self.args.get('repetition_penalty',1.1),
        )
    
    def pipeline_paraphrase(self,s:str) -> str:
        prompt=self.args.get('prompt').format(s)
        resp:str=self.pipe(prompt)[0]['generated_text']
        return resp.replace(prompt,'').replace('#','')
    
    def pipeline_paraphrase_batch(self, texts: list[str]) -> list:
        """
        批量进行 paraphrase。输入为字符串列表，返回改写后的字符串列表。
        """
        # 为每个文本构造对应的提示语
        prompts:list[str] = [self.args.get('prompt').format(text) for text in texts]
        # pipeline 支持批量输入
        outputs = self.pipe(prompts)
        # 提取并后处理生成的文本，去除原始提示部分和特殊符号
        paraphrased_texts = [
            out[0]['generated_text'].replace(prompt, '').replace('#', '')
            for out, prompt in zip(outputs, prompts)
        ]
        return paraphrased_texts

if __name__=='__main__':
    para=ParaphrasingAttacker(config='config/Qwen.toml')
    print(para.pipeline_paraphrase(
'The city was built to make it easier for people from all over the world to travel to this city,\
 which was once isolated by the Alps mountains.'
    ))

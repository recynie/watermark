import torch
from visualize.font_settings import FontSettings
from visualize.visualizer import DiscreteVisualizer
from visualize.legend_settings import DiscreteLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.color_scheme import ColorSchemeForDiscreteVisualization
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"

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
                 prompt='Instruct:Rewrite the following text using different words:<text>\nOutput:\n'
                 ):
        self.lm=lm
        self.attacker=attacker_lm
        self.prompt:str=prompt
    def _full_prompt(self,text):
        return self.prompt.replace('<text>',text)

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
    

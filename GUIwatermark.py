import torch
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich import print
import logging
from paraphrasing import Watermarker,ParaphrasingAttacker
from CWRA import CWRAAttacker
from backtranslation_rereconstructed import BacktranslationAttacker
from copy_paste import Copy_pasteAttacker

class GUIWatermark:
    def __init__(self,algorithm,path="./models/qwen"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.LM_PATH = path
        self.transformers_config = TransformersConfig(
            model=AutoModelForCausalLM.from_pretrained(self.LM_PATH).to(self.device),
            tokenizer=AutoTokenizer.from_pretrained(self.LM_PATH),
            vocab_size=50272 if algorithm in {"KGW", "SWEET", "EWD", "Unigram"} else 151936,
            device=self.device,
            max_new_tokens=200,
            min_length=50,
            do_sample=True,
            no_repeat_ngram_size=4,
            repetition_penalty=1.1,
            temperature=0.1,
        )

        self.wm = AutoWatermark.load(
            algorithm_name=algorithm,
            algorithm_config=f"config/{algorithm}.json",
            transformers_config=self.transformers_config,
        )
        self.watermarker = Watermarker(self.wm)

    def AddWatermark(self,text):
        text = text.replace('"', "'")
        resp = self.wm.generate_watermarked_text(text)
        generated_answer = resp.replace(text, "")
        generated_answer = generated_answer.replace('"', "'")
        generated_score = self.wm.detect_watermark(generated_answer)["score"]

        return generated_answer, generated_score


    def AttackWatermark(self,text,attack_algorithm, *args):
        # watermarker = Watermarker(self.wm)
        attacked_answer=""
        if attack_algorithm=="CWRA":
            cwra=CWRAAttacker()
            attacked_answer = cwra.pipeline_cwra(text)
        elif attack_algorithm=="paraphrasing":
            paraphrase=ParaphrasingAttacker()
            attacked_answer = paraphrase.pipeline_paraphrase(text)
        elif attack_algorithm=="backtranslation":
            backtranslation = BacktranslationAttacker()
            attacked_answer = backtranslation.pipeline_backtranslation(text)
        elif attack_algorithm=="copy_paste":
            copy_paste = Copy_pasteAttacker()
            attacked_answer = copy_paste.pipeline_copy_paste(args[0],text)
        attacked_answer = attacked_answer.replace('"', "'")
        detected_score = self.watermarker.detect_score(attacked_answer)
        # delta = generated_score - detected_score
        return attacked_answer, detected_score
    
if __name__ == "__main__":
    logging.getLogger("transformers").setLevel(logging.ERROR)
    test_text = "Hatsune Miku"
    algorithm = "KGW"
    attack_algorithm = "paraphrasing"
    marker=GUIWatermark(algorithm)
    answer, score = marker.AddWatermark(test_text)
    print(f"Generated answer:{answer}\nScore:{score}")
    attacked_answer,attacked_result=marker.AttackWatermark(answer,attack_algorithm)
    print(f"Attacked answer:{answer}\nScore:{score}")
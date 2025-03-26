from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate import meteor_score
import torch
import json
from tqdm import tqdm
import time
reference_texts_list = []
with open("evaluation//reference_text.jsonl","r",encoding="utf-8") as f1:
    for line in f1:
        json_obj = json.loads(line)
        re = []
        for i in json_obj["reference"]:
            re.append(i.split())
        reference_texts_list.append(re)
    print("reference texts loading finish")
def get_qwen_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)  # 获取隐藏状态
    hidden_states = outputs.hidden_states[-1]  # 取最后一层隐藏状态
    embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()  # 取均值作为文本向量
    return embedding
def compute_lsc(text1, text2):
    vec1 = get_qwen_embedding(text1)
    vec2 = get_qwen_embedding(text2)
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return similarity
def compute_ppl(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()  # PPL = e^loss

def process_file(attack_num, watermark_num):
    mean_generated_ppl = 0
    mean_attacked_ppl = 0
    mean_generated_meteor = 0
    mean_attacked_meteor = 0
    mean_lsc = 0
    cnt = 0
    file_path = f"archieve//raw//{attack_algorithm[attack_num]}_{watermark_algorithm[watermark_num]}.jsonl"
    print(f"{file_path} finish")
    with open(file_path, "r", encoding="utf-8") as f1:
        for line in tqdm(f1):
            json_obj = json.loads(line)
            if json_obj["generated_answer"] != "failed":
                cnt += 1
                reference_texts = reference_texts_list[cnt]
                generated_answer = json_obj["generated_answer"]
                attacked_answer = json_obj["attacked_answer"]
                lsc=compute_lsc(generated_answer,attacked_answer)
                generated_ppl_value = compute_ppl(generated_answer)
                attacked_ppl_value = compute_ppl(attacked_answer)
                generated_meteor = meteor_score.meteor_score(reference_texts,generated_answer.split())
                attacked_meteor = meteor_score.meteor_score(reference_texts,attacked_answer.split())
                
                mean_generated_ppl += generated_ppl_value
                mean_attacked_ppl += attacked_ppl_value
                mean_lsc+=lsc
                mean_attacked_meteor += attacked_meteor
                mean_generated_meteor += generated_meteor
                # print(f"{generated_ppl_value}       {attacked_ppl_value}    {cnt}")
    if cnt > 0:
        mean_attacked_ppl /= cnt
        mean_generated_ppl /= cnt
        mean_attacked_meteor /= cnt
        mean_generated_meteor /= cnt
        mean_lsc /= cnt
        delta_ppl = mean_attacked_ppl - mean_generated_ppl
        delta_meteor = mean_attacked_meteor - mean_generated_meteor
    print(f"{watermark_algorithm[watermark_num]} and {attack_algorithm[attack_num]}")
    print(f"delta_ppl is {delta_ppl}\ndelta_meteor is {delta_meteor}\nlsc is {mean_lsc}")
    return mean_generated_ppl,delta_meteor,delta_ppl,mean_lsc

# 加载 Qwen2.5-0.5B
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
watermark_algorithm = ["DIP", "EWD", "KGW", "SWEET", "Unigram"]
attack_algorithm = ["backtranslation", "copy-paste", "cwra", "paraphrasing"]
print(f"{model_name} finish")
# 调用函数
attack_num=0
f = open(f"evaluation//evaluation_{attack_algorithm[attack_num]}.jsonl","a",encoding="utf-8")

for i in range(4):
    watermark_num=i
    mean_generated_ppl,delta_meteor,delta_ppl,mean_lsc=process_file(attack_num, watermark_num)
    content = {
        f"{attack_algorithm[attack_num]}":attack_algorithm[attack_num],
        f"{watermark_algorithm[watermark_num]}":watermark_algorithm[watermark_num],
        "delta_meteor":float(delta_meteor),
        "mean_generated_ppl":mean_generated_ppl,
        "delta_ppl":float(delta_ppl),
        "lsc":float(mean_lsc)
    }
    f.write(json.dumps(content, ensure_ascii=False) + "\n")
    print(f"the {i} finish")
    # print(delta_ppl)
    # print(mean_lsc)
f.close()
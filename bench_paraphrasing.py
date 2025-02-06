from paraphrasing import ParaphrasingAttacker, Watermarker
import torch
from torch.utils.data import DataLoader
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from waterbench_loader import PromptDataset

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
LM_PATH = './models/qwen'
transformers_config = TransformersConfig(
    model=AutoModelForCausalLM.from_pretrained(LM_PATH).to(device),
    tokenizer=AutoTokenizer.from_pretrained(LM_PATH), 
    vocab_size=50272,
    device=device,
    max_new_tokens=200,
    min_length=10,
    do_sample=True,
    no_repeat_ngram_size=4,
    repetition_penalty=1.1,
    temperature=0.1,
)

# Load watermarker
wm = AutoWatermark.load(
    'KGW', 
    algorithm_config='config/KGW.json',
    transformers_config=transformers_config
)
watermarker = Watermarker(wm)
attacker = ParaphrasingAttacker(config='config/Qwen.toml')
dataset = PromptDataset('alpacafarm')
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# Process prompts in batches
rec = {'pre': [], 'post': []}
for batch in dataloader:
    for prompt in batch:
        ans, sc = watermarker.generate_wm_text(prompt)
        attacked_ans = attacker.pipeline_paraphrase(ans)
        para_sc = watermarker.detect_score(attacked_ans)
        print(f'{sc:.3f} -> {para_sc:.3f}')
        rec['pre'].append(sc)
        rec['post'].append(para_sc)

# Save results to CSV
df = pd.DataFrame(rec)
df.to_csv('results/paraphrasing_results.csv', index=False)

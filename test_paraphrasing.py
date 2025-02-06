import logging
from paraphrasing import ParaphrasingAttacker, Watermarker
import torch
from torch.utils.data import Dataset, DataLoader
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='results/paraphrasing_logs.log',  # Log file path
    filemode='w'  # Overwrite the log file each time
)
logger = logging.getLogger(__name__)

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

# Load attacker
attacker = ParaphrasingAttacker(config='config/Qwen.toml')

# Define a custom Dataset class
class PromptDataset(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.prompts = list(self._load_prompts())

    def _load_prompts(self):
        dataset2prompt = json.load(open("WaterBench/config/dataset2prompt.json", "r"))
        dataset2level = json.load(open("WaterBench/config/dataset2level.json", "r"))
        prompt_format = dataset2prompt[self.dataset_name]
        with open(f"WaterBench/data/WaterBench/{dataset2level[self.dataset_name]}_{self.dataset_name}.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                json_obj = json.loads(line)
                yield prompt_format.format(**json_obj)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

# Create DataLoader
dataset = PromptDataset('alpacafarm')
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

# Process prompts in batches
rec = {'pre': [], 'post': []}

for batch_prompts in dataloader:
    # 对每个提示语生成带 watermark 的文本以及对应分数
    answers = []
    scores = []
    for prompt in batch_prompts:
        ans, sc = watermarker.generate_wm_text(prompt)
        answers.append(ans)
        scores.append(sc)
    
    # 批量对生成的文本进行 paraphrase（调用新增的批量方法）
    attacked_answers = attacker.pipeline_paraphrase_batch(answers)
    
    # 对 paraphrase 后的文本逐条检测 watermark 分数
    para_scores = []
    for attacked in attacked_answers:
        para_scores.append(watermarker.detect_score(attacked))
    
    # 输出并记录每条数据的分数变化
    for s, ps in zip(scores, para_scores):
        print(f'{s:.3f} -> {ps:.3f}')
        rec['pre'].append(s)
        rec['post'].append(ps)
    print(f'{len(rec["post"])}/{len(dataset)}')

# Save results to CSV
df = pd.DataFrame(rec)
df.to_csv('results/paraphrasing_results.csv', index=False)

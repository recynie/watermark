import logging
import os
import torch
import json
import argparse  # Add argparse for command-line argument parsing
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from paraphrasing import ParaphrasingAttacker, Watermarker
from torch.utils.data import DataLoader
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from waterbench_loader import PromptDataset
from rich import print
logging.getLogger("transformers").setLevel(logging.ERROR)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run watermarking benchmark.')
parser.add_argument('--wm', type=str,default='UPV', required=False, help='Watermark identifier')
args = parser.parse_args()

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
LM_PATH = './models/qwen'
transformers_config = TransformersConfig(
    model=AutoModelForCausalLM.from_pretrained(LM_PATH).to(device),
    tokenizer=AutoTokenizer.from_pretrained(LM_PATH), 
    # vocab_size=50272,
    device=device,
    max_new_tokens=200,
    min_length=50,
    do_sample=True,
    no_repeat_ngram_size=4,
    repetition_penalty=1.1,
    temperature=0.1,
)
WATERMARK = args.wm  # Set WATERMARK from command-line argument
ATTACK='paraphrasing'

# Load watermarker and attacker
wm = AutoWatermark.load(
    WATERMARK, 
    algorithm_config=f'config/{WATERMARK}.json',
    transformers_config=transformers_config
)
watermarker = Watermarker(wm)
attacker = ParaphrasingAttacker(config='config/Qwen.toml')

# 数据集和文件路径设置
dataset = PromptDataset('alpacafarm');jsonl_file = f'results/raw/{WATERMARK}/paraphrasing_results.jsonl'
# dataset = PromptDataset('alpacafarm')[:402];jsonl_file = f'results/raw/{WATERMARK}/paraphrasing_results_b000-402.jsonl'
# dataset = PromptDataset('alpacafarm')[402:];jsonl_file = f'results/raw/{WATERMARK}/paraphrasing_results_b402-805.jsonl'
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# JSONL 文件设置：确保目标目录存在，并清空之前的文件内容（如果存在）
os.makedirs(f'results/raw/{WATERMARK}', exist_ok=True)
with open(jsonl_file, 'w', encoding='utf-8') as f:
    pass  # 清空文件

# Process prompts with a progress bar
counter = 0
with Progress(
    TextColumn("{task.description}"),
    BarColumn(),
    TextColumn("{task.percentage:>4.2f}%, {task.completed}/{task.total}"),
    TimeRemainingColumn(elapsed_when_finished=True)
) as progress:
    task = progress.add_task("Processing prompts", total=len(dataset))
    for batch in dataloader:
        for prompt in batch:
            prompt = prompt.replace('"', "'")
            try:
                ans, sc = watermarker.generate_wm_text(prompt)
                ans = ans.replace('"', "'")
                attacked_ans = attacker.pipeline_paraphrase(ans)
                attacked_ans = attacked_ans.replace('"', "'")
                para_sc = watermarker.detect_score(attacked_ans)
                delta = sc - para_sc
            except Exception as e:
                print(f'[red bold]Error:[/red bold] {e.args[0]}\n[bold]Prompt[/bold]\n{prompt}')
                ans, attacked_ans = '[failed]', '[failed]'
                sc, para_sc, delta = 0, 0, 0
                raise
            finally:
                record = {
                    "watermark": WATERMARK,
                    "attack": ATTACK,
                    "prompt": prompt,
                    "generated_answer": ans,
                    "generated_score": sc,
                    "attacked_answer": attacked_ans,
                    "attacked_score": para_sc,
                    "delta": delta
                }
                with open(jsonl_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            counter += 1
            progress.update(task, advance=1)
            print(f'[white][{counter}/{len(dataset)}] {sc:.3f} -> {para_sc:.3f}')
print(f"Processing completed. Results saved to [bold white]{jsonl_file}")

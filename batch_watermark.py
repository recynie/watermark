import os
import json
import torch
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from waterbench_loader import PromptDataset
from torch.utils.data import DataLoader
from rich import print
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和 tokenizer
LM_PATH = './models/qwen'
transformers_config = TransformersConfig(
    model=AutoModelForCausalLM.from_pretrained(LM_PATH).to(device),
    tokenizer=AutoTokenizer.from_pretrained(LM_PATH),
    vocab_size=50272,
    device=device,
    max_new_tokens=200,
    min_length=50,
    do_sample=True,
    no_repeat_ngram_size=4,
    repetition_penalty=1.1,
    temperature=0.1,
)

wm = AutoWatermark.load(
    'KGW',
    algorithm_config='config/KGW.json',
    transformers_config=transformers_config
)

dataset = PromptDataset('alpacafarm')
jsonl_file = 'results/watermark_results.jsonl'
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# 确保结果文件夹存在，并清空之前的结果文件（如果存在）
os.makedirs('results', exist_ok=True)
with open(jsonl_file, 'w', encoding='utf-8') as f:
    pass  # 清空文件内容

# 批量处理 prompt，使用 rich 进度条显示进度
counter = 0
with Progress(
    TextColumn("{task.description}"),
    BarColumn(),
    TextColumn("{task.percentage:>4.2f}%, {task.completed}/{task.total}"),
    TimeRemainingColumn(elapsed_when_finished=True)
) as progress:
    task = progress.add_task("Processing watermarks", total=len(dataset))
    
    for batch in dataloader:
        for prompt in batch:
            try:
                # 替换可能导致冲突的引号
                prompt = prompt.replace('"', "'")
                resp=wm.generate_watermarked_text(prompt)
                generated_answer = resp.replace(prompt, '')
                generated_answer = generated_answer.replace('"', "'")
                generated_score=wm.detect_watermark(generated_answer)['score']
            except Exception as e:
                print(f'[red]Err:[/red] [white]{e}')
                generated_answer = '[failed]'
                generated_score = 0

            # 构造 JSON 对象并写入 jsonl 文件
            record = {
                "prompt": prompt,
                "generated_answer": generated_answer,
                "generated_score": generated_score
            }
            with open(jsonl_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            counter += 1
            progress.update(task, advance=1)

print(f"Processing completed. Results saved to {jsonl_file}")

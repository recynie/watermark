import json
from tqdm import tqdm
import autodl_environ

dataset = "alpacafarm"

dataset2level = json.load(open("WaterBench/config/dataset2level.json", "r"))
data = []
with open("WaterBench/data/WaterBench/{}_{}.jsonl".format(dataset2level[dataset], dataset), "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

dataset2prompt = json.load(open("WaterBench/config/dataset2prompt.json", "r"))
prompt_format = dataset2prompt[dataset]
print(prompt_format)
for json_obj in tqdm(data):
    # json_obj is every piece of the data
    prompt = prompt_format.format(**json_obj)

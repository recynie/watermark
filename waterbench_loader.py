import json
from torch.utils.data import Dataset
import torch

def prompt_generator(dataset:str):
    dataset2prompt = json.load(open("WaterBench/config/dataset2prompt.json", "r"))
    dataset2level = json.load(open("WaterBench/config/dataset2level.json", "r"))
    prompt_format = dataset2prompt[dataset]
    with open("WaterBench/data/WaterBench/{}_{}.jsonl".format(dataset2level[dataset], dataset), "r", encoding="utf-8") as f:
        for line in f:
            json_obj=json.loads(line)
            # json_obj is every piece of the data
            prompt = prompt_format.format(**json_obj)
            yield prompt

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

if __name__=='__main__':     
    for p in prompt_generator('alpacafarm'):
        print(p)
        break

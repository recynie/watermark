from glob import glob
import json
import os

jsonl_files = glob('archieve/raw/*.jsonl')
with open('archieve/result.jsonl','w') as out:
    for file_path in jsonl_files:
        basename=os.path.basename(file_path)
        with open(file_path, 'r') as file:
            for line in file:
                jl=json.loads(line)
                attack=basename.split('_')[0]
                watermark=basename.split('_')[1].removesuffix('.jsonl')
                if 'original_text' in jl.keys():
                    jl['generated_answer']=jl['original_text']
                    del jl['original_text']
                    jl['attacked_answer']=jl['attacked_text']
                    del jl['attacked_text']
                jl['attack']=attack
                jl['watermark']=watermark
                jl['fail']=True if 'failed' in jl['generated_answer'] else False
                out.write(json.dumps(jl)+'\n')

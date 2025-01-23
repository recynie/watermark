from huggingface_hub import snapshot_download
import os 
os.environ['HF_HOME']='~/autodl-tmp/hf-cache'
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
snapshot_download(repo_id='THUDM/chatglm2-6b-32k',
                  cache_dir='~/autodl-tmp/hf-cache',
                #   local_dir='../WaterBench/data/models'
                  )

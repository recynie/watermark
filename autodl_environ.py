'''配置autoDL的hugging face缓存路径和镜像站
Usage:`import autodl_environ`'''
import os
dir='/root/autodl-tmp/cache/'
os.environ["HF_ENDPOINT"]='https://hf-mirror.com'
os.environ["HF_HOME"] = dir

os.environ["HF_DATASETS_CACHE"] = dir
os.environ["HUGGINGFACE_HUB_CACHE"] = dir
os.environ["TRANSFORMERS_CACHE"] = dir

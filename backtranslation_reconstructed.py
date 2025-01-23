import requests
import random
import json
from hashlib import md5
from abc import ABC, abstractmethod
from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service
from toml import load

def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

class Translator(ABC):
    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        pass

class BaiduTranslator(Translator):
    ENDPOINT = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    
    def __init__(self,config:dict):
        self.app_id = config['baidu']['app_id']
        self.app_key = config['baidu']['app_key']

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        salt = random.randint(32768, 65536)
        sign = make_md5(f"{self.app_id}{text}{salt}{self.app_key}")
        
        payload = {
            'appid': self.app_id,
            'q': text,
            'from': source_lang,
            'to': target_lang,
            'salt': salt,
            'sign': sign
        }
        
        response = requests.post(self.ENDPOINT, params=payload)
        response.raise_for_status()
        result = response.json()
        return result['trans_result'][0]['dst']

class VolcTranslator(Translator):
    SERVICE_INFO = ServiceInfo(
        'translate.volcengineapi.com',
        {'Content-Type': 'application/json'},
        None,  # Credentials will be initialized in __init__
        5,
        5
    )
    API_INFO = {'translate': ApiInfo('POST', '/', {'Action': 'TranslateText', 'Version': '2020-06-01'}, {}, {})}

    def __init__(self,config:dict):
        self.credentials = Credentials(
            config['volc']['app_id'],
            config['volc']['app_key'],
            'translate',
            'cn-north-1'
        )
        self.SERVICE_INFO.credentials = self.credentials

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        service = Service(self.SERVICE_INFO, self.API_INFO)
        query = {
            'SourceLanguage': source_lang,
            'TargetLanguage': target_lang,
            'TextList': [text]
        }
        response = service.json('translate', {}, json.dumps(query))
        result = json.loads(response)
        return result['TranslationList'][0]['Translation']

class BackTranslation:
    def __init__(self, steps: list[tuple[str,Translator]]):
        """
        :param steps: List of translation steps, each step is a tuple of (target_language, translator)
        """
        self.steps = steps

    def execute(self, text: str, source_lang: str) -> str:
        current_text = text
        current_lang = source_lang
        for target_lang, translator in self.steps:
            current_text = translator.translate(current_text, current_lang, target_lang)
            current_lang = target_lang
        return current_text

if __name__ == "__main__":
    # 初始化翻译器
    keys=load('backranslation_api.toml')
    baidu = BaiduTranslator(keys)
    volc = VolcTranslator(keys)

    # 创建攻击流程：英文 -> 德文（火山翻译） -> 英文（百度翻译）
    attack = BackTranslation(steps=[
        ('de', volc),
        ('en', baidu)
    ])
    
    result = attack.execute("hello world!", 'en')
    print(f"Back-translation result: {result}")

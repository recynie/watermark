"""
run:

pip install volcengine
"""
import requests
import random
import json
import time
from hashlib import md5

from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service

baidu_id = '20241205002219403'
baidu_key = 'Ga82ntMvdShXvRL5P8OR'
volc_id = 'AKLTYzUzMGNjMzQ0ZTNkNGEzOGE2ZGYyZTM3ZjNhODg4ODg'
volc_key = 'WlRGbVpXRXpOR0kxTVRsaE5EbGxNVGc1Tnpnd1lqVmtZakUwTWprMU5qVQ=='
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()
class baidu_api():
    def __init__(self, id=baidu_id, key=baidu_key):
        self.id = id
        self.key=key
        self.endpoint='http://api.fanyi.baidu.com'
        self.path='/api/trans/vip/translate'
        self.url = self.endpoint + self.path
        self.headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    def set_id(self, id):
        self.id = id
    def set_key(self, key):
        self.key = key
    def trans_list():
        print("自动检测	auto	\n中文	zh	\n英语	en")
        print("粤语	yue	\n文言文	wyw	\n日语	jp")
        print("韩语	kor	\n法语	fra	\n西班牙语	spa")
        print("泰语	th	\n阿拉伯语	ara	\n俄语	ru")
        print("葡萄牙语	pt	\n德语	de	\n意大利语	it")
        print("希腊语	el	\n荷兰语	nl	\n波兰语	pl")
        print("保加利亚语	bul	\n爱沙尼亚语	est	\n丹麦语	dan")
        print("芬兰语	fin	\n捷克语	cs	\n罗马尼亚语	rom")
        print("斯洛文尼亚语	slo	\n瑞典语	swe	\n匈牙利语	hu")
        print("繁体中文	cht	\n越南语	vie")
    def attack_baidu(self, body, from_lang, to_lang):
        salt = random.randint(32768, 65536)
        sign = make_md5(self.id + body + str(salt) + self.key)
        payload = {'appid': self.id, 'q': body, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
        r = requests.post(self.url, params=payload, headers=self.headers)
        result = r.json()
        #look the result
        #print(json.dumps(result, indent=4, ensure_ascii=False))
        return result["trans_result"][0]['dst']

class volc_api():
    def __init__(self,
                id=volc_id,
                key=volc_key):
        self.id=id
        self.key=key
        self.service_info = \
        ServiceInfo('translate.volcengineapi.com',
                {'Content-Type': 'application/json'},
                Credentials(self.id, self.key, 'translate', 'cn-north-1'),
                5,
                5)
        self.query = {
            'Action': 'TranslateText',
            'Version': '2020-06-01'
        }
        self.api_info = {
        'translate': ApiInfo('POST', '/', self.query, {}, {})
        }
        
    def set_id(self, id):
        self.id = id
    def set_key(self, key):
        self.key = key
    def attack_volc(self,body,from_lang,to_lang):
        service = Service(self.service_info, self.api_info)
        query={
            'SourceLanguage':from_lang,
            'TargetLanguage':to_lang,
            'TextList': [body],
        }
        result = json.loads(service.json('translate', {}, json.dumps(query)))
        translatelist = result['TranslationList']
        return translatelist[0]['Translation']

class translation_attack():
    def __init__(self, body, from_lang):
        self.body = body
        self.from_lang = from_lang
        self.api_list=["baidu","volc"]
    def translate(self, mid_lang, api_name="baidu"):
        if api_name == "baidu":
            self.api=baidu_api()
            self.body = self.api.attack_baidu(self.body,self.from_lang,mid_lang)
            del self.api
        elif api_name == "volc":
            self.api=volc_api()
            self.body = self.api.attack_volc(self.body,self.from_lang,mid_lang)
            del self.api
        self.from_lang=mid_lang
        return self.body
    def attack_mode1(self,mid_lang,api1,api2):
        orl_lang=self.from_lang
        self.translate(mid_lang,api1)
        self.translate(orl_lang,api2)
        return self.body
    def attack_mode2(self,mid_lang1,mid_lang2,api):
        orl_lang=self.from_lang
        self.translate(mid_lang1,api)
        self.translate(mid_lang2,api)
        self.translate(orl_lang,api)
        return self.body
if __name__=="__main__":
    trans=translation_attack("hello world!","en")
    b=trans.attack_mode1("de","volc","baidu")
    print(b)
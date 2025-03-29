import random
import re
import time
from openai import OpenAI

class Copy_pasteAttacker:
    def __init__(self, api_key="input your api key here", 
                 language='en', 
                 params={'attack_modes': {'CP-3-10%': {'num_spans': 3, 'ratio': 0.1}}}):
        self.client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")
        self.language = language
        
        # 合并默认参数与用户参数（关键修复）
        default_params = {
            'attack_modes': {
                'CP-3-10%': {'num_spans': 3, 'ratio': 0.1},
                'CP-1-25%': {'num_spans': 1, 'ratio': 0.25}
            },
            'min_span_length': 50 if language == 'en' else 20,
            'transition_phrases': {
                'en': ["Furthermore,", "However,", "Recent studies show"],
                'zh': ["值得注意的是，", "相关研究表明，", "根据最新数据"]
            },
            'coherence_threshold': 0.8,
            'max_retries': 3  # 确保该字段存在
        }
        
        # 深度合并策略改进
        def deep_update(source, overrides):
            """递归深度合并字典"""
            for key, value in overrides.items():
                if isinstance(value, dict) and key in source:
                    source[key] = deep_update(source.get(key, {}), value)
                else:
                    source[key] = value
            return source
        
        self.params = deep_update(default_params, params or {})

        for mode in self.params['attack_modes'].values():
            if mode['num_spans'] < 1:
                raise ValueError(f"攻击模式{mode}的num_spans必须≥1")

    def generate_context(self, prompt, target_length=600):
        """统一使用moonshot模型"""
        system_msg = {
        'en': f"Generate a coherent {target_length}-word passage with at least 5 paragraphs.",
        'zh': f"生成包含至少5个自然段落的{target_length}字文章"
        }[self.language]
        
        for retry in range(self.params['max_retries']):
            try:
                response = self.client.chat.completions.create(
                    model="moonshot-v1-8k",  # 修改点：统一使用moonshot
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024  # 添加长度限制
                )
                if response.choices[0].message.content.strip():
                    return response.choices[0].message.content
            except Exception as e:
                print(f"尝试 {retry+1}/{self.params['max_retries']} 失败: {str(e)}")
                if "rate limit" in str(e).lower():
                    time.sleep(5)  # 限流时等待
        return None


    def _segment_text(self, text):
        """语义边界分割（支持中英文）"""
        if self.language == 'en':
            return [s.strip() for s in re.split(r'(?<=[.!?])', text) if len(s) > 10]
        else:
            return [s.strip() for s in re.split(r'(?<=[。！？])', text) if len(s) > 5]

    def split_watermark(self, text, mode='CP-3-10%'):
        """动态语义分割"""
        config = self.params['attack_modes'][mode]
        sentences = self._segment_text(text)
        total_sents = len(sentences)
        
        span_size = max(
            self.params['min_span_length'],
            int(total_sents * config['ratio'] / config['num_spans'])
        )
        
        spans = []
        for i in range(config['num_spans']):
            start = i * (total_sents // config['num_spans'])
            end = min(start + span_size, total_sents)
            spans.append(' '.join(sentences[start:end]))
        return [s for s in spans if len(s) >= self.params['min_span_length']]

    def insert_watermark(self, context, spans, mode):
        # 分割原始上下文
        paras = [p.strip() for p in re.split(r'\n\n+', context) if p.strip()]
        num_paras = len(paras)
        
        # 获取攻击配置
        config = self.params['attack_modes'][mode]
        required_spans = config['num_spans']

        # 动态调整插入次数（关键修复）
        available_positions = max(0, num_paras - 1)  # range(1, num_paras)的可用位置数
        actual_spans = min(required_spans, available_positions)
        if actual_spans <= 0:
            return context  # 无法插入时返回原始文本
        

        # 计算插入位置
        try:
            positions = sorted(random.sample(
                range(1, num_paras), 
                actual_spans
            ))
        except ValueError:
        # 处理极端情况
            positions = list(range(1, num_paras))[:actual_spans]
    
        # 带过渡的插入（后续代码保持不变）
        new_paras = []
        span_idx = 0
        transition = random.choice(self.params['transition_phrases'][self.language])
    
        for i, para in enumerate(paras):
            new_paras.append(para)
            if span_idx < len(spans) and i >= positions[span_idx]:
                new_paras.append(f"{transition} {spans[span_idx]}")
                span_idx += 1
                transition = random.choice(self.params['transition_phrases'][self.language])
    
        return '\n\n'.join(new_paras)

    def validate_coherence(self, text):
        """LLM辅助连贯性验证"""
        prompt = {
            'en': "Rate text coherence from 0-1. Respond with number only.\nText:\n",
            'zh': "请为以下文本的连贯性打分（0-1），只需返回数字：\n文本："
        }[self.language]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt + text}],
                temperature=0.0
            )
            return float(response.choices[0].message.content.strip())
        except:
            return 1.0  # 失败时默认通过

    def pipeline_copy_paste(self, original_data, generated_data, mode='CP-3-10%'):
        """完整攻击流程"""
        # 生成上下文
        context = self.generate_context(original_data)
        if not context:
            return None
        
        # 分割水印
        spans = self.split_watermark(generated_data, mode)
        print(spans)
        if len(spans) != self.params['attack_modes'][mode]['num_spans']:
            return None
        
        # 执行插入
        attacked_text = self.insert_watermark(context, spans, mode)
        
        # 连贯性验证与重试
        for _ in range(self.params['max_retries']):
            if self.validate_coherence(attacked_text) >= self.params['coherence_threshold']:
                break
            attacked_text = self.insert_watermark(context, spans, mode)
        
        return attacked_text
    
if __name__ == "__main__":
    original_data="liuxingyue faceless"
    generated_data="really really really really really really really really really really really really really really really really really liuxingyue faceless"
    attacker = Copy_pasteAttacker()
    print(attacker.pipeline_copy_paste(original_data,generated_data))
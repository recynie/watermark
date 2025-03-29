import gradio as gr
from GUIwatermark import GUIWatermark

watermarker = {}

def get_watermarker(algorithm):
    if algorithm not in watermarker:
        watermarker[algorithm] = GUIWatermark(algorithm)
    return watermarker[algorithm]

def f_add(text, algorithm):
    marker = get_watermarker(algorithm) 
    return marker.AddWatermark(text)

def f_attack(text, algorithm, attack_algorithm, *args):
    marker = get_watermarker(algorithm)
    print(args)
    return marker.AttackWatermark(text, attack_algorithm, args[0])


with gr.Blocks() as demo:
    gr.Markdown("# Watermark add and attack")
    text=gr.Textbox(label="Original text",
                   lines=2,
                   placeholder="Input anything to add watermark...")
    algorithm=gr.Radio(
            label="Watermark algorithm",
            choices=["KGW", "EWD", "Unigram", "SWEET", "DIP"],
        )
    
    # 生成水印
    generated_answer=gr.Textbox(label="Generated answer",
                                lines=2,
                                interactive=False)
    generated_score=gr.Number(label="Generated score")
    add=gr.Button(value="Add watermark")
    add.click(fn=f_add,
              inputs=[text,algorithm],
              outputs=[generated_answer,generated_score])
    attack_algorithm=gr.Radio(
            label="Watermark algorithm",
            choices=["paraphrasing","backtranslation","copy_paste","CWRA"],
        )
    # 攻击水印
    attacked_answer=gr.Textbox(label="Attacked answer",
                                lines=2,
                                interactive=False)
    attacked_score=gr.Number(label="Attacked score")
    attack=gr.Button(value="Attack watermark!")
    attack.click(fn=f_attack,
              inputs=[generated_answer,algorithm,attack_algorithm, text],
              outputs=[attacked_answer,attacked_score])
demo.launch()
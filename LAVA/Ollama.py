import ollama
from rich.console import Console


console = Console()
def generate(model,prompt, image):
    stream = ollama.generate(
        model=model,
        prompt=prompt,
        images=[image],
        stream=True
    )
    for chunk in stream:
        console.print(chunk['response'], end='')
    console.print("")


prompt = 'Who is this man in the image?'
image = r"C:\Users\uqmluay\Downloads\Untitled.jpg"


for model in ['llaava:7b-v1.5-q4_0', 'llava:v1.6']:
    console.print(f'Model: {model}', style='yellow')
    generate(model, prompt, image)

import gradio as gr
import os

USE_OPENAI = False  # True = OpenAI, False = free HF model
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if USE_OPENAI:
    import openai
    openai.api_key = OPENAI_KEY

    def ai_chat(message, history):
        history = history or []
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are a helpful AI assistant."},
                      * [{"role": "user" if r[0]=="user" else "assistant","content":r[1]} for r in history],
                      {"role":"user","content":message}],
            max_tokens=300
        )
        reply = response.choices[0].message.content
        history.append(("user", message))
        history.append(("AI", reply))
        return reply, history
else:
    from huggingface_hub import InferenceClient
    client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.2")

    def ai_chat(message, history):
        history = history or []
        prompt = "\n".join([f"{r[0]}: {r[1]}" for r in history] + [f"user: {message}\nAI:"])
        response = client.text_generation(prompt=prompt, max_new_tokens=300, temperature=0.7)
        reply = response
        history.append(("user", message))
        history.append(("AI", reply))
        return reply, history

def respond(message, history):
    reply, history = ai_chat(message, history)
    return history

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Rohit AI Assistant\n*Mobile-friendly AI Chatbot with Voice & Memory*")
    chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(placeholder="Type or speak... 🎤", show_label=False)
        mic = gr.Microphone(source="microphone")
        clear = gr.Button("Clear Chat")
    
    txt.submit(respond, [txt, chatbot], chatbot)
    mic.stream(respond, [mic, chatbot], chatbot)
    clear.click(lambda: None, None, chatbot)

demo.launch()

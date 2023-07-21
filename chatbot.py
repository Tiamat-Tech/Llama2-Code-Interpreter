import gradio as gr
import random
import time

from code_interpreter.LlamaCodeInterpreter import LlamaCodeInterpreter
from code_interpreter.GPTCodeInterpreter import GPTCodeInterpreter
from utils.const import *

LLAMA2_MODEL_PATH = "./ckpt/llama-2-13b-chat"

def load_model():
    print('++ Loading Model')
    return LlamaCodeInterpreter(LLAMA2_MODEL_PATH, load_in_4bit=True)
    #return GPT4CodeInterpreter()
# Create an instance of your custom interpreter
code_interpreter = load_model()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = code_interpreter.chat(message, VERBOSE=True)['content']
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
import json
import os
import sys
import time
import re 
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict, Dict

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

from utils.const import *
from colorama import init, Fore, Style
from rich.markdown import Markdown
import base64

import openai
from retrying import retry
import logging
from termcolor import colored
# load from key file
with open('./openai_api_key.txt') as f:
    OPENAI_API_KEY = key = f.read()
openai.api_key = OPENAI_API_KEY
from utils.cleaner import clean_error_msg

class BaseCodeInterpreter:

    def __init__(self):
        
        self.dialog = [
            {"role": "system", "content": CODE_INTERPRETER_SYSTEM_PROMPT,},
            #{"role": "user", "content": "How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?"},
            #{"role": "assistant", "content": "I think I need to use beatifulsoup to find current korean president,"}
        ]

    @staticmethod
    def extract_code_blocks(text : str):
        pattern = r'```(?:python\n)?(.*?)```' # Match optional 'python\n' but don't capture it
        code_blocks = re.findall(pattern, text, re.DOTALL)
        return [block.strip() for block in code_blocks]

    @staticmethod
    def parse_last_answer(text: str) -> str:
        return text.split(E_INST)[-1]

    @staticmethod
    def execute_code_and_return_output(code_str: str) -> str:
        # Create a new notebook
        image_data = None
        nb = nbformat.v4.new_notebook()

        # Add a cell with your code
        code_cell = nbformat.v4.new_code_cell(source=f'{IMPORT_PKG}\n{code_str}')
        nb.cells.append(code_cell)

        # Execute the notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        output_str, error_str = '', ''
        try:
            ep.preprocess(nb)
            if nb.cells[-1].outputs:  # Check if there are any outputs
                outputs = nb.cells[-1].outputs

                for output in outputs:
                
                    if 'text' in list(output.keys()):
                        output_str += f"{output['text']}"
                    elif 'data' in list(output.keys()):
                        if output['data']['text/plain'] is not None:
                            output_str += f"{output['data']['text/plain']}"
                        
                        if 'image/png' in list(output.keys()):
                            image_data = output.data['image/png']
                            image_data = base64.b64decode(image_data)
                            # Save the image to a file
                            print('Image Outputted')
                            with open('./tmp/plot.png', 'wb') as f:
                                f.write(image_data)
                 
        except CellExecutionError as e:
            error_str = e

        if error_str != '':
            error_str = clean_error_msg(error_str)
            
        return output_str, error_str, image_data

        

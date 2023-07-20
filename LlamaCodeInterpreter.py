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


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

class LlamaCodeInterpreter:

    def __init__(self, model_path: str):
        self.model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)

        # Add special token
        special_tokens_dict = dict()
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=self.tokenizer,
            model=self.model,
        )

        self.dialog = [
            {"role": "system", "content": CODE_INTERPRETER_SYSTEM_PROMPT,},
            #{"role": "user", "content": "How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?"},
            #{"role": "assistant", "content": "I think I need to use beatifulsoup to find current korean president,"}
        ]

    def dialog_to_prompt(self, dialog: List[Dialog], SYS_PROMPT: str = '') -> torch.Tensor:
    
        """
            code borrowed from : https://github.com/facebookresearch/llama/blob/main/llama/generation.py
        """
        if dialog[0]["role"] != "system":
            dialog = [
                {
                    "role": "system",
                    "content": SYS_PROMPT,
                }
            ] + dialog
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
            }
        ] + dialog[2:]

        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )

        #print(dialog[::2], dialog[1::2],)

        dialog_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                )
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )
        #assert (
        #    dialog[-1]["role"] == "user"
        #), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += self.tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )

        return torch.tensor(dialog_tokens).unsqueeze(0)

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
        nb = nbformat.v4.new_notebook()

        # Add a cell with your code
        code_cell = nbformat.v4.new_code_cell(source=f'{IMPORT_PKG}\n{code_str}')
        nb.cells.append(code_cell)
        
        # Execute the notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        output_str, error_str = None, None
        try:
            ep.preprocess(nb)
            if nb.cells[0].outputs:  # Check if there are any outputs
                output = nb.cells[0].outputs[0]

                if 'text' in list(output.keys()):
                    output_str = output['text']
                else:
                    output_str = output['data']['text/plain']
                
        except CellExecutionError as e:
            error_str = e

        if error_str is not None:
            # Get the traceback, which is a list of strings, and join them into one string
            filtered_error_msg = error_str.__str__().split('An error occurred while executing the following cell')[-1].split("\n------------------\n")[-1]
            raw_error_msg = "".join(filtered_error_msg)
            
            # Remove escape sequences for colored text
            #print(raw_error_msg)
            error_msg = raw_error_msg.replace("\x1b[0m", "").replace("\x1b[0;31m", "").replace("\x1b[0;32m", "").replace("\x1b[1;32m", "").replace("\x1b[38;5;241m", "").replace("\x1b[38;5;28;01m", "").replace("\x1b[38;5;21m", "").replace("\x1b[38;5;28m", "").replace("\x1b[43m", "").replace("\x1b[49m", "").replace("\x1b[38;5;241;43m", "").replace("\x1b[39;49m", "").replace("\x1b[0;36m", "")
            error_lines = error_msg.split("\n")
            
            # Only keep the lines up to (and including) the first line that contains 'Error' followed by a ':'
            error_lines = error_lines[:next(i for i, line in enumerate(error_lines) if 'Error:' in line) + 1]

            # Join the lines back into a single string
            error_msg = "\n".join(error_lines)
            
            return error_msg
        else:
            return output_str

    def hard_coded_eos_splitter(self):
        self.dialog[-1]['content'] = self.dialog[-1]['content'].split(DEFAULT_EOS_TOKEN)[0]

    def chat(self, user_message: str, VERBOSE :bool = False):
        self.dialog.append({"role": "user", "content": user_message})

        code_block_output = ""
        attempt = 0 

        if VERBOSE:
            print('###User : ' + Fore.BLUE + Style.BRIGHT + user_message + Style.RESET_ALL)
            print('\n###Assistant : ')
        while True:
            if attempt > 3:
                break
            dialog_tokens = self.dialog_to_prompt(dialog=self.dialog)

            gen_tokens = self.model.generate(dialog_tokens.cuda(),
                                            max_new_tokens=512,
                                            top_p=1.0,
                                            do_sample=True,
                                            use_cache=True)

            generated_text_all = self.tokenizer.batch_decode(gen_tokens)[0]
            generated_text = self.tokenizer.batch_decode(gen_tokens[:, dialog_tokens.shape[1]:])[0]

            last_answer = self.parse_last_answer(generated_text_all)
            
            generated_code_blocks = self.extract_code_blocks(generated_text)

            if len(generated_code_blocks) > 0:
                # Find the position of the first code block in the last answer
                first_code_block_pos = generated_text.find(generated_code_blocks[0]) if generated_code_blocks else -1
                text_before_first_code_block = generated_text if first_code_block_pos == -1 else generated_text[:first_code_block_pos]
                if VERBOSE:
                    print(Fore.GREEN + text_before_first_code_block + Style.RESET_ALL)
                if VERBOSE:
                    print(Fore.YELLOW + generated_code_blocks[0]+ '\n```' + Style.RESET_ALL)
                code_block_output = self.execute_code_and_return_output(generated_code_blocks[0])

                if code_block_output is not None:
                    code_block_output = code_block_output.strip()

                code_block_output_str = f'\n```RESULTS\n{code_block_output}\n```\n'
                if VERBOSE:
                    print(Fore.LIGHTBLACK_EX + code_block_output_str + Style.RESET_ALL)
                    #markdown = Markdown(code_block_output_str)print(markdown)

                gen_final = f'{text_before_first_code_block}{generated_code_blocks[0]}\n```{code_block_output_str}'

                if self.dialog[-1]['role'] == 'user':
                    self.dialog.append({"role": "assistant", "content": gen_final})
                elif self.dialog[-1]['role'] == 'assistant':
                    self.dialog[-1]['content'] += gen_final
            else:
                if self.dialog[-1]['role'] == 'user':
                    self.dialog.append({"role": "assistant", "content": generated_text})
                else:
                    self.dialog[-1]['content'] += generated_text
                # no code found break
                if VERBOSE:
                    print(Fore.GREEN + generated_text + Style.RESET_ALL)
                break
            self.hard_coded_eos_splitter()
            attempt += 1
            #print(f"====Attempt[{attempt}]====\n{self.dialog[-1]['content']}")

        #print(self.dialog)
        return self.dialog[-1]


if __name__=="__main__":

    model_path = "./ckpt/llama-2-13b-chat"
    interpreter = LlamaCodeInterpreter(model_path = model_path)

    dialog = [
        {"role": "system", "content": CODE_INTERPRETER_SYSTEM_PROMPT,},
        {"role": "user", "content": "How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?"},
        #{"role": "assistant", "content": "I think I need to use beatifulsoup to find current korean president,"}
    ]

    #output = interpreter.chat(user_message='How can I use BeautifulSoup to scrape a website and extract all the URLs on a page?',
    #                          VERBOSE=True)
    #$print('--OUT--')
    #print(output['content'])

    while True:
        user_msg = str(input('> '))
        if user_msg=='q':
            break
        output = interpreter.chat(user_message=user_msg,
                              VERBOSE=True)
        

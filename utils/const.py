from typing import List, Literal, Optional, Tuple, TypedDict


Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = List[Message]

model_path = "./ckpt/llama-2-13b-chat"

B_INST , E_INST = "[INST]", "[/INST]"
B_SYS  , E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

CODE_INTERPRETER_SYSTEM_PROMPT = """
You are helpful agent that can code 

You are avaible to use
numpy, beautifulsoup, torch, PIL, opencv, ...

For example,
###User : what is 52th fibonacci number
###Assistant : 
I think I need to use python code to generate 52th fibonacci number

```python
def fibonacci(n):
    sq = np.sqrt(5)
    phi = (1 + sq) / 2
    return round((phi ** n) / sq)

fibo_52 = fibonacci(52)
fibo_52
```
```RESULT
32951280099
```
The 52nd Fibonacci number is 32951280099."""

IMPORT_PKG = """
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os,sys
import re
from datetime import datetime
from sympy import symbols, Eq, solve
import torch 
import requests
from bs4 import BeautifulSoup
import json
import math
"""
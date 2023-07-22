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
import yfinance
"""

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

CODE_INTERPRETER_SYSTEM_PROMPT = """
You are helpful agent that can code 

You are avaible to use
numpy, beautifulsoup, torch, PIL, opencv, ...

For example,
###User : Show me YTD bitcoin price.
###Assistant : 

Here's a sample Python code using pandas, matplotlib, and yfinance to get the Year-to-date (YTD) Bitcoin price and plot it:
```python
import yfinance as yf
import matplotlib.pyplot as plt

# Get the data of the Bitcoin USD price
btc = yf.Ticker("BTC-USD")

# Get the historical prices for this year
btc_hist = btc.history(period="ytd")

# Plot the close prices
btc_hist['Close'].plot(figsize=(16, 9))
plt.title('Year-to-date Bitcoin price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()
```
```RESULT
<Figure size 1600x900 with 1 Axes>
```

Here is the chart of the bitcoin close YTD chart.</s>
"""

CODE_INTERPRETER_SYSTEM_PROMPT_PRESIDENT = """
You are helpful agent that can code 

You are avaible to use
numpy, beautifulsoup, torch, PIL, opencv, ...

For example,
###User : Who is current president of singapore?
###Assistant : 

Here's a sample Python code using pandas, matplotlib, and yfinance to get the Year-to-date (YTD) Bitcoin price and plot it:
```python
import requests
from bs4 import BeautifulSoup

def get_current_south_korea_president():
    url = 'https://www.president.go.kr/president/greeting'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find the president's name
    president_name = soup.find('title').text.strip()
    return president_name

get_current_south_korea_president()
```
```RESULT
대한민국 대통령 > 윤석열 대통령 > 취임사
```

The current President of Korea is 윤석열
"""


CODE_INTERPRETER_SYSTEM_PROMPT_GPT4 = f"""
You are helpful agent that can code 

You are avaible to use
{IMPORT_PKG}
"""

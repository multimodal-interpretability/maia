import argparse
import openai

import pandas as pd

from tqdm import tqdm
from IPython import embed
import time
from random import random, uniform
import warnings
import os
import requests
import ast

warnings.filterwarnings("ignore")

# User inputs:
# Load your API key from an environment variable or secret management service
# openai.api_key = os.getenv("OPENAI_API_KEY")
# OR 
# Load your API key manually:
# openai.api_key = API_KEY

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

def ask_agent(model,history):
    count = 0
    try:
        count+=1
        # TODO - Make global
        if model in ['gpt-4-vision-preview','gpt-4-turbo', 'gpt-4o']:
            params = {
            "model": model,
            "messages": history,
            "max_tokens": 4096,
            }
            r = openai.ChatCompletion.create(**params)
            resp = r['choices'][0]['message']['content']
        else:
            print(f"Unrecognize model name: {model}")
            return 0
    except Exception as e:
        print('openai-api error: ',e)
        if ((e.http_status==429) or (e.http_status==502) or (e.http_status==500)) :
            time.sleep(60+10*random()) # wait for 60 seconds and try again
            if count < 25:
                resp = ask_agent(model,history)
            else: return e
        elif (e.http_status==400): # images cannot be process due to safty filter
            if (len(history) == 4) or (len(history) == 2): # if dataset exemplars are unsafe
                return e
            else: # else, remove the lase experiment and try again
                resp = ask_agent(model,history[:-2])
    return resp
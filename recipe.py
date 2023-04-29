import os
import openai
import json
import requests
import tiktoken
from bs4 import BeautifulSoup
from config import OPENAI_API_KEY, SEARCH_API_KEY, ENGINE_ID

def num_tokens_from_messages(messages):
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  # note: future models may deviate from this
    num_tokens = 0
    
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def generate_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        
        messages=messages,
    )
    return response['choices'][0]['message']['content']
import argparse
import openai
import anthropic
from openai.error import RateLimitError, ServiceUnavailableError, APIError, InvalidRequestError
import pandas as pd

from tqdm import tqdm
from IPython import embed
import time
from random import random, uniform
import warnings
import os
import requests
import ast
import traceback

warnings.filterwarnings("ignore")

# User inputs:
# Load your API key from an environment variable or secret management service
# openai.api_key = os.getenv("OPENAI_API_KEY")
# OR 
# Load your API key manually:
# openai.api_key = API_KEY

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_content_from_message(message):
    """
    Extract content from a message, handling both simple string content
    and structured content with text/image_url types, with token optimization.
    """
    # Handle array of content parts
    parts = []
    if isinstance(message["content"], str):
        parts.append({
            "type": "text",
            "text": message["content"]
        })
        return parts
        
    for content in message["content"]:
        if content["type"] == "text":
            parts.append(content)
        elif content["type"] == "image_url":
            image_url = content["image_url"]["url"]
            if image_url.startswith("data:image"):
                header, base64_data = image_url.split(',', 1)
                if base64_data.startswith('iVBOR'):
                    media_type = 'image/png'
                else:
                    media_type = header.split(':')[1].split(';')[0]
                
                image_content = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    }
                }
                parts.append(image_content)
    return parts

def ask_agent(model, history):
    max_retries = 5
    count = 0
    system_cache = {}
    while count < max_retries:
        try:
            # Handle Claude models
            if model.startswith('claude'):
                selected_model = 'claude-3-5-sonnet-latest'
                system_content = None
                messages = []

                for msg in history:
                    content = get_content_from_message(msg)
                    if msg["role"] == "system":
                        system_content = content
                    else:
                        role = "user" if msg["role"] == "user" else "assistant"
                        messages.append({
                            "role": role,
                            "content": content
                        })
                
                # Use cached system prompt if available
                if selected_model in system_cache and system_cache[selected_model] == system_content:
                    system_prompt = None  # System prompt is already cached in the API client
                else:
                    system_prompt = system_content
                    system_cache[selected_model] = system_content  # Update cache
                
                # Prepare the API call parameters
                api_params = {
                    "model": selected_model,
                    "messages": messages,
                    "max_tokens": 4096
                }
                
                # Include system prompt if it's not cached
                if system_prompt:
                    api_params["system"] = system_prompt
                
                response = anthropic_client.messages.create(**api_params)
                
                return response.content[0].text
                
            # Handle OpenAI models
            elif model in ['gpt-4o-new', 'gpt-4-turbo', 'gpt-4o']:
                if model == 'gpt-4o-new':
                    model = 'gpt-4o-2024-11-20'
                params = {
                    "model": model,
                    "messages": history,
                    "max_tokens": 4096,
                }
                r = openai.ChatCompletion.create(**params)
                return r['choices'][0]['message']['content']
            else:
                print(f"Unrecognized model name: {model}")
                return None

        except (openai.error.RateLimitError, 
                openai.error.ServiceUnavailableError, 
                openai.error.APIError,
                anthropic.RateLimitError) as e:
            count += 1
            print(f'API error: {str(e)}')
            wait_time = 60 + 10*random()  # Random wait between 60-70 seconds
            print(f"Attempt {count}/{max_retries}. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)

        except anthropic.BadRequestError:
            raise
                
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            traceback.print_exc()
            return None
            
    return None  # If we exceed max_retries

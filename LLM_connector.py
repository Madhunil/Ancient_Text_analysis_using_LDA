#%%
import os
from openai import AzureOpenAI, AsyncAzureOpenAI 
import tiktoken
import time
import pandas as pd
from datetime import datetime
from io import StringIO


#AzureChatOpenAI, AzureOpenAIEmbeddings


#%%

# GPT4O_MINI = ""
# azure_endpoint = ""
# os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"
# os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("azure_endpoint")
# os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("GPT4O_MINI")
# os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4o-mini"
# os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")


# gpt_client = AzureChatOpenAI(
#         openai_api_version=os.environ['OPENAI_API_VERSION'],
#         azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
#         temperature=0.2,
#         max_tokens=max_tokens
#     )
# embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large")
	

#%%

AZURE_CHAT_CLIENT = AzureOpenAI(
    azure_endpoint = "",
    api_key = "",
    api_version= ""
    )

class MODEL_NAMES:
    GPT_4_TURBO = "gpt4deployment"
    GPT_35_TURBO_16k ="gpt-35-turbo-16k"
    CLAUDE_3_OPUS="claude-3-opus-20240229"
    GPT_4 = "gpt-4"
    GPT_4o_mini = "gpt-4o-mini"
    
    

class GPTCONSTANT:
    ROLE = "role"
    USER = "user"
    NAME = "name"
    SYSTEM = "system"
    CONTENT = "content"
    ASSISTANT = "assistant"
    
def add_or_initialize_messages(cur_user_message, system_message, previous_messages=None):
    default_messages = [
        {GPTCONSTANT.ROLE: GPTCONSTANT.SYSTEM, GPTCONSTANT.CONTENT: system_message},
        {GPTCONSTANT.ROLE: GPTCONSTANT.USER, GPTCONSTANT.CONTENT: cur_user_message},
    ]
    if not previous_messages:
        return default_messages
    else:
        return previous_messages + default_messages




def _gpt_reponse_helper(messages, model_name, response_format=None):

    if response_format == "json_object" or response_format:
        response = AZURE_CHAT_CLIENT.chat.completions.create(
            messages=messages, temperature=0,
            model=model_name, seed = 42,
            n=1, response_format = { "type": "json_object" }
        )

    else:
        response = AZURE_CHAT_CLIENT.chat.completions.create(
            messages=messages, temperature=0,
            model=model_name, seed = 42,
            n=1
        )

    return response

#%%

def get_tokenizer(model_name):
    return tiktoken.get_encoding("cl100k_base")

# Function to count tokens
def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

# Function to get total tokens and break down
def calculate_token_details(messages, response_content, tokenizer):
    prompt_tokens = sum(count_tokens(msg[GPTCONSTANT.CONTENT], tokenizer) for msg in messages)
    completion_tokens = count_tokens(response_content, tokenizer)
    total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens

def calculate_cost(total_tokens, cost_per_1000_tokens):
    return (total_tokens / 1000) * cost_per_1000_tokens




def get_azure_response_query(
    system_message,
    cur_user_message,
    model_name=MODEL_NAMES.GPT_4o_mini,
    return_json=False
):
    response = None
    messages = add_or_initialize_messages(cur_user_message, system_message)
    tokenizer = get_tokenizer(model_name)
   
    try:
        response = _gpt_reponse_helper(messages, model_name, response_format=return_json)
        response_content = response.choices[0].message.content
        input_tokens, output_tokens, total_tokens = calculate_token_details(messages, response_content, tokenizer)
       
        # Cost per 1000 tokens
        input_cost_per_1000_tokens = 0.000150
        output_cost_per_1000_tokens = 0.000600
       
        input_cost = calculate_cost(input_tokens, input_cost_per_1000_tokens)
        output_cost = calculate_cost(output_tokens, output_cost_per_1000_tokens)
        total_cost = input_cost + output_cost
       
        print(f"Input Tokens: {input_tokens}")
        print(f"Output Tokens: {output_tokens}")
        print(f"Total Tokens: {total_tokens}")
        print(f"Input Token Cost: ${input_cost:.4f}")
        print(f"Output Token Cost: ${output_cost:.4f}")
        print(f"Total Cost: ${total_cost:.4f}")

        return response_content
    except Exception as err:
        print("Could not generate output:", err)
        return None

#%%
 
system_message = '"role": "system", "content": "Write Appropriate System Message."'

user_message = '"role": "system", "content": "Write Appropriate User Message."'

#%%
response = get_azure_response_query(system_message, user_message)
print(response)
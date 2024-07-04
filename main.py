import argparse
import pandas as pd
from getpass import getpass
import pandas as pd
import os
from tqdm import tqdm
import time
from random import random, uniform
import torch
import json
from call_agent import ask_agent
from IPython import embed
from datetime import datetime
from maia_api import *
import random

random.seed(0000)

# layers to explore for each model
layers = {
    "resnet152": ['conv1','layer1','layer2','layer3','layer4'],
    "clip-RN50" : ['layer1','layer2','layer3','layer4'],
    "dino_vits8": ['blocks.1.mlp.fc1','blocks.3.mlp.fc1','blocks.5.mlp.fc1','blocks.7.mlp.fc1','blocks.9.mlp.fc1','blocks.11.mlp.fc1']
}

def call_argparse():
    parser = argparse.ArgumentParser(description='Process Arguments')	
    parser.add_argument('--maia', type=str, default='gpt-4-vision-preview', choices=['gpt-4-vision-preview','gpt-4-turbo'], help='maia agent name')	
    parser.add_argument('--task', type=str, default='neuron_description', choices=['neuron_description'], help='task to solve, default is neuron description') #TODO: add other tasks
    parser.add_argument('--model', type=str, default='resnet152', choices=['resnet152','clip-RN50','dino_vits8'], help='model to interp') #TODO: add synthetic neurons
    parser.add_argument('--units', type=str2dict, default='layer4=122', help='units to interp')	
    parser.add_argument('--unit_mode', type=str, default='manual', choices=['from_file','random','manual'], help='units to interp')	
    parser.add_argument('--unit_file_path', type=str, default='./neuron_indices/', help='units to interp')	
    parser.add_argument('--num_of_units', type=int, default=10, help='units to interp (if mode "unit_mode" is set to "random")')	
    parser.add_argument('--debug', action='store_true', help='debug mode, print dialogues to screen', default=False)
    parser.add_argument('--path2save', type=str, default='./results/', help='a path to save the experiment outputs')	
    parser.add_argument('--path2prompts', type=str, default='./prompts/', help='path to prompt to use')	
    parser.add_argument('--path2exemplars', type=str, default='./exemplars/', help='path to net disect top 15 exemplars images')	
    parser.add_argument('--device', type=int, default=0, help='gpu decvice to use (e.g. 1)')	
    parser.add_argument('--text2image', type=str, default='sd', choices=['sd','dalle'], help='name of text2image model')	
    args = parser.parse_args()
    return args

# Convert a comma-separated key=value pairs into a dictionary
def str2dict(arg_value):
    my_dict = {}
    if arg_value:
        for item in arg_value.split(':'):
            key, value = item.split('=')
            values = value.split(',')
            my_dict[key] = [int(v) for v in values]
        embed()
    return my_dict


# return the prompt according to the task
def return_Prompt(prompt_path,setting='neuron_description'):
    with open(f'{prompt_path}/api.txt', 'r') as file:
        sysPrompt = file.read()
    with open(f'{prompt_path}/user_{setting}.txt', 'r') as file:
        user_prompt = file.read()
    return sysPrompt, user_prompt

# save the field from the history to a file
def save_feild(history, filepath, field_name, first=False, end=True):
    text2save = None
    for i in history:
        if i['role'] == 'assistant':
            for entry in i['content']:
                if field_name in entry["text"]:
                    if end:
                        text2save = entry["text"].split(field_name)[-1].split('\n')[0]
                    else:
                        text2save = entry["text"].split(field_name)[-1]
                    if first: break
    if text2save!= None:
        with open(filepath,'w') as f:
            f.write(text2save) 
        f.close()   

# save the full experiment history to a .json file
def save_history(history,filepath):
    with open(filepath+'.json', 'w') as file:
        json.dump(history, file)
    file.close()

# save the full experiment history, and the final description and label to .txt files.
def save_dialouge(history,path2save):
    save_history(history,path2save+'/history')
    save_feild(history, path2save+'/description.txt', '[DESCRIPTION]: ')
    save_feild(history, path2save+'/label.txt', '[LABEL]: ', end=True)

# returns a dictionary of {'layer':[units]} to explore
def units2explore(unit_mode): 
    if unit_mode == 'random':
        unit_inx = {}
        for layer in layers[args.model]:
            unit_inx[layer]  = random.choices(range(0, 64 + 1), k=args.num_of_units)
    elif unit_mode == 'from_file':
        with open(os.path.join(args.unit_file_path,args.model+'.json'), 'r') as json_file:
            unit_inx = json.load(json_file)
    elif unit_mode == 'manual':
        unit_inx = args.units
    else:
        raise ValueError("undefined unit mode.")
    return unit_inx

# final instructions to maia
def overload_instructions(prompt_path='./prompts/'):
    with open(f'{prompt_path}/final.txt', 'r') as file:
        final_instructions = file.read()
        tools.update_experiment_log(role='user', type="text", type_content=final_instructions)

# execute the experiment provided by the maia agent
def execute_maia_experiment(code,system,tools): 
    exec(compile(code, 'code', 'exec'), globals())
    execute_command(system,tools)
    return  

def get_code(maia_experiment):
    maia_code = maia_experiment.split('```python')[1].split('```')[0]
    return maia_code

# maia experiment loop
def interpretation_experiment(maia,system,tools,debug=False):
    round_count = 0
    while True:
        round_count+=1 
        maia_experiment = ask_agent(maia,tools.experiment_log) # ask maia for the next experiment given the results log to the experiment log (in the first round, the experiment log contains only the system prompt (maia api) and the user prompt (the query))
        tools.update_experiment_log(role='maia', type="text", type_content=str(maia_experiment)) # update the experiment log with maia's response (str casting is for exceptions)
        tools.generate_html() # generate the html file to visualize the experiment log
        if debug: # print the dialogue to the screen
            print(maia_experiment)
        if round_count>25: # if the interpretation process exceeds 25 rounds, ask the agent to provide final description
            overload_instructions()
        elif "```python" in maia_experiment: # if the response contains python code, execute the code
            maia_code = get_code(maia_experiment)
            if "execute_command" in maia_code:
                try:
                    execute_maia_experiment(maia_code,system,tools) # execute the code fro maia, code itself should contain the tools.update_experiment_log(...) to update the experiment log with the execution results
                except Exception as e:
                    tools.update_experiment_log(role='execution', type="text", type_content=f"Error while executing 'execute_command':\n{str(e)}")
                tools.generate_html()
            else: 
                tools.update_experiment_log(role='execution', type="text", type_content="No 'execute_command' function was provided.")
        else:
            if "[DESCRIPTION]" in maia_experiment: return # stop the experiment if the response contains the final description. "[DESCRIPTION]" is the stopping signal.  
            else: # if the response is not the final description, and does not contains any python code, ask maia to provide more information
                tools.update_experiment_log(role='execution', type="text", type_content="No code to run was provided, please continue with the experiments based on your findings.")


def main(args):
    maia_api, user_query = return_Prompt(args.path2prompts, setting=args.task) # load system prompt (maia api) and user prompt (the user query)
    unit_inx = units2explore(args.unit_mode) # returns a dictionary of {'layer':[units]} to explore
    for layer in unit_inx.keys(): 
        units = unit_inx[layer]
        net_dissect = DatasetExemplars(args.path2exemplars, args.path2save, args.model, layer, units) # precomputes dataset examplars for tools.dataset_exemplars
        for unit in units: 
            print(layer,unit)
            path2save = os.path.join(args.path2save,args.maia,args.model,str(layer),str(unit))
            if os.path.exists(path2save+'/description.txt'): continue
            os.makedirs(path2save, exist_ok=True)
            system = System(unit, layer, args.model, args.device, net_dissect.thresholds) # initialize the system class
            tools = Tools(path2save, args.device, net_dissect, text2image_model_name=args.text2image) # initialize the tools class
            tools.update_experiment_log(role='system', type="text", type_content=maia_api) # update the experiment log with the system prompt
            tools.update_experiment_log(role='user', type="text", type_content=user_query) # update the experiment log with the user prompt
            interp_count = 0
            while True:
                try:
                    interp_count+=1
                    interpretation_experiment(args.maia,system,tools,args.debug) # this is where the magic happens! maia interactively execute experiments on the specified unit
                    save_dialouge(tools.experiment_log,path2save)
                    break
                except Exception as e:
                    print(e)
                    if interp_count>5: # if the interpretation process exceeds 5 rounds, save the current state and move to the next unit
                        break

if __name__ == '__main__':
    args = call_argparse()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)




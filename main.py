import argparse
import os
import random
from IPython import embed

import torch
from dotenv import load_dotenv

from utils.call_agent import ask_agent
from maia_api import System, Synthetic_System, Tools
from utils.ExperimentEnvironment import ExperimentEnvironment
from utils.DatasetExemplars import DatasetExemplars
from utils.SyntheticExemplars import SyntheticExemplars
from utils.main_utils import *
from utils.api_utils import str2image
import json

random.seed(0000)

# layers to explore for each model
layers = {
    "resnet152": ['conv1','layer1','layer2','layer3','layer4'],
    "clip-RN50" : ['layer1','layer2','layer3','layer4'],
    "dino_vits8": ['blocks.1.mlp.fc1','blocks.3.mlp.fc1','blocks.5.mlp.fc1','blocks.7.mlp.fc1','blocks.9.mlp.fc1','blocks.11.mlp.fc1'],
    "synthetic_neurons": ['mono','or','and']
}

def call_argparse():
    parser = argparse.ArgumentParser(description='Process Arguments')	
    parser.add_argument('--maia', type=str, default='claude', choices=['claude','gpt-4-vision-preview','gpt-4-turbo'], help='maia agent name')	
    parser.add_argument('--task', type=str, default='neuron_description', choices=['neuron_description'], help='task to solve, default is neuron description')
    parser.add_argument('--model', type=str, default='resnet152', choices=['resnet152','clip-RN50','dino_vits8','synthetic_neurons'], help='model to interp')
    parser.add_argument('--units', type=str2dict, default='layer4=122', help='units to interp')	
    parser.add_argument('--unit_mode', type=str, default='manual', choices=['from_file','random','manual'], help='units to interp')	
    parser.add_argument('--unit_file_path', type=str, default='./neuron_indices/', help='units to interp')	
    parser.add_argument('--num_of_units', type=int, default=10, help='units to interp (if mode "unit_mode" is set to "random")')	
    parser.add_argument('--debug', action='store_true', help='debug mode, print dialogues to screen', default=False)
    parser.add_argument('--path2save', type=str, default='./results/', help='a path to save the experiment outputs')	
    parser.add_argument('--path2prompts', type=str, default='./prompts/', help='path to prompt to use')	
    parser.add_argument('--path2exemplars', type=str, default='./exemplars/', help='path to net disect top 15 exemplars images')	
    parser.add_argument('--device', type=int, default=0, help='gpu decvice to use (e.g. 1)')	
    parser.add_argument('--text2image', type=str, default='flux', choices=['flux','sd','dalle'], help='name of text2image model')	
    parser.add_argument('--p2p_model', type=str, default='instdiff', choices=['instdiff','ip2p'], help='name of p2p model')
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
    return my_dict

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

# maia experiment loop
def interpretation_experiment(maia,system,tools,experiment_env,path2save,debug=False):
    round_count = 0
    while True:
        round_count+=1 
        maia_experiment = ask_agent(maia,tools.experiment_log) # ask maia for the next experiment given the results log to the experiment log (in the first round, the experiment log contains only the system prompt (maia api) and the user prompt (the query))
        tools.update_experiment_log(role='maia', type="text", type_content=str(maia_experiment)) # update the experiment log with maia's response (str casting is for exceptions)
        tools.generate_html(path2save) # generate the html file to visualize the experiment log
        if debug: # print the dialogue to the screen
            print(maia_experiment)
        if round_count>25: # if the interpretation process exceeds 25 rounds, ask the agent to provide final description
            overload_instructions()
        if "[DESCRIPTION]" in maia_experiment: return # stop the experiment if the response contains the final description. "[DESCRIPTION]" is the stopping signal.  
        try:
            output = experiment_env.execute_experiment(maia_experiment) # if the response does not contain the final description, execute the experiment
            if output:
                tools.update_experiment_log(role='user', type="text", type_content=output)
        except Exception as exec_e:
            tools.update_experiment_log(role='user', type="text", type_content=f"Error during experiment execution: {str(exec_e)}")


def main(args):
    maia_api, user_query = return_prompt(args.path2prompts, setting=args.task) # load system prompt (maia api) and user prompt (the user query)
    unit_inx = units2explore(args.unit_mode) # returns a dictionary of {'layer':[units]} to explore
    for layer in unit_inx.keys(): # for the synthetic neurons, the layer is the neuron type ("mono", "or", "and")
        units = unit_inx[layer]
        if args.model == 'synthetic_neurons':
            net_dissect = SyntheticExemplars(os.path.join(args.path2exemplars, args.model), args.path2save, layer) # precomputes synthetic dataset examplars for tools.dataset_exemplars. 
            with open(os.path.join('./synthetic-neurons-dataset/labels/',f'{layer}.json'), 'r') as file: # load the synthetic neuron labels
                synthetic_neuron_data = json.load(file)
        else:
            net_dissect = DatasetExemplars(args.path2exemplars, args.path2save, args.model, layer, units) # precomputes dataset examplars for tools.dataset_exemplars
        for unit in units: 
            print(layer,unit)
            path2save = os.path.join(args.path2save,args.maia,args.model,str(layer),str(unit))
            if os.path.exists(path2save+'/description.txt'): continue
            os.makedirs(path2save, exist_ok=True)
            if args.model == 'synthetic_neurons':
                gt_label = synthetic_neuron_data[unit]["label"].rsplit('_')
                print("groundtruth label: ",gt_label)
                system = Synthetic_System(unit, gt_label, layer, args.device)
            else:
                system = System(unit, layer, args.model, args.device, net_dissect.thresholds) # initialize the system class
            tools = Tools(path2save, args.device, net_dissect, text2image_model_name=args.text2image) # initialize the tools class
            experiment_env = ExperimentEnvironment(system, tools, globals()) # initialize the experiment environment

            tools.update_experiment_log(role='system', type="text", type_content=maia_api) # update the experiment log with the system prompt
            tools.update_experiment_log(role='user', type="text", type_content=user_query) # update the experiment log with the user prompt
            try:
                interpretation_experiment(args.maia,system,tools,experiment_env,path2save,args.debug) # this is where the magic happens! maia interactively execute experiments on the specified unit
                save_dialogue(tools.experiment_log,path2save)
            except Exception as e:
                print(e)
                break

if __name__ == '__main__':
    args = call_argparse()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
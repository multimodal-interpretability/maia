import argparse
import os
import random
from IPython import embed

import torch

from utils.call_agent import ask_agent
from maia_api import System, Tools
from utils.ExperimentEnvironment import ExperimentEnvironment
from utils.DatasetExemplars import DatasetExemplars
from utils.main_utils import return_prompt, save_dialogue, overload_instructions, load_unit_config, generate_save_path, create_unit_config

random.seed(0000)

def call_argparse():
    parser = argparse.ArgumentParser(description='Process Arguments')	
    parser.add_argument('--maia', type=str, default='gpt-4o', choices=['gpt-4-vision-preview','gpt-4-turbo', 'gpt-4o'], help='maia agent name')	
    parser.add_argument('--task', type=str, default='mult', help='task to solve, default is unit description') #TODO: add other tasks
    parser.add_argument('--unit_config_name', type=str, help='json specifying the units to interpret, if not provided, must specify manually')	
    parser.add_argument('--model1', type=str, default='finetune_resnet_gelu', help='name of the first model')
    parser.add_argument('--model2', type=str, default='gradnorm_resnet_gelu', help='name of the second model')
    parser.add_argument('--layer', type=str, default='layer4', help='layer to interpret')
    parser.add_argument('--neurons', nargs='+', type=int, help='list of neurons to interpret')
    parser.add_argument('--n_exemplars', type=int, default=15, help='number of examplars for initialization')	
    # TODO - Must be 1 for now, for displaying original images
    parser.add_argument('--images_per_prompt', type=int, default=1, help='name of text2image model')	
    parser.add_argument('--path2save', type=str, default='./results/', help='a path to save the experiment outputs')	
    parser.add_argument('--path2prompts', type=str, default='./prompts/', help='path to prompt to use')	
    parser.add_argument('--path2exemplars', type=str, default='./exemplars/', help='path to net disect top 15 exemplars images')	
    parser.add_argument('--device', type=int, default=0, help='gpu decvice to use (e.g. 1)')	
    parser.add_argument('--text2image', type=str, default='sd', choices=['sd','dalle'], help='name of text2image model')	
    parser.add_argument('--debug', action='store_true', help='debug mode, print dialogues to screen', default=False)
    args = parser.parse_args()
    return args


def main(args):
    maia_api, user_query = return_prompt(args.path2prompts, setting=args.task) # load system prompt (maia api) and user prompt (the user query)
    # if the unit file name is provided, only one experiment is run, otherwise, multiple experiments are run based on the number of neurons provided
    if args.unit_config_name is not None:
        unit_config_name = args.unit_config_name
        num_experiments = 1
    else:
        num_experiments = len(args.neurons)
    
    for i in range(num_experiments):
        if args.unit_config_name is None:
            unit_config_name = f"{args.model1}_{args.model2}_{args.layer}_{args.neurons[i]}"
            unit_config = create_unit_config(args.model1, args.model2, args.layer, args.neurons[i])
        else:
            unit_config = load_unit_config(unit_config_name)
        path2save = generate_save_path(args.path2save, args.maia, unit_config_name)
        os.makedirs(path2save, exist_ok=True)

        net_dissect = DatasetExemplars(args.path2exemplars, args.n_exemplars, args.path2save, unit_config)
        system = System(unit_config, net_dissect.thresholds, args.device)
        tools = Tools(path2save, args.device, net_dissect, images_per_prompt=args.images_per_prompt, text2image_model_name=args.text2image, image2text_model_name=args.maia)
        experiment_env = ExperimentEnvironment(system, tools, globals())
        # TODO - Dyanamically add enumerated units to user prompt
        tools.update_experiment_log(role='system', type="text", type_content=maia_api) # update the experiment log with the system prompt
        tools.update_experiment_log(role='user', type="text", type_content=user_query) # update the experiment log with the user prompt
        interp_count = 0
        while True:
            try:
                interp_count+=1
                interpretation_experiment(args.maia, tools, experiment_env, args.debug) # this is where the magic happens! maia interactively execute experiments on the specified unit
                save_dialogue(tools.experiment_log,path2save)
                break
            except Exception as e:
                print(e)
                if interp_count>5: # if the interpretation process exceeds 5 rounds, save the current state and move to the next unit
                    break


# maia experiment loop
def interpretation_experiment(maia, tools: Tools, experiment_env: ExperimentEnvironment, debug=False):
    round_count = 0
    while True:
        round_count+=1 
        maia_experiment = ask_agent(maia,tools.experiment_log) # ask maia for the next experiment given the results log to the experiment log (in the first round, the experiment log contains only the system prompt (maia api) and the user prompt (the query))
        tools.update_experiment_log(role='maia', type="text", type_content=str(maia_experiment)) # update the experiment log with maia's response (str casting is for exceptions)
        tools.generate_html() # generate the html file to visualize the experiment log
        if debug: # print the dialogue to the screen
            print(maia_experiment)
        if round_count>25: # if the interpretation process exceeds 25 rounds, ask the agent to provide final description
            overload_instructions(tools)
        else:
            # TODO - Make token dynamic
            if "[Difference]" in maia_experiment: return # stop the experiment if the response contains the final description. "[DESCRIPTION]" is the stopping signal.  
            try:
                experiment_output = experiment_env.execute_experiment(maia_experiment)
                if experiment_output != "":
                    tools.update_experiment_log(role='user', type="text", type_content=experiment_output)
            except ValueError:
                tools.update_experiment_log(role='execution', type="text", type_content="No code to run was provided, please continue with the experiments based on your findings, or output your final [Difference].")


if __name__ == '__main__':
    args = call_argparse()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
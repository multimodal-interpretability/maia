import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from maia_api import Synthetic_System, System, Tools
from utils.agents.factory import create_agent
from utils.DatasetExemplars import DatasetExemplars
from utils.ExperimentEnvironment import ExperimentEnvironment
from utils.flux import FluxDev
from utils.flux_kontext import FluxKontextDev
from utils.main_utils import *
from utils.SyntheticExemplars import SyntheticExemplars

random.seed(0000)

# layers to explore for each model
layers = {
    'resnet152': ['conv1', 'layer1', 'layer2', 'layer3', 'layer4'],
    'clip-RN50': ['layer1', 'layer2', 'layer3', 'layer4'],
    'dino_vits8': [
        'blocks.1.mlp.fc1',
        'blocks.3.mlp.fc1',
        'blocks.5.mlp.fc1',
        'blocks.7.mlp.fc1',
        'blocks.9.mlp.fc1',
        'blocks.11.mlp.fc1',
    ],
    'synthetic_neurons': ['mono', 'or', 'and'],
}


def call_argparse():
    parser = argparse.ArgumentParser(description='Process Arguments')
    parser.add_argument(
        '--agent',
        type=str,
        default='claude',
        help='Maia Agent Backbone',
    )
    parser.add_argument(
        '--base_url',
        type=str,
        default='http://torralba-3090-1:11434',
        help='local maia server base_url (e.g. localhost:8000)',
    )
    parser.add_argument(
        '--task',
        type=str,
        default='neuron_description',
        help='task to solve, default is neuron description',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='resnet152',
        choices=['resnet152', 'clip-RN50', 'dino_vits8', 'synthetic_neurons'],
        help='model to interp',
    )
    parser.add_argument(
        '--units', type=str2dict, default='layer4=122', help='units to interp'
    )
    parser.add_argument(
        '--unit_mode',
        type=str,
        default='manual',
        choices=['from_file', 'random', 'manual'],
        help='units to interp',
    )
    parser.add_argument(
        '--unit_file_path',
        type=str,
        default='./neuron_indices/',
        help='units to interp',
    )
    parser.add_argument(
        '--num_of_units',
        type=int,
        default=10,
        help='units to interp (if mode "unit_mode" is set to "random")',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='debug mode, print dialogues to screen',
        default=False,
    )
    parser.add_argument(
        '--path2save',
        type=str,
        default='./results/',
        help='a path to save the experiment outputs',
    )
    parser.add_argument(
        '--path2prompts',
        type=str,
        default='./prompts/gemma/',
        help='path to prompt to use',
    )
    parser.add_argument(
        '--path2exemplars',
        type=str,
        default='./exemplars',
        help='path to net disect top 15 exemplars images',
    )
    parser.add_argument(
        '--device', type=int, default=0, help='gpu decvice to use (e.g. 1)'
    )
    parser.add_argument(
        '--total_chunks',
        type=int,
        default=1,
        help='Number of chunks to split the compute',
    )
    parser.add_argument(
        '--chunk_id',
        type=int,
        default=1,
        help='Chunk id to process (1 to chunks)',
    )
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
            unit_inx[layer] = random.choices(range(0, 64 + 1), k=args.num_of_units)
    elif unit_mode == 'from_file':
        with open(os.path.join(args.unit_file_path, args.model + '.json')) as json_file:
            unit_inx = json.load(json_file)
    elif unit_mode == 'manual':
        unit_inx = args.units
    else:
        raise ValueError('undefined unit mode.')
    return unit_inx


def is_completed(layer, unit):
    base_path = Path(args.path2save) / args.agent / args.model / str(layer) / str(unit)
    completion_files = ['description.txt', 'history.json']
    return any((base_path / filename).exists() for filename in completion_files)


# maia experiment loop
def interpretation_experiment(
    maia, system, tools, experiment_env, path2save, debug=False, base_url=None
):
    agent = create_agent(
        model=maia,
        max_attempts=5,
        max_output_tokens=4096,
        **({'base_url': base_url} if 'local' in maia else {}),
    )
    round_count = 0
    while True:
        round_count += 1
        maia_experiment = agent.ask(
            tools.experiment_log
        )  # ask maia for the next experiment given the results log to the experiment log (in the first round, the experiment log contains only the system prompt (maia api) and the user prompt (the query))
        tools.update_experiment_log(
            role='maia', type='text', type_content=str(maia_experiment)
        )  # update the experiment log with maia's response (str casting is for exceptions)
        tools.generate_html(
            path2save
        )  # generate the html file to visualize the experiment log
        if debug:  # print the dialogue to the screen
            print(maia_experiment)
        if (
            round_count > 25
        ):  # if the interpretation process exceeds 25 rounds, ask the agent to provide final description
            overload_instructions(tools, prompt_path='./prompts/open/')
        if '[DESCRIPTION]' in maia_experiment:
            return  # stop the experiment if the response contains the final description. "[DESCRIPTION]" is the stopping signal.
        try:
            output = experiment_env.execute_experiment(
                maia_experiment
            )  # if the response does not contain the final description, execute the experiment
            if output:
                tools.update_experiment_log(
                    role='user', type='text', type_content=output
                )
        except Exception as exec_e:
            tools.update_experiment_log(
                role='user',
                type='text',
                type_content=f'Error during experiment execution: {str(exec_e)}',
            )


def main(args):
    if args.chunk_id < 1 or args.chunk_id > args.total_chunks:
        raise ValueError(
            f'--chunk_id must be in [1, {args.total_chunks}], got {args.chunk_id}'
        )

    maia_api, user_query = return_prompt(args.path2prompts, setting=args.task)

    # Map: layer -> [units]
    unit_inx = units2explore(args.unit_mode)

    # Precompute (layer, unit) pairs, but filter out those that already have results
    all_pairs = [
        (layer, unit)
        for layer, units in unit_inx.items()
        for unit in units
        if not is_completed(layer, unit)
    ]

    # split into ~equal chunks and select the 1-based chunk_id
    all_pairs = np.array_split(all_pairs, args.total_chunks)
    all_pairs = list(map(tuple, all_pairs[args.chunk_id - 1]))

    # Caches so we only init per-layer resources once
    net_cache = {}
    labels_cache = {}

    # Load text2image and image2image
    text2image_model = FluxDev()
    img2img_model = FluxKontextDev()
    for layer, unit in tqdm(all_pairs, desc='Units overall'):
        unit = int(unit)
        if layer not in net_cache:
            if args.model == 'synthetic_neurons':
                nd = SyntheticExemplars(
                    os.path.join(args.path2exemplars, args.model),
                    args.path2save,
                    layer,
                )
                with open(
                    os.path.join(
                        './synthetic_neurons_dataset/labels/', f'{layer}.json'
                    ),
                ) as f:
                    labels_cache[layer] = json.load(f)
            else:
                nd = DatasetExemplars(
                    args.path2exemplars,
                    args.path2save,
                    args.model,
                    layer,
                    unit_inx[layer],
                )
            net_cache[layer] = nd

        net_dissect = net_cache[layer]

        # Make sure save dir exists here, since we filtered out the completed ones
        path2save = os.path.join(
            args.path2save, args.agent, args.model, str(layer), str(unit)
        )
        os.makedirs(path2save, exist_ok=True)

        if args.model == 'synthetic_neurons':
            synthetic_neuron_data = labels_cache[layer]
            gt_label = synthetic_neuron_data[unit]['label'].rsplit('_')
            print('groundtruth label:', gt_label)
            system = Synthetic_System(unit, gt_label, layer, args.device)
        else:
            system = System(
                unit, layer, args.model, args.device, net_dissect.thresholds
            )

        tools = Tools(
            path2save,
            args.device,
            net_dissect,
            text2image_model=text2image_model,
            img2img_model=img2img_model,
        )
        experiment_env = ExperimentEnvironment(system, tools, globals())

        tools.update_experiment_log(role='system', type='text', type_content=maia_api)
        tools.update_experiment_log(role='user', type='text', type_content=user_query)

        try:
            interpretation_experiment(
                args.agent,
                system,
                tools,
                experiment_env,
                path2save,
                args.debug,
                args.base_url,
            )
            save_dialogue(tools.experiment_log, path2save)
        except Exception as e:
            print(e)
            break


if __name__ == '__main__':
    args = call_argparse()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    main(args)

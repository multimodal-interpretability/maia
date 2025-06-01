'''Utils for the main.py file'''
import os
import json

# return the prompt according to the task
def return_prompt(prompt_path,setting='unit_description'):
    with open(f'{prompt_path}/api.txt', 'r') as file:
        sysPrompt = file.read()
    with open(f'{prompt_path}/user_{setting}.txt', 'r') as file:
        user_prompt = file.read()
    return sysPrompt, user_prompt

# save the field from the history to a file
def save_field(history, filepath, field_name, first=False, end=True):
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
def save_dialogue(history,path2save):
    save_history(history,path2save+'/history')
    save_field(history, path2save+'/description.txt', '[DESCRIPTION]: ')
    save_field(history, path2save+'/label.txt', '[LABEL]: ', end=True)

# final instructions to maia
def overload_instructions(tools, prompt_path='./prompts/'):
    with open(f'{prompt_path}/final.txt', 'r') as file:
        final_instructions = file.read()
        tools.update_experiment_log(role='user', type="text", type_content=final_instructions)


def load_unit_config(unit_file_name):
    with open(os.path.join("neuron_indices", unit_file_name)) as json_file:
        unit_config = json.load(json_file)
    return unit_config

def create_unit_config(model1: str, model2: str, layer_name: str, neuron: int):
    unit_config = {
        model1: {
            layer_name: [neuron]
        },
        model2: {
            layer_name: [neuron]
        }
    }
    return unit_config

def generate_save_path(path2save: str, maia: str, unit_file_name: str):
    path2save = os.path.join(path2save, maia, unit_file_name)
    return path2save

def generate_numbered_path(path:str, file_extension:str=""):
    i = 0
    numbered_path = path + "_" + str(i) + file_extension
    while os.path.exists(numbered_path):
        i += 1
        numbered_path = path + "_" + str(i) + file_extension
    return numbered_path
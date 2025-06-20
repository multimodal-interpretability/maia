'''Utils for the main.py file'''
import json
from utils.api_utils import str2image

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

def plot_results_notebook(experiment_log):
    for entry in experiment_log:
        if (entry['role'] == 'assistant'):
            print('\n\n*** MAIA: ***\n\n')  
        else: 
            print('\n\n*** Experiment Execution: ***\n\n')
        for item in entry['content']:
            if item['type'] == 'text':
                print(item['text'])
            elif item['type'] == 'image_url':
                display(str2image(item['image_url']['url'].split(',')[1]))
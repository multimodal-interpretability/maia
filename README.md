# A Multimodal Automated Interpretability Agent #
### ICML 2024 ###

### [Project Page](https://multimodal-interpretability.csail.mit.edu/maia) | [Arxiv](https://multimodal-interpretability.csail.mit.edu/maia)

<img align="right" width="42%" src="/docs/static/figures/maia_teaser.jpg">

[Tamar Rott Shaham](https://tamarott.github.io/)\*, [Sarah Schwettmann](https://cogconfluence.com/)\*, <br>
[Franklin Wang](https://frankxwang.github.io/), [Achyuta Rajaram](https://twitter.com/AchyutaBot), [Evan Hernandez](https://evandez.com/), [Jacob Andreas](https://www.mit.edu/~jda/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/) <br>
\*equal contribution <br><br>

MAIA is a system that uses neural models to automate neural model understanding tasks like feature interpretation and failure mode discovery. It equips a pre-trained vision-language model with a set of tools that support iterative experimentation on subcomponents of other models to explain their behavior. These include tools commonly used by human interpretability researchers: for synthesizing and editing inputs, computing maximally activating exemplars from real-world datasets, and summarizing and describing experimental results. Interpretability experiments proposed by MAIA compose these tools to describe and explain system behavior.

**News** 
\
[July 3]: We release MAIA implementation code for neuron labeling 

**This repo is under active development. Sign up for updates by email using [this google form](https://forms.gle/Zs92DHbs3Y3QGjXG6).**


### Installations ###
clone this repo and install all requirements:
```bash
git clone https://github.com/multimodal-interpretability/maia.git
cd maia
bash setup_env.sh
```

download [net-dissect](https://netdissect.csail.mit.edu/) precomputed exemplars:
```bash
bash download_exemplars.sh
```

### Quick Start ###
You can run demo experiments on individual units using ```demo.ipynb```:
\
\
Install Jupyter Notebook via pip (if Jupyter is already installed, continue to the next step)
```bash
pip install notebook
```
Launch Jupyter Notebook
```bash
jupyter notebook
```
This command will start the Jupyter Notebook server and open the Jupyter Notebook interface in your default web browser. The interface will show all the notebooks, files, and subdirectories in this repo (assuming is was initiated from the maia path). Open ```demo.ipynb``` and proceed according to the instructions.

### Batch experimentation ###
To run a batch of experiments, use ```main.py```:

#### Load openai api key ####
(in case you don't have an openai api-key, you can get one by following the instructions [here](https://platform.openai.com/docs/quickstart)).

Set your api-key as an environment variable (this is a bash command, look [here](https://platform.openai.com/docs/quickstart) for other OS)
```bash
export OPENAI_API_KEY='your-api-key-here'
```

#### Run MAIA ####
Manually specify the model and desired units in the format ```layer#1=unit#1,unit#2... : layer#1=unit#1,unit#2...``` by calling e.g.:
```bash
python main.py --model resnet152 --unit_mode manual --units layer3=229,288:layer4=122,210
``` 
OR by loading a ```.json``` file specifying the units (see example in ```./neuron_indices/```)
```bash
python main.py --model resnet152 --unit_mode from_file --unit_file_path ./neuron_indices/
```
Adding ```--debug``` to the call will print all results to the screen.
Refer to the documentation of ```main.py``` for more configuration options.

Results are automatically saved to an html file under ```./results/``` and can be viewed in your browser by starting a local server:
```bash
python -m http.server 80
```
Once the server is up, open the html in [http://localhost:80](http://localhost:80/results/)

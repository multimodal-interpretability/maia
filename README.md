# A Multimodal Automated Interpretability Agent #
### ICML 2024 ###

### [Project Page](https://multimodal-interpretability.csail.mit.edu/maia) | [Arxiv](https://multimodal-interpretability.csail.mit.edu/maia) | [Experiment browser](https://multimodal-interpretability.csail.mit.edu/maia/experiment-browser/)

<img align="right" width="42%" src="/docs/static/figures/maia_teaser.jpg">

[Tamar Rott Shaham](https://tamarott.github.io/)\*, [Sarah Schwettmann](https://cogconfluence.com/)\*, <br>
[Franklin Wang](https://frankxwang.github.io/), [Achyuta Rajaram](https://twitter.com/AchyutaBot), [Evan Hernandez](https://evandez.com/), [Jacob Andreas](https://www.mit.edu/~jda/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/) <br>
\*equal contribution <br><br>

MAIA is a system that uses neural models to automate neural model understanding tasks like feature interpretation and failure mode discovery. It equips a pre-trained vision-language model with a set of tools that support iterative experimentation on subcomponents of other models to explain their behavior. These include tools commonly used by human interpretability researchers: for synthesizing and editing inputs, computing maximally activating exemplars from real-world datasets, and summarizing and describing experimental results. Interpretability experiments proposed by MAIA compose these tools to describe and explain system behavior.

**News** 
\
[June 19 2025]: Releasing MAIA 2.0: MAIA's code execution now runs free-form blocks of code, and it calls a flexible display tool to show the results in the experiment log. Additionally, added support for Claude 3.5 Sonnet, GPT-4o, and GPT-4 Turbo as backbones for MAIA backbone, Flux for image generation, and InstructDiffusion for image editing. 
\
[August 14 2024]: Synthetic neurons are now available (both in `demo.ipynb` and in `main.py`)
\
[July 3 2024]: We release MAIA implementation code for neuron labeling 

**This repo is under active development. Sign up for updates by email using [this google form](https://forms.gle/Zs92DHbs3Y3QGjXG6).**


### Installations ###
clone this repo and create a conda environment:
```bash
git clone https://github.com/multimodal-interpretability/maia.git
cd maia
conda create -n maia python=3.10 --file conda_packages.txt -c nvidia
conda activate maia
```

install packages and dependencies
```bash
pip install -r torch_requirements.txt
pip install -r requirements.txt
pip install -r torch_requirements.txt --force-reinstall
pip install git+https://github.com/huggingface/transformers.git
```

install InstructDiffusion and Flux
```bash
cd utils
git clone https://github.com/cientgu/InstructDiffusion.git
pip install -r requirements_instdiff_flux.txt
cd InstructDiffusion
bash scripts/download_pretrained_instructdiffusion.sh
cd ../../
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

NEW: `demo.ipynb` now supports synthetic neurons. Follow installation instructions at `./synthetic-neurons-dataset/README.md`. After installation is done, you can define MAIA to run on synthetic neurons according to the instructions in `demo.ipynb`.

### Batch experimentation ###
To run a batch of experiments, use ```main.py```:

#### Load OpenAI or Anthropic API key ####
(you can get an OpenAI API key by following the instructions [here](https://platform.openai.com/docs/quickstart) and an Anthropic API key by following the instructions [here](https://docs.anthropic.com/en/docs/get-started)).

Set your API key as an environment variable
```bash
export OPENAI_API_KEY='your-openai-api-key-here'
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
```

#### Load Huggingface key ####
You will need a Huggingface API key if you want to use Stable Diffusion 3.5 as the text2image model (you can get a HuggingFace API key by following the instructions [here](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)).

Set your API key as an environment variable
```bash
export HF_TOKEN='your-hf-token-here'
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

#### Run MAIA on sythetic neurons ####

You can now run maia on synthetic neurons with ground-truth labels (see sec. 4.2 in the paper for more details).

Follow installation instructions at `./synthetic-neurons-dataset/README.md`. Then you should be able to run `main.py` on synthetic neurons by calling e.g.:
```bash
python main.py --model synthetic_neurons --unit_mode manual --units mono=1,8:or=9:and=0,2,5
``` 
(neuron indices are specified according to the neuron type: "mono", "or" and "and").

You can also use the .json file to run all synthetic neurons (or specify your own file):
```bash
python main.py --model synthetic_neurons --unit_mode from_file --unit_file_path ./neuron_indices/
```

### Running netdissect

MAIA uses pre-computed exemplars for its experiments. Thus, to run MAIA on a new model, you must first compute the exemplars netdissect. MAIA was built to use exemplar data as generated by MILAN’s implementation of netdissect. 

First, clone MILAN.

```python
git clone https://github.com/evandez/neuron-descriptions.git
```

Then, set the environment variables to control where the data is inputted/outputted. Ex:

```python
%env MILAN_DATA_DIR=./data
%env MILAN_MODELS_DIR=./models
%env MILAN_RESULTS_DIR=./results
```

Make those three directories mentioned above.

When you have a new model you want to dissect:

1. Make a folder in `models` directory
2. Name it whatever you want to call your model (in this case we'll use `resnet18syn`)
3. Inside that folder, place your saved model file, but name it `imagenet.pth` (since you'll be dissecting with ImageNet)

Now, place ImageNet in the data directory. If you have imagenet installed, you create a symbolic link to it using:

```bash
ln -s /path/to/imagenet /path/to/data/imagenet
```

If you don’t have imagenet installed, follow the download instructions [here](https://image-net.org/download.php): 

Next, add under your keys in `models.py` (located in `src/exemplars/models.py`):

```python
KEYS.RESNET18_SYNTHETIC = 'resnet18syn/imagenet'
```

Then add this in the keys dict:

```python
KEYS.RESNET18_SYNTHETIC:
    ModelConfig(
        models.resnet18,
        load_weights=True,
        layers=LAYERS.RESNET18,
    ),
```

Once that's done, you should be ready to run the compute exemplars script:

```bash
python3 -m scripts.compute_exemplars resnet18syn imagenet --device cuda
```

This will run the compute exemplars script using the ResNet18 synthetic model on the ImageNet dataset, utilizing CUDA for GPU acceleration.

Finally, move the computed exemplars to the `exemplars/` folder.

### To set up the synthetic neurons:

1. **init Grounded-SAM submodule**  
  ```
  git submodule init
  git submodule update
  ```

2. **Follow the setup instructions on Grounded SAM setup:**
   - Export global variables (choose whether to run on CPU or GPU; note that running on CPU is feasible but slower, approximately 3 seconds per image):

     ```bash
     export AM_I_DOCKER="False"
     export BUILD_WITH_CUDA="True"
     export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
     export CC=$(which gcc-12)
     export CXX=$(which g++-12)
     ```
   - Install Segment Anything:
     ```bash
     pip install git+https://github.com/facebookresearch/segment-anything.git
     ```
   - Install Grounding Dino:
     ```bash
     pip install git+https://github.com/IDEA-Research/GroundingDINO.git
     ```
     
3. **Download Grounding DINO and Grounded SAM .pth files**  
   - Download Grounding DINO: 
     ```bash
     wget "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
     ```
   - Download Grounded SAM: 
     ```bash
     wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
     ```
    - Try running Grounded SAM demo:
      ```bash
      export CUDA_VISIBLE_DEVICES=0
      python grounded_sam_demo.py \
        --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint groundingdino_swint_ogc.pth \
        --sam_checkpoint sam_vit_h_4b8939.pth \
        --input_image assets/demo1.jpg \
        --output_dir "outputs" \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt "bear" \
        --device "cpu"
      ```

#### Creating Custom Synthetic-Neurons

1. Decide the mode and label(s) for your neuron.
    1. ex. Cheese OR Lemon
2. Identify an object-classification dataset you want to create exemplars from. The example notebooks use COCO.
3. Query the dataset for relevant images to your synthetic neuron
    1. i.e. if you want to build a neuron that’s selective for dogs (monosemantic neuron), query the dataset for dog images
    2. If you want to build a neuron that’s selective for dogs but only if they’re wearing collars (and neuron), query the dataset for dogs wearing collars
4. Instantiate synthetic neuron with desired setting and labels
5. Run SAMNeuron on all candidate images and save the top 15 highest activating images and their activations
6. Save the data in the format “path2data/label” or for multi-label neurons “path2data/label1_label2”
7. Convert the saved data to the format expected by MAIA, demonstrated [here](synthetic-neurons-dataset/create_synthetic_neurons.ipynb)

### Acknowledgment ###
[Christy Li](https://christykl.github.io/) and [Jake Touchet](https://www.linkedin.com/in/jake-touchet-557329297/) contributed to MAIA 2.0 release.
[Christy Li](https://christykl.github.io/) helped with cleaning up the synthetic neurons code for release.

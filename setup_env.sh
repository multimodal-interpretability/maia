conda create -n maia python=3.10
conda activate maia
conda install --name maia --file environment.txt

cd utils
git clone https://github.com/cientgu/InstructDiffusion.git
pip install -r requirements_instdiff_flux.txt
cd InstructDiffusion
bash scripts/download_pretrained_instructdiffusion.sh
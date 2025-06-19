# Setup Instructions for Synthetic Neurons

## To set up the synthetic neurons:

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

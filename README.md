## EDADepth: Enhanced Data Augmentation for Monocular Depth Estimation

### Installation:
Clone the repo and run the following commands:
```bash
cd EDADepth
conda env create -f edadepth_env.yml
conda activate EDADepth
```

### Preparing datasets:
Prepare Official NYUDepthv2 dataset following the instructions from [BTS](https://github.com/cleinc/bts/tree/master). 
After extracting the datasets, copy the datasets to the following directories:

NYUv2: 
```
EDADepth/depth/data/nyu
```
KITTI: 
```
EDADepth/depth/data/kitti
```
The dataset structure should look similar to the following:

```bash
nyu
├── nyu_depth_v2
│   ├── official_splits
│   └── sync
├── sync/

kitti
├──data_depth_annotated/
├──raw_data/
├──val_selection_cropped/
```
### Stable Diffusion Checkpoint
Please download the Stable-diffusion-v1-5-eamonly-pruned checkpoint from [this link](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt) and paste it to the following directory:

```commandline
EDADepth/checkpoints
```
### Evaluation using Pre-trained models:
If you want to evaluate the test dataset using our pre-trained models, you can download our pre-trained checkpoints. 

- Download our pre-trained checkpoint for NYUv2 dataset using [this link](https://drive.google.com/file/d/1vfQYNrL2jC12l_lnoYTM7RhIEh1grnX0/view)
- Download our pre-trained checkpoint for KITTI dataset using [this link](https://drive.google.com/file/d/1qMtKEcHM6qSSuairBW7FAL7QWcs5jsNB/view?usp=sharing)

After downloading the checkpoints, copy them and paste them to the following directory:
```
EDADepth/depth/checkpoints_depth
```
Finally, navigate to depth directory and run the following commands:

For NYUv2:
```bash
bash test_nyu.sh
```
For KITTI:
```bash
bash test_kitti.sh
```

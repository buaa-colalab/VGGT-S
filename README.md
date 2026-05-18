<div align="center">

<h1>VGGT-Segmentor: Geometry-Enhanced Cross-View Segmentation</h1>

<div>
    <a href='https://scholar.google.com/citations?user=s3u33VAAAAAJ&hl=en&oi=ao' target='_blank'>Yulu Gao*</a><sup>1</sup>&emsp;
    <a href='https://github.com/bohaocheung' target='_blank'>Bohang Zhang*</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=jrgMNxEAAAAJ&hl=en&oi=ao' target='_blank'>Zongheng Tang</a><sup>1</sup>&emsp;
    <a href='https://github.com/nikkukun' target='_blank'>Jitong Liao</a><sup>2</sup>&emsp;
    <a href='https://iai.buaa.edu.cn/info/1013/1093.htm' target='_blank'>Wenjun Wu</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=-QtVtNEAAAAJ&hl=en&oi=ao' target='_blank'>Si Liu†</a><sup>2</sup>
</div>
<div>
    <sup>1</sup>Hangzhou International Innovation Institute of Beihang University 1&emsp;
    <sup>2</sup>Beihang University 2
</div>

<div>
    <strong>CVPR 2026 Oral 🎉</strong>
</div>

<div>
    <h4 align="center">
        <a href="https://arxiv.org/abs/2604.13596" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2604.13596-b31b1b.svg">
        </a>
        <!-- <a href="URL_HERE" target='_blank'>
        <img src="https://img.shields.io/badge/Project-Page-green">
        </a> -->
        <a href="https://scholar.googleusercontent.com/scholar.bib?q=info:jq38NPHn4IQJ:scholar.google.com/&output=citation&scisdr=ClgBRwZHEN-wr8OW2eU:AFyMTJUAAAAAagaQweUMI9W3d4bE2S1_35XpMEM&scisig=AFyMTJUAAAAAagaQwVDNP28VXvXE-Xq1pW_q-jU&scisf=4&ct=citation&cd=0&hl=en" target='_blank'>
        <img src="https://img.shields.io/badge/Cite-BibTeX-blue">
        </a>
    </h4>
</div>

<strong>VGGT-S is a geometry-grounded segmentation framework built upon [VGGT](https://github.com/facebookresearch/vggt) for instance-level object segmentation across egocentric and exocentric views under large viewpoint, scale, and occlusion variations.</strong>

<div align="center">
    <img src="https://raw.githubusercontent.com/bohaocheung/VGGT-S-demo/main/compressed-repo-demo.gif" width="100%">
</div>

</div>

## 📢 News

* **[2026-05-15]** 🚀 Code and checkpoint are open-sourced.
* **[2026-04-15]** 🔥 VGGT-Segmentor paper is released on arxiv.


## 💡 Highlights

* **Union Segmentation Head**. We design this head to effectively translate VGGT’s high-level feature alignment into precise segmentation masks.
* **Single-Image Self-Supervised Training Strategy**. This strategy eliminates the need for paired annotations in correspondence tasks by deforming a single image to simulate different viewpoints.
* **New SOTA Results**. Our proposed VGGT-Segmentor achieves state-of-the-art results on EgoExo4d dataset.


## 🛠️ Usage
### Installation

#### Clone Repository

* Clone [VGGT-S](https://github.com/buaa-colalab/VGGT-S)

```bash
git clone https://github.com/buaa-colalab/VGGT-S.git
cd VGGT-S
```

#### Create Environment

```bash
conda create -n vggts python=3.10
conda activate vggts
```

#### Install VGGT Dependency

This project is built upon [VGGT](https://github.com/facebookresearch/vggt), proposed in the CVPR 2025 Best Paper:

> Wang et al., “VGGT: Visual Geometry Grounded Transformer”, CVPR 2025.

Please clone the official VGGT repository under `third_party`:

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/facebookresearch/vggt.git vggt_main
cd vggt_main
pip install -r requirements.txt # This is for VGGT
pip install -e .

# Last, download the VGGT Checkpoint, name it model.pt and place it under vggt_main folder.
```

The directory structure should look like:

```text
VGGT-S/
├── src/
├── data/
├── third_party/
│   └── vggt_main/
│           └── model.pt
└── requirements.txt
```

#### Install Requirements

```bash
cd VGGT-S
pip install -r requirements.txt # This is for VGGT-S
hf download zbbhhh/VGGT-S main_exp.pth --local-dir official_ckpts # Download checkpoint
```

---

### Modify VGGT Track Head

To extract VGGT encoder feature maps for downstream correspondence learning, please modify the `forward` function in:

```text
third_party/vggt_main/vggt/heads/track_head.py
```

Replace the original `forward` function (around Line 72) with the following implementation:

```python
def forward(self, aggregated_tokens_list, images, patch_start_idx, query_points=None, iters=None, feat=False):
    """
    Forward pass of the TrackHead.

    Args:
        aggregated_tokens_list (list): List of aggregated tokens from the backbone.
        images (torch.Tensor): Input images of shape (B, S, C, H, W), where:
                                B = batch size,
                                S = sequence length.
        patch_start_idx (int): Starting index for patch tokens.
        query_points (torch.Tensor, optional): Initial query points to track.
                                                If None, points are initialized automatically.
        iters (int, optional): Number of refinement iterations.
                                If None, uses self.iters.
        feat (bool, optional): Whether to additionally return feature maps.

    Returns:
        tuple:
            - coord_preds (torch.Tensor): Predicted coordinates.
            - vis_scores (torch.Tensor): Visibility scores.
            - conf_scores (torch.Tensor): Confidence scores.
            - feature_maps (torch.Tensor, optional): Extracted VGGT feature maps.
    """

    B, S, _, H, W = images.shape

    # Extract feature maps from VGGT tokens
    # Shape: (B, S, C, H//2, W//2)
    feature_maps = self.feature_extractor(
        aggregated_tokens_list,
        images,
        patch_start_idx
    )

    # Use default iterations if not specified
    if iters is None:
        iters = self.iters

    # Tracking
    coord_preds, vis_scores, conf_scores = self.tracker(
        query_points=query_points,
        fmaps=feature_maps,
        iters=iters
    )

    if feat:
        return coord_preds, vis_scores, conf_scores, feature_maps

    return coord_preds, vis_scores, conf_scores
```

---

### Dataset Preparation

#### Download Ego-Exo4D

Please follow the official [SegSwap Ego-Exo4D correspondence pipeline](https://github.com/EGO4D/ego-exo4d-relation/tree/main/correspondence/SegSwap) to:

1. Download the Ego-Exo4D dataset
2. Extract video frames into images
3. Generate correspondence pairs

After preprocessing, the dataset should be organized as:

```text
data_root/
├── take_id_01/
│   ├── ego_cam/
│   │   ├── 0.jpg
│   │   ├── ...
│   │   └── n.jpg
│   ├── exo_cam/
│   │   ├── 0.jpg
│   │   ├── ...
│   │   └── n.jpg
│   └── annotation.json
├── ...
├── take_id_n/
└── split.json
```

Then run the official SegSwap pair-generation script:

[create_pairs.py](https://github.com/EGO4D/ego-exo4d-relation/blob/main/correspondence/SegSwap/data/create_pairs.py)

This will generate:

```text
train_egoexo_pairs.json
train_exoego_pairs.json
val_egoexo_pairs.json
val_exoego_pairs.json
```

---

#### Configure Dataset Path

If the Ego-Exo4D dataset has already been downloaded, modify:

```text
src/dataloader.py
```

and set:

```python
EGOEXO4D_ROOT = "your/data/path"
```

---

### Data Preprocessing
> We provide all intermediate result files generated from the following preprocessing steps. Therefore, if you have already downloaded the Ego-Exo4D dataset, you can directly use the provided files without rerunning the preprocessing pipeline. Nevertheless, we still release the complete data preprocessing code to support further community research and help readers better understand our pipeline and methodology.

#### Generate Scene JSON
This step reorganizes the dataset structure and converts the pair_json to the following hierarchical JSON format:

```text
scene
└── [obj_info]
    ├── ego_info
    │   ├── ego_rgb
    │   └── ego_mask
    └── exo_info
        ├── exo_rfb
        └── exo_mask
```

where each `scene` contains multiple object annotations (`obj_info`), and each object stores the corresponding ego-view and exo-view information, including RGB frames (`*_rgb`) and segmentation masks (`*_mask`). Meanwhile, only objects that simultaneously appear in both ego and exo views are retained.


```bash
cd data
python gen_scenes.py

# Or, downloading from huggingface
huggingface-cli download zbbhhh/VGGT-S train_scenes.json --local-dir data
```

Outputs:

```text
train_scenes.json
val_scenes.json
```

---

#### Generate Single-Object JSON
This step converts `train_scenes.json` and `val_scenes.json` into a sample-based format, where each individual object instance is treated as a training sample for convenient dataloader indexing and loading.

Each sample follows the format:

```text
scene_id//ego_cam//exo_cam//object//frame_id
```

For example:

```text
c9e0bbae-4092-4ab1-95cd-16b905709a0e//aria01_214-1//gp01//white spatula_0//002160
```

which represents:

* `scene_id`: `c9e0bbae-4092-4ab1-95cd-16b905709a0e`
* `ego_cam`: `aria01_214-1`
* `exo_cam`: `gp01`
* `object`: `white spatula_0`
* `frame_id`: `002160`

```bash
python gen_one_object.py

# Or, downloading from huggingface
huggingface-cli download zbbhhh/VGGT-S train_obj.json --local-dir data
```

Outputs:

```text
train_obj.json
val_obj.json
```

---

#### Generate VGGT Projected Points
This step precomputes the VGGT projected points as the first-stage mapping points and saves them offline. By caching these mappings in advance, the training and inference stages can directly perform the second-stage mapping without rerunning the initial VGGT projection process, significantly reducing computation time.

Optionally, the first-stage mapping can also be performed online if needed.

##### Ego → Exo

```bash
python extract_point_g2x.py
```

Outputs:

```text
train_obj_wp_g2x.json
val_obj_wp_g2x.json
```

##### Exo → Ego

```bash
python extract_point_x2g.py
```

Outputs:

```text
train_obj_wp_x2g.json
val_obj_wp_x2g.json
```

---

#### Merge Point Annotations
This step merges the projected points generated in the previous step and produces the final data annotation JSON file used for training and inference.

```bash
python merge_point.py

# Or, downloading from huggingface
huggingface-cli download zbbhhh/VGGT-S train_obj_wp.json --local-dir data
```

Final outputs:

```text
train_obj_wp.json
val_obj_wp.json
```

---

### Training & Validation

Navigate to the experiment directory:

```bash
cd src/main_exp
```

Launch distributed training:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29600 \
    hybrid.py \
    --config hybrid.yaml
```

> Training and validation only require modifying the parameters in `hybrid.yaml`, where all configuration parameters are carefully explained.
---


## 📝 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{gao2026vggt,
  title={VGGT-Segmentor: Geometry-Enhanced Cross-View Segmentation},
  author={Gao, Yulu and Zhang, Bohao and Tang, Zongheng and Liao, Jitong and Wu, Wenjun and Liu, Si},
  journal={arXiv preprint arXiv:2604.13596},
  year={2026}
}
```

## 📄 License

This project is licensed under the Apache-2.0 License. See [LICENSE](LICENSE.txt) for more information.

## 🙏 Acknowledgement

This project is built upon several excellent open-source projects:

* [VGGT](https://github.com/facebookresearch/vggt)
* [EgoExo4d](https://github.com/EGO4D/ego-exo4d-relation)
* [SAM2](https://github.com/facebookresearch/sam2)

We thank the authors for releasing their code and models to the community.

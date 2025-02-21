# HECLIP

This code is prepared for "HECLIP: Histology-Enhanced Contrastive Learning for Imputation of Transcriptomics Profiles". In this work, we introduce HECLIP, an advanced deep learning framework that infers gene expression profiles directly from H&E-stained images, eliminating the need for costly spatial transcriptomics assays. By employing an innovative image-centric contrastive learning technique, HECLIP effectively captures critical morphological patterns and translates them into accurate gene expression profiles, enhancing predictive power while reducing dependency on molecular data. Extensive evaluations across publicly available datasets demonstrate HECLIP’s superior performance and scalability, making it a transformative and cost-effective tool for advancing precision medicine in both research and clinical settings. [paper](https://arxiv.org/pdf/2501.14948)

# Directory Structure


```plaintext
HECLIP-main/
├── README.md
├── environment.yaml
├── code/
    ├── ebd/
        ├── GSE240429/
            ├── heg/
            └── hvg/
        ├── GSE245620/
        ├── spatialLIBD_1/
        └── spatialLIBD_2/
    ├── figure/
        ├── EBD/
        └── EXP/
    ├── save/
        ├── GSE240429/
            ├── hvg_best.pt
            └── heg_best.pt
        ├── GSE245620/
        ├── spatialLIBD_1/
        └── spatialLIBD_2/
    ├── main.py
    ├── utils.py
    ├── infer.py
    └── ...
├── GSE240429/
    ├── data/
        ├── filtered_expression_matrices/
            ├── 1/
                ├── barcodes.tsv
                ├── features.tsv
                ├── matrix.mtx
                └── ...
            ├── 2/
            └── ...
        └── tissue_pos_matrices/
            ├── tissue_positions_list_1.csv
            ├── tissue_positions_list_2.csv
            └── ...
    └── image/
        ├── GEX_C73_A1_Merged.tiff
        └── ...
├── GSE245620/
    ├── data/
    └── image/
└── GSE/
    ├── std.py
    ├── spatialLIBD_1/
        ├── data/
        └── image/
    └── spatialLIBD_2/
        ├── data/
        └── image/
```

## Installation
Download HECLIP:
```git clone https://github.com/QSong-github/HECLIP```

Install Environment:
```conda env create -f environment.yaml```


## Running

### Prepare data.

   
   (1) Download the data.
       * [GSE240429](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240429)
       * [GSE245620](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE245620)
       * [spatialLIBD_1](https://research.libd.org/spatialLIBD/)
       * [spatialLIBD_2](https://research.libd.org/spatialLIBD/)

   Note: For spatialLIBD_1 and spatialLIBD_2, it is necessary to run ```std.py``` to convert the h5 to mtx format. 
   (spatialLIBD_1: 151507, 151508, 151509, 151510; spatialLIBD_2: 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676)


   (2) Preprocess the data.
   ```bash
   $ cd /path/to/code/
   $ python preprocess_GSE240429.py
   $ python preprocess_GSE245620.py
   $ python preprocess_spatialLIBD_1.py
   $ python preprocess_spatialLIBD_2.py
   ```
### Train the HECLIP.

   (3) Train the model.
   ```bash
   $ cd /path/to/code/
   $ python main.py
   ```
   
### Inference   

   (4) Inference.
   ```bash
   $ cd /path/to/code/
   $ python infer.py
   ```


## Quick start

If you want to skip the training, you can download the pre-trained HECLIP model from [here](https://drive.google.com/file/d/1q1MYoICLeY7w30CuT2eBxGw0kiHESMgK/view?usp=drive_link) and quickly try it by the ```'infer.py.'```

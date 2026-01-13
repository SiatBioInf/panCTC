In this Github repo, we developed a deep learning technology called panCTC to directly identify pan-cancer CTCs in peripheral blood and reliably trace their primary tumor lesions. The panCTC method relies on a unique parameter defined as chromatin unwinding segment (CUS), derived from single-cell transcriptomics, that indicates regions of active transcription. See [algorithm for calculating CUS](https://github.com/SiatBioInf/panCTC/blob/main/pics/cal_CUS.md) for details.

The workflow of the panCTC algorithm is shown in Figure (a) below:

![image](https://github.com/SiatBioInf/panCTC/blob/main/pics/a.png)

The first step involves converting single-cell RNA (scRNA) sequencing data into chromosome unwinding segments (CUSs). These specific CUSs, which are associated with different cell types, are utilized to train and validate both Model 1 (binary class) for classifying circulating tumor cells (CTCs) and Model 2 (multiclass) for identifying the corresponding primary cancer types of CTCs. This algorithm is called the panCTC algorithm. The performance of panCTC for CTC classification and identification of primary cancer types is evaluated using pseudoperipheral blood mononuclear cells (PBMCs). Pseudo-PBMCs are created by combining scRNA-seq data from primary tumors, metastatic tumors, healthy blood immune cells, and CTCs. Then panCTC is employed to identify CTCs in the PBMC scRNA-seq data obtained from tumor patients. The identified CTCs are validated based on their biological characteristics.


There are two progressive hypotheses, as in Figure (b), for identifying CTCs and primary cancer types: (1) CUSs are the general and intrinsic characteristics of cells and are related to cell type, and (2) the CUS characteristics of CTCs are mainly inherited from primary tumors. 

![image](https://github.com/SiatBioInf/panCTC/blob/main/pics/b.png)


The panCTC algorithm is constructed based on the CUS hypotheses.  

**Step 1**: The calculation of CUS values. In the scRNA data of a single cell, genes are aligned along chromosomes, and the level of the gene sequence in each segment is subsequently compared with the base level of the whole gene. 

**Step 2**: Model 1 for CTC classification. The model is based on an attention network and uses both immune-specific CUS features and tumor-specific CUS features to classify CTCs/cancer cells and immune cells. 

**Step 3**: Model 2 for identifying primary cancer types of CTCs. The model is still based on an attention network and uses specific CUSs from 12 cancer types to trace the sites of primary tumors from CTCs.

![image](https://github.com/SiatBioInf/panCTC/blob/main/pics/c-e.png)



PanCTC allows for the identification of rare CTCs, and has  capability to accurately track 12 types of primary tumor lesions solely from the analysis of a simple tube of peripheral blood (PBMC) based on scRNA-seq. The 12 cancer types are: 

- `0`  cervical cancer (CC)
- `1`  non-small cell lung cancer (NSCLC)
- `2`  colorectal cancer (CRC)
- `3`  pancreatic ductal adenocarcinoma (PDAC)
- `4`  nasopharyngeal carcinoma (NPC)
- `5`  endometrium cancer (EC)
- `6`  ovary cancer (OVC)
- `7`  breast cancer (BC)
- `8`  prostate cancer (PC)
- `9`  gastric cancer (GC)
- `10`  hepatocellular carcinoma (HCC)
- `11`  melanoma

It provides two tables of features. The first one contains immune features and pan-cancer features, which can be used to classify CTCs or cancer cells from blood immune cells through a pre-trained attention neural network of binary-classification (Model 1). The second one contains all CUSs used to identify the cancer type of a cell through a pre-trained attentional neural network of multi-classification (Model 2). 

# Requirements
## Python 3.9
- `numpy`==1.24.3
- `pandas`==1.5.3
- `torch`==2.0.1
- `rpy2`==3.5.13

## R 4.1.0
- `Seurat`==4.0.5
- `copykat`==1.0.8
- `Matrix`==1.4-0
- `dplyr`==1.0.7
- `purrr`==0.3.4
- `tidyr`==1.1.4


# Input

You need to prepare a gene expression count matrix of scRNA-seq for a PBMC sample, and save the count matrix to a `.rds` file. By using `Seurat` package, the count matrix should be gene symbols in rows and cell barcodes in columns.


# Usage

Firstly, the count matrix  `.rds` file should be located in the `input` folder.

Then, you just need to define `--sp_name` as the file name of the `.rds` file.

```
cd panCTC
python Py_panCTC_pred.py --sp_name BM_P34_primaryCancer_counts_edited
```

# Demo Video

A demonstration video of panCTC usage is available in the `pics` folder:

```
pics/demo_video.mp4
```

This video provides a visual guide on how to run the panCTC pipeline and interpret the results.


# Output

- `immune_cancer_label_position.csv`: This file is the selected features for model 1.

- `CUS_<sp_name>_bin10.Rdata`: The CUS matrix for all cells. After loading the `.Rdata`, you can use the object `CUS` in it, which is CUS breaks in rows and cell barcodes in columns.

- `Model1_input_<sp_name>.csv`: The input `.csv` file for Model 1. A CUS feature matrix only containing the selected CUS features in columns, and all cell barcodes in rows. 

- `predict_label_Model1_<sp_name>.csv`: This file shows you whether the cells are CTCs or immune cells. Predicted CTCs are `1`s, and predicted immune cells are `0`s. You may perform downstream analyses according to the cell barcodes of predicted CTCs.

- `Model2_input_<sp_name>.csv`: The input `.csv` file for Model 2, which contains all CUS values for predicted CTCs for predicting primary tumor lesions. 

- `predict_label_Model2_<sp_name>.csv`: This file shows you which cancer type the CTC is tracked. For each CTC, we provide the prior type and the top 3 predicted cancer type with probability. 

# More

author: "B. Ye", "Z. Wang"    
date: 2024-05-06

More information could be found in our paper: 
Bin Ye, Zhen Wang, Xu Zhang, Rui Zhang, Jingru Lian, Xuefei Liu, Yan Zhang, Zhiyuan Xu, Li Yang, Haiman Jin, Fang Chen, Zhihao Xie, Ping Zhou, Jun Tan, Shan Zeng, Changzheng Du, Yang Min, Huahui Li, Jingxian Duan, Zhicheng Li, Hui Yang, Yunpeng Cai, Hongyan Wu, Catherine C. Liu, Jing Cai, Hao Wu, Yi Lu, Jian Zhang, Xin Hong, Hairong Zheng and Hao Yu. PanCTC: In Situ Identification and Tracing of Circulating Tumor Cells Using Chromatin Unwinding Segment-Based Deep Learning (http://)(Submitting). 







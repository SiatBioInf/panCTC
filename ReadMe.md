In this Github repo, we developed a deep learning technology called panCTC to directly identify pan-cancer CTCs in peripheral blood and reliably trace their primary tumor lesions. The panCTC method relies on a unique parameter defined as chromatin unwinding segment (CUS), derived from single-cell transcriptomics, that indicates regions of active transcription. 

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
Bin Ye, Zhen Wang, Xu Zhang, Rui Zhang, Jingru Lian, Xuefei Liu, Yan Zhang, Zhiyuan Xu, Li Yang, Haiman Jin, Fang Chen, Zhihao Xie, Ping Zhou, Jun Tan, Shan Zeng, Changzheng Du, Yang Min, Huahui Li, Jingxian Duan, Zhicheng Li, Hui Yang, Yunpeng Cai, Hongyan Wu, Catherine C. Liu, Jing Cai, Hao Wu, Yi Lu, Jian Zhang, Xin Hong, Hairong Zheng and Hao Yu PanCTC: In Situ Identification and Tracing of Circulating Tumor Cells Using Chromatin Unwinding Segment-Based Deep Learning (http://). 








from rpy2 import robjects
import pandas as pd
import os

import sys
sys.path.append('./model1_binary')
from models_binary import Net_binary
from binary_predict import *

import sys
sys.path.append('./model2_multi')
from models_multi import Net_multi
from multi_predict import *

import argparse


#########################################################

# In this version, total 12 cancer types are composed of:
## cervical cancer (0),
## non-small cell lung cancer (1),
## colorectal cancer (2),
## pancreatic ductal adenocarcinoma (3),
## nasopharyngeal carcinoma (4),
## endometrium cancer (5),
## ovary cancer (6),
## breast cancer (7),
## prostate cancer (8),
## gastric cancer (9),
## hepatocellular carcinoma (10),
## melanoma (11).

#########################################################

def main(sp_name):

    #### Calculate CUSs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    r = robjects.r

    CUS_path = "./CUS_" + sp_name + "_bin10.Rdata"
    if os.path.exists(CUS_path):
        r['load'](CUS_path)
        CUS1 = r['CUS']
        r.source("./R_panCTC_func0.R")  # Output file for Model 1 input
        rfunc0 = robjects.globalenv['func0']
        smp = robjects.vectors.StrVector([sp_name])
        rfunc0(CUS1, smp)
    else:
        r.source("./R_panCTC_func1.R")
        rfunc1 = robjects.globalenv['func1']
        smp = robjects.vectors.StrVector([sp_name])

        CUS1 = rfunc1(smp)  # Output file for Model 1 input

    #### Model 1
    file1 = "./Model1_input_" + sp_name + ".csv"
    model1_input = pd.read_csv(file1, index_col=0)

    device = "cpu"
    model1 = load_binary_checkpoint("./model1_binary/checkpoint_binary.pth", device)
    y_pred = binary_predict(model1_input, model1, device)
    model1_pred = pd.DataFrame(index=model1_input.index)
    model1_pred['Sample_name'] = sp_name
    model1_pred['Predict'] = y_pred
    pred_label1 = "./predict_label_Model1_" + sp_name + ".csv"
    model1_pred.to_csv(pred_label1)


    #### Model 2
    r.source("./R_panCTC_func2.R")
    rfunc2 = robjects.globalenv['func2']
    smp = robjects.vectors.StrVector([sp_name])
    out2 = rfunc2(CUS1, smp)

    if len(out2) > 0:
        file2 = "./Model2_input_" + sp_name + ".csv"
        model2_input = pd.read_csv(file2, index_col=0)
        model2 = load_multi_checkpoint("./model2_multi/checkpoint_multi.pth", device)

        model2_pred = multi_pred(model2_input, model2, device)
        pred_label2 = "./predict_label_Model2_"  + sp_name + ".csv"
        model2_pred.to_csv(pred_label2)
    else:
        print('There are no CTCs need to tracing their cancer type!')


###################################################################
## Enter the parameter of patient name 

# sp_name = "PBMC_GSE156405_PDAC_P2"
# sp_name = "BM_P34_primaryCancer_counts_edited"

####################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sp_name', type=str, help='patient name', default='PBMC_GSE156405_PDAC_P2')
    
    args = parser.parse_args()
    sp_name = args.sp_name
    main(sp_name)





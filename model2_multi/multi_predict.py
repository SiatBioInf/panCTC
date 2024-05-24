import numpy as np
import pandas as pd
import torch

from models_multi import Net_multi


# Use model2 to predict the cancer type
def multi_pred(df, model, device):
    '''
    Params:
        df: DataFrame, with last column of label
    Returns:
        pred_df: DataFrame, prediction result
    '''
    
    x = torch.tensor(df.values).float().to(device)

    with torch.no_grad():
        model.eval()
        model = model.to(device)
        top_class, top_p = model.predict(x)

        pred_df = pd.DataFrame(columns=['Label1st', 'Prob1st', 'Label2nd', 'Prob2nd', 'Label3rd', 'Prob3rd'])
        pred_df[['Label1st', 'Label2nd', 'Label3rd']] = top_class.cpu().numpy()
        pred_df[['Prob1st', 'Prob2nd', 'Prob3rd']] = np.round(top_p.cpu().numpy(), 4)

        cancer_dict = {
                    0: 'cervical cancer',
                    1: 'non-small cell lung cancer',
                    2: 'colorectal cancer',
                    3: 'pancreatic ductal adenocarcinoma',
                    4: 'nasopharyngeal carcinoma',
                    5: 'endometrium cancer',
                    6: 'ovary cancer',
                    7: 'breast cancer',
                    8: 'prostate cancer',
                    9: 'gastric cancer',
                    10: 'hepatocellular carcinoma',
                    11: 'melanoma'}
        pred_df = pred_df.replace(cancer_dict)

    result = round(pred_df['Label1st'].value_counts()/len(pred_df), 4)[:3].to_frame().reset_index()
    result.columns = ['Pred_label', 'Probability']
    result = result.replace(cancer_dict)
    print(result)

    return pred_df


# load model2
def load_multi_checkpoint(filepath, device):

    checkpoint = torch.load(filepath, map_location=device)
    model = Net_multi(checkpoint['input_size'], checkpoint['hidden_size'], checkpoint['n_input'],
                      checkpoint['d_model'], checkpoint['ffn_hidden'], checkpoint['n_head'], 
                      checkpoint['n_layers'], checkpoint['drop_prob'], checkpoint['n_hidden'])
  
    model.load_state_dict(checkpoint['state_dict'])

    return model

eturn model


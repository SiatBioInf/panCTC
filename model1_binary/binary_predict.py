
import pandas as pd
import torch

from models_binary import Net_binary

# use model1 to identify CTC
def binary_predict(df, model, device):
    '''
    df: DataFrame
    '''
    x = torch.tensor(df.values).float().to(device)

    with torch.no_grad():
        model.eval()
        model = model.to(device)
        y_pred = model.predict(x).cpu().numpy()

    return y_pred       


# load the model1
def load_binary_checkpoint(filepath, device):

    checkpoint = torch.load(filepath, map_location=device)

    pos_info = pd.read_csv("./immune_cancer_label_position.csv", header=0,
                        names=['auto_pos', 'pos', 'pos_binary'])
    pos_info['auto_pos'] = pos_info['auto_pos'] - 1
    pos = pos_info.iloc[:, checkpoint['pos_n']]

    model = Net_binary(checkpoint['d_model'], pos, checkpoint['ffn_hidden'], 
                        checkpoint['n_head'], checkpoint['n_layers'], 
                        checkpoint['drop_prob'], checkpoint['n_hidden'], 
                        checkpoint['n_input'], device=device)
   
    model.load_state_dict(checkpoint['state_dict'])
    return model

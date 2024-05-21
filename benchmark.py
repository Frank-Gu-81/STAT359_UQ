import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lib.metrics import All_Metrics
from lib.dataloader import get_dataloader
from model.AGCRN_UQ import AVWDCRNN
from model.AGCRN_UQ import AGCRN_UQ as Network
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
import argparse
import configparser
from model.test_methods import combined_test
import matplotlib.pyplot as plt

#*************************************************************************#
Mode = 'train'
DEBUG = 'True'
DATASET = 'PEMS04'      #PEMS04/8/3/7
DEVICE = 'cuda:0'
MODEL = 'AGCRN'
MODEL_NAME = "combined"#"combined/basic/dropout/heter"

PEMS03_MODEL = "./model/saved_model/PEMS03_saved_model.pth"
PEMS04_MODEL = "./model/saved_model/PEMS04_saved_model.pth"
PEMS08_MODEL = "./model/saved_model/PEMS08_saved_model.pth"

# PEMS03_DATA = pd.read_csv("./data/PEMS03/PEMS03.csv")
# PEMS04_DATA = pd.read_csv("./data/PEMS04/PEMS04.csv")
# PEMS08_DATA = pd.read_csv("./data/PEMS08/PEMS08.csv")

config_file = 'model/{}_{}.conf'.format('PEMS04', 'AGCRN')
config = configparser.ConfigParser()
config.read(config_file)

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
#data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
#model
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
args.add_argument('--p1', default=config['model']['p1'], type=float)

#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
#args.add_argument('--loss_func', default='mse', type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
#args.add_argument('--epochs', default=500, type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
#args.add_argument('--lr_init', default=1e-2, type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
#test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args.add_argument('--model_name', default=MODEL_NAME, type=str)

#save model
args.add_argument('--save_path', default='./model/saved_model/', type=str)
args.add_argument('--save_filename', default='{}_{}.pth'.format(DATASET, 'saved_model'), type=str)


args = args.parse_args([])
init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

#init model
model = Network(args).to(args.device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)

def load_model(model_path):
    # model = AVWDCRNN(model_name, p1, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers)
    state_dict = torch.load(model_path)
    
    # Debugging: Print keys in the state_dict and model's state_dict
    print("State Dict Keys (from file):", state_dict.keys())
    print("****************************************")
    print("Model State Dict Keys (expected):", model.state_dict().keys())
    print("****************************************")
    print("Are the keys the same?", state_dict.keys() == model.state_dict().keys())
    print("****************************************")
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def metrics():
    # mae, rmse, mape = benchmark()
    # print("MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape*100))
    load_model(PEMS04_MODEL)
    train_loader, _, test_loader, scaler = get_dataloader(args,
                                                            normalizer=args.normalizer,
                                                            tod=args.tod, dow=False,
                                                            weather=False, single=False)
    combined_test(model, 10, args, test_loader, scaler, T=torch.zeros(1), logger=None, path=None)

def generate_uncertainty_qualification():
    load_model(PEMS04_MODEL)
    train_loader, _, test_loader, scaler = get_dataloader(args,
                                                            normalizer=args.normalizer,
                                                            tod=args.tod, dow=False,
                                                            weather=False, single=False)
    y_pred, y_true, aleatoric_uncertainty, epistemic_uncertainty, total_std, lower_bond, upper_bond = combined_test(model, 10, args, test_loader, scaler, T=torch.zeros(1), logger=None, path=None)
    save_uncertainty_qualification(y_pred, y_true, aleatoric_uncertainty, epistemic_uncertainty, total_std, lower_bond, upper_bond)


def plot_uncertainty_qualification():
    aleatoric_uncertainty = np.load("aleatoric_uncertainty.npy")
    print(aleatoric_uncertainty.shape)
    aleatoric_uncertainty = aleatoric_uncertainty.mean(axis=(1)) / 10000
    print(aleatoric_uncertainty.shape)
    print(f'Aleatoric Uncertainty shape {aleatoric_uncertainty.shape}')

    epistemic_uncertainty = np.load("epistemic_uncertainty.npy")
    epistemic_uncertainty = epistemic_uncertainty.mean(axis=(1))
    print(epistemic_uncertainty)
    print(f'Epistemic Uncertainty shape {epistemic_uncertainty.shape}')

    y_pred = np.load("y_pred.npy")
    y_pred = y_pred.mean(axis=(1,2))
    print(f'y_pred shape {y_pred.shape}')
    y_true = np.load("y_true.npy")
    y_true = y_true.mean(axis=(1,2))
    print(f'y_true shape {y_true.shape}')

    total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
    total_std = total_uncertainty**0.5

    print(f'Total Uncertainty shape {total_uncertainty.shape}')

    print(f'Total Uncertainty shape {total_uncertainty.shape}')
    time_steps = np.arange(y_pred.shape[0])
    print(f'Time steps shape {time_steps.shape}')
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, y_true, label='True', color='black')
    plt.plot(time_steps, y_pred, label='Predicted', color='red')
    plt.fill_between(time_steps, y_pred-1.96*total_std, y_pred+1.96*total_std,
                        color='blue', alpha=0.3, label='Total Uncertainty')
    plt.xlabel('Time/Hour')
    plt.ylabel('Traffic Flow')
    plt.legend()
    plt.title('Uncertainty Quantification Results')
    plt.show()


def random_RS_plot_uncertainty_qualification():
    aleatoric_uncertainty = np.load("aleatoric_uncertainty.npy")
    print(aleatoric_uncertainty.shape)
    aleatoric_uncertainty = aleatoric_uncertainty[1670:2300,:,3]
    print("after: ", aleatoric_uncertainty.shape)
    aleatoric_uncertainty = aleatoric_uncertainty.mean(axis=(1))
    print(aleatoric_uncertainty.shape)
    print(f'Aleatoric Uncertainty shape {aleatoric_uncertainty.shape}')

    epistemic_uncertainty = np.load("epistemic_uncertainty.npy")
    epistemic_uncertainty = epistemic_uncertainty[1670:2300,:,3]
    epistemic_uncertainty = epistemic_uncertainty.mean(axis=(1))
    print(epistemic_uncertainty)
    print(f'Epistemic Uncertainty shape {epistemic_uncertainty.shape}')

    y_pred = np.load("y_pred.npy")
    print(f'y_pred shape before {y_pred.shape}')
    y_pred = y_pred[1670:2300, :, 3]
    y_pred = y_pred.mean(axis=(1))
    print(f'y_pred shape {y_pred.shape}')
    y_true = np.load("y_true.npy")
    y_true = y_true[1670:2300, :, 3]
    y_true = y_true.mean(axis=(1))
    print(f'y_true shape {y_true.shape}')

    total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
    total_std = total_uncertainty**0.5

    print(f'Total Uncertainty shape {total_uncertainty.shape}')

    print(f'Total Uncertainty shape {total_uncertainty.shape}')
    time_steps = np.arange(y_pred.shape[0])
    print(f'Time steps shape {time_steps.shape}')
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, y_true, label='True', color='black')
    plt.plot(time_steps, y_pred, label='Predicted', color='red')
    plt.fill_between(time_steps, y_pred-1.96*total_std, y_pred+1.96*total_std,
                        color='blue', alpha=0.3, label='Total Uncertainty')
    plt.xlabel('Time/Hour')
    plt.ylabel('Traffic Flow')
    plt.legend()
    plt.title('Uncertainty Quantification Results')
    plt.show()

def save_uncertainty_qualification(y_pred, y_true, aleatoric_uncertainty, epistemic_uncertainty, total_std, lower_bound, upper_bound):
    y_pred = y_pred.cpu().numpy().squeeze()
    np.save("y_pred.npy", y_pred)
    y_pred = y_pred.mean(axis=2)
    y_true = y_true.cpu().numpy().squeeze()
    np.save("y_true.npy", y_true)
    y_true = y_true.mean(axis=2)
    total_std = total_std.cpu().numpy().squeeze()
    aleatoric_uncertainty = aleatoric_uncertainty.cpu().numpy().squeeze()
    # aleatoric_uncertainty = aleatoric_uncertainty.mean(axis=2)
    epistemic_uncertainty = epistemic_uncertainty.cpu().numpy().squeeze()
    # epistemic_uncertainty = epistemic_uncertainty.mean(axis=2)
    total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
    print(y_pred.shape, y_true.shape, aleatoric_uncertainty.shape, epistemic_uncertainty.shape, total_std.shape, lower_bound.shape, upper_bound.shape)

    print("aleatoric_uncertainty:", aleatoric_uncertainty, "\n")
    print("epistemic_uncertainty:", epistemic_uncertainty, "\n")
    print("total_uncertainty:", total_uncertainty, "\n")

    # save aleatoric_uncertainty, epistemic_uncertainty, total_uncertainty
    np.save("aleatoric_uncertainty.npy", aleatoric_uncertainty)
    np.save("epistemic_uncertainty.npy", epistemic_uncertainty)
    np.save("total_uncertainty.npy", total_uncertainty)

if __name__ == "__main__":
    # metrics()
    # generate_uncertainty_qualification()
    # plot_uncertainty_qualification()
    random_RS_plot_uncertainty_qualification()

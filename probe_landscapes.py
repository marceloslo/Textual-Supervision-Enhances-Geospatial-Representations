import torch
import numpy as np
import os, json
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import h5py
import time
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
from utils import HDF5ImageDataset, VisionTransformerWrapper, get_model_traces, train_ridge, train_mlp
import pandas as pd

def get_activation_regression(representations,y,bias,result_dir,probe="ridge"):
    n_samples, L, D = representations.shape
    results = {}
    best_r = -10000
    best_y = None
    os.makedirs(os.path.join(result_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "representations"), exist_ok=True)
    for l in range(L):
        print(f"layer {l}===========================")
        X = representations[:, l, :]  # shape: (n_samples, d)
        if probe=="ridge":
            mses,rmses,maes,r2s,spr,y_hat,ids,model,stds = train_ridge(X,y,results,bias=bias)
        else:
            mses,rmses,maes,r2s,spr,y_hat,ids,model,stds = train_mlp(X,y,results,bias=bias)
        results[l] = {
            'MSE': np.mean(mses),
            'RMSE': np.mean(rmses),
            'MAE': np.mean(maes),
            'R2': np.mean(r2s),
            'r2_error': np.std(r2s,ddof=1) / np.sqrt(len(r2s)),
            'spearmanr': np.mean(spr),
            "std":stds.tolist()
        }
        if probe == "ridge":
            results[l]['latitude_coefficients'] = model.coef_[0].tolist()  
            results[l]['longitude_coefficients'] = model.coef_[1].tolist()  
            results[l]['intercept'] = model.intercept_.tolist()  # store intercept too if useful
        y_true = y[ids]
        pred_df = pd.DataFrame({
                "id": ids,
                "predicted_latitude": y_hat[:, 0],
                "predicted_longitude": y_hat[:, 1],
                "true_latitude": y_true[:, 0],
                "true_longitude": y_true[:, 1]
            })
        pred_path = os.path.join(result_dir, "predictions", f"layer_{l}.csv")
        pred_df.to_csv(pred_path, index=False)
        
        if results[l]['R2'] > best_r:
                best_r = results[l]['R2']
                best_y = np.stack(y_hat)
    return results, best_y



def plot_map(data,n=None, preds= None,filename=None):
    if n is not None:
        indices = np.sort(np.random.choice(data["latitudes"].shape[0],n,replace=False))
    latitudes = data["latitudes"][:]
    longitudes = data["longitudes"][:]

    fig, ax = plt.subplots(1,1,figsize=(15,7.5))
    ax.scatter(longitudes,latitudes)
    if preds is not None:
        ax.scatter(preds[:,1],preds[:,0])
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    #clear plot
    plt.clf()

def main(experiment,data_file,model,args):
    print(f"[INFO] Starting model {model}")
    #make directory for results
    model_name = model.split("/")[-1]
    result_dir = f"results/{experiment}/{model_name}"
    if args.probe !="ridge":
        result_dir+="/"+args.probe
    if os.path.exists(f"{result_dir}/landmark_activation.json"):
        print(f"[INFO] Skipping model {model} as results already exist.")
        return
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = VisionTransformerWrapper(model, device=device, prompt=args.prompt)

    def get_labels_idx(data):
        latitudes = data["latitudes"][:]
        longitudes = data["longitudes"][:]
        return np.concat([latitudes.reshape(-1,1),longitudes.reshape(-1,1)],axis=1)

    h5_file = h5py.File(data_file, 'r')
    y = get_labels_idx(h5_file)
    
    #load the dataset
    dataset = HDF5ImageDataset(data_file, transform=None)
    dataloader = DataLoader(dataset, batch_size = 1, num_workers = 4, pin_memory = True, prefetch_factor = 4, persistent_workers = True, shuffle = False, drop_last = False)


    mode = wrapper.get_modality()
    attentions, activations = get_model_traces(wrapper,dataloader,aggregation=args.aggregation)
    y = y[:len(activations)]
    

    activation_results, best_y = get_activation_regression(activations,y,bias=bool(args.bias),result_dir=result_dir+f"/landmarks",probe=args.probe)
    plot_map(h5_file,preds=best_y,filename=f"{result_dir}/landmark_activation.png")
    #save the results as json
    with open(f"{result_dir}/landmark_activation.json", "w") as f:
        json.dump(activation_results, f)

    if mode =="text":
        print(f"[INFO] Starting Text")
        text_attentions,text_activations = get_model_traces(wrapper,dataloader,mode,aggregation=args.aggregation)
        
        activation_results, best_y = get_activation_regression(text_activations,y,bias=bool(args.bias),result_dir=result_dir+f"/landmarks_text",probe=args.probe)
        plot_map(h5_file,preds=best_y,filename=f"{result_dir}/text_landmark_activation.png")
        #save the results as json
        with open(f"{result_dir}/text_landmark_activation.json", "w") as f:
            json.dump(activation_results, f)
    print(f"[INFO] Elapsed time:{(time.time()-start_time)/60:.2f} minutes. Finished running {model} for landscapes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--geotagged_file','-f', type=str, default='../vit/sample_landmarks.hdf5', help='Path to the geotagged file')
    parser.add_argument('--model','-m',type=str,required=True)
    parser.add_argument('--aggregation',type=str,required=True,default="cls")
    parser.add_argument('--bias',type=int,required=True,default=0)
    parser.add_argument('--probe',type=str,default="ridge")
    parser.add_argument('--prompt',type=str,default=None)
    args = parser.parse_args()
    main(args.experiment,args.geotagged_file,args.model,args)

    

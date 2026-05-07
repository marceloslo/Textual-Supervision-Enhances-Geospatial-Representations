import torch
import numpy as np
import os, json
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt
import h5py
import time, os
import argparse
import pandas as pd
from utils import HDF5ImageDataset, VisionTransformerWrapper, get_model_traces, train_ridge, train_mlp


def get_activation_regression(representations,y,bias,result_dir,probe="ridge"):
    os.makedirs(os.path.join(result_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "representations"), exist_ok=True)

    n_samples, L, D = representations.shape
    results = {}
    best_r = -10000
    best_y = None
    for l in range(L):
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
                # np.save(os.path.join(result_dir, "representations", "best_activation.npy"),
                #             representations[:, l, :])
    return results, best_y


def plot_map(indices,data,n=None, preds= None,filename=None):
    if n is not None:
        indices = np.sort(np.random.choice(indices,n,replace=False))
    latitudes = data["latitudes"][:][indices]
    longitudes = data["longitudes"][:][indices]

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


def main(experiment,geotagged_file,model,args):
    h5_file = h5py.File(geotagged_file, 'r')
    clusters = np.unique(h5_file["clusters"][:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = VisionTransformerWrapper(model, device=device, prompt=args.prompt)
    for cluster in clusters:
        #make directory for results
        model_name = model.split("/")[-1]
        result_dir = f"results/{experiment}/{model_name}"
        if args.probe !="ridge":
            result_dir+="/"+args.probe
        #only run the cluster if it is not already done
        if (wrapper.get_modality()=="text") and not os.path.exists(f"{result_dir}/text_activation_{cluster.decode()}.json"):
            pass
        elif os.path.exists(f"{result_dir}/activation_{cluster.decode()}.json"):
            print(f"[INFO] Skipping cluster {cluster.decode()} for model {model} as results already exist.")
            continue
            
        print(f"[INFO] Starting cluster {cluster.decode()} for model {model}")
        start_time = time.time()

        def get_cluster_idx(data,cluster): 
            idx = np.where(data["clusters"][:]==cluster)[0]
            return idx

        def get_labels_idx(data,cluster_indexes):
            latitudes = data["latitudes"][:][cluster_indexes]
            longitudes = data["longitudes"][:][cluster_indexes]
            return np.concat([latitudes.reshape(-1,1),longitudes.reshape(-1,1)],axis=1)

        sample_size=None
        cluster_indexes = get_cluster_idx(h5_file,cluster,sample_size)
        y = get_labels_idx(h5_file,cluster_indexes)
        
        #load the dataset
        dataset = HDF5ImageDataset(geotagged_file, transform=None, custom_indexes=cluster_indexes, max_size=sample_size)
        dataloader = DataLoader(dataset, batch_size = 1, num_workers = 4, pin_memory = True, prefetch_factor = 4, persistent_workers = True, shuffle = False, drop_last = False)


        mode = wrapper.get_modality()
        
        
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(result_dir+f"/{cluster.decode()}", exist_ok=True)
        
        attentions, activations = get_model_traces(wrapper,dataloader,aggregation=args.aggregation)
        y = y[:len(activations)]

        activation_results, best_y = get_activation_regression(activations,y,bias=bool(args.bias),result_dir=result_dir+f"/{cluster.decode()}",probe=args.probe)
        plot_map(cluster_indexes,h5_file,preds=best_y,filename=f"{result_dir}/activation_{cluster.decode()}.png")
        #save the results as json
        with open(f"{result_dir}/activation_{cluster.decode()}.json", "w") as f:
             json.dump(activation_results, f)

        del attentions, activations
        #get the text results if they exist
        if mode=="text":
            text_attentions,text_activations = get_model_traces(wrapper,dataloader,mode,aggregation=args.aggregation)
            print(f"[INFO] Starting Text")

            activation_results, best_y = get_activation_regression(text_activations,y,bias=bool(args.bias),result_dir=result_dir+f"/{cluster.decode()}_text",probe=args.probe)
            plot_map(cluster_indexes,h5_file,preds=best_y,filename=f"{result_dir}/text_activation_{cluster.decode()}.png")
            #save the results as json
            with open(f"{result_dir}/text_activation_{cluster.decode()}.json", "w") as f:
                json.dump(activation_results, f)
            del text_attentions,text_activations

        print(f"[INFO] Elapsed time:{(time.time()-start_time)/60:.2f} minutes. Finished running {model} on {cluster.decode()}")
        torch.cuda.empty_cache()

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


    

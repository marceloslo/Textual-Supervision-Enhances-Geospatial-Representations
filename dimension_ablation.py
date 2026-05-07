import torch
import torch.nn as nn
import numpy as np
import cv2, os, json
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CLIPVisionModel, CLIPImageProcessor,
    ViTModel, ViTForImageClassification, ViTImageProcessor,
    ViTMAEForPreTraining, AutoImageProcessor,
)
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from baukit import Trace, TraceDict
from einops import rearrange
from scipy.stats import spearmanr
import time, os
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import pandas as pd
from utils import HDF5ImageDataset, VisionTransformerWrapper, get_model_traces, train_ridge, train_mlp



def collect_coeffs(file_path):
    print(file_path)
    # Collect R² values from cluster files
    with open(file_path,"r") as file:
        data = json.load(file)
    print(data.keys())
    df = pd.DataFrame.from_dict(data,orient="index")
    #identify last layer index
    last_layer = df.index[-1]
    print(last_layer)
    last_layer_data = data[last_layer]
    last_layer_data["layer"]=int(last_layer)
    #get absolute coefficient value
    last_layer_data["absolute_latitude"] = np.absolute(last_layer_data["latitude_coefficients"])
    last_layer_data["absolute_longitude"] = np.absolute(last_layer_data["longitude_coefficients"])
    #negate the array to sort descending order
    last_layer_data["latitude_indexes"]= np.argsort(-last_layer_data["absolute_latitude"])
    last_layer_data["longitude_indexes"]= np.argsort(-last_layer_data["absolute_longitude"])

    #identify best_layer_index
    best_layer = df[df["R2"]==df[:-1]["R2"].max()].index[0]
    print(best_layer)
    best_layer_data = data[best_layer]
    best_layer_data["layer"]=int(best_layer)
    best_layer_data["absolute_latitude"] = np.absolute(best_layer_data["latitude_coefficients"])
    best_layer_data["absolute_longitude"] = np.absolute(best_layer_data["longitude_coefficients"])
    best_layer_data["latitude_indexes"]= np.argsort(-best_layer_data["absolute_latitude"])
    best_layer_data["longitude_indexes"]= np.argsort(-best_layer_data["absolute_longitude"])

    first_layer = '0'
    print(first_layer)
    first_layer_data = data[first_layer]
    first_layer_data["layer"]=int(first_layer)
    first_layer_data["absolute_latitude"] = np.absolute(first_layer_data["latitude_coefficients"])
    first_layer_data["absolute_longitude"] = np.absolute(first_layer_data["longitude_coefficients"])
    first_layer_data["latitude_indexes"]= np.argsort(-first_layer_data["absolute_latitude"])
    first_layer_data["longitude_indexes"]= np.argsort(-first_layer_data["absolute_longitude"])
    return last_layer_data,best_layer_data, first_layer_data
    
def run_two_ridges(X_lat,X_lon,y_lat,y_lon, k=5, bias=False):
    skf = KFold(n_splits=k, shuffle=True, random_state=42)

    mses, rmses, maes, r2s = [], [], [], []
    models = []
    y_hat = []
    ids = []
    for train_idx, test_idx in skf.split(X_lat, y_lat):
        # Train/test split for latitude
        X_lat_train, X_lat_test = X_lat[train_idx], X_lat[test_idx]
        y_lat_train, y_lat_test = y_lat[train_idx], y_lat[test_idx]

        # Train/test split for longitude
        X_lon_train, X_lon_test = X_lon[train_idx], X_lon[test_idx]
        y_lon_train, y_lon_test = y_lon[train_idx], y_lon[test_idx]

        # Two ridge regressions
        model_lat = RidgeCV(alphas=np.logspace(-1, 4, 12), fit_intercept=bias)
        model_lon = RidgeCV(alphas=np.logspace(-1, 4, 12), fit_intercept=bias)

        model_lat.fit(X_lat_train, y_lat_train)
        model_lon.fit(X_lon_train, y_lon_train)

        y_lat_pred = model_lat.predict(X_lat_test)
        y_lon_pred = model_lon.predict(X_lon_test)

        # Stack predictions and ground truth for joint metrics
        y_test = np.stack([y_lat_test, y_lon_test], axis=1)
        y_pred = np.stack([y_lat_pred, y_lon_pred], axis=1)

        # Metrics
        mses.append(mean_squared_error(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        maes.append(mean_absolute_error(y_test, y_pred))
        r2s.append(r2_score(y_test, y_pred))

        models.append((model_lat, model_lon))
        y_hat.extend(y_pred)
        ids.extend(test_idx)

    best_idx = np.argmax(r2s)
    best_models = models[best_idx]
    y_hat = np.stack(y_hat)

    return mses, rmses, maes, r2s, y_hat, ids, best_models
    
def get_one_layer_result(representations,y,layer_data,size,bias):
    latitude_indexes = layer_data["latitude_indexes"][:size]
    longitude_indexes = layer_data["longitude_indexes"][:size]
    
    X_lat = representations[:,layer_data["layer"],latitude_indexes]
    X_lon = representations[:,layer_data["layer"],latitude_indexes]
    mses,rmses,maes,r2s,y_hat,ids,model = run_two_ridges(X_lat,X_lon,y[:,0],y[:,1],bias=bias)
    results = {
                'MSE': np.mean(mses),
                'RMSE': np.mean(rmses),
                'MAE': np.mean(maes),
                'R2': np.mean(r2s),
            }
    return results

def get_ablation_regression(representations,y,bias,result_dir,file_path):
    output_dir =  os.path.join(result_dir, "ablation")
    os.makedirs(output_dir, exist_ok=True)

    last_layer_data, best_layer_data, first_layer_data = collect_coeffs(os.path.join(result_dir,file_path))
    n_samples, L, D = representations.shape
    results = {}
    for fraction in np.linspace(0.1,1,10):
        size = round(D*fraction)
        #first get the last layer
        last_layer_results = get_one_layer_result(representations, y, last_layer_data, size, bias)
        #then get best layer
        best_layer_results = get_one_layer_result(representations, y, best_layer_data, size, bias)
        #then get first layer
        first_layer_results = get_one_layer_result(representations, y, first_layer_data, size, bias)

        results["{:.1f}".format(fraction)] =  {
            best_layer_data["layer"] : best_layer_results,
            last_layer_data["layer"] : last_layer_results, 
            first_layer_data["layer"] : first_layer_results
        }
    with open(os.path.join(output_dir,file_path),"w") as file:
        json.dump(results,file)
    return results

def main(experiment,model,args):
    # for model in ["google/vit-base-patch16-224","google/vit-large-patch16-224","openai/clip-vit-large-patch14", 
    #             "openai/clip-vit-base-patch32","facebook/vit-mae-base","facebook/vit-mae-large","geolocal/StreetCLIP",
    #             "llava-hf/llava-1.5-7b-hf"]:
    h5_file = h5py.File("../vit/geocells_clusters.hdf5", 'r')
    clusters = np.unique(h5_file["clusters"][:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = VisionTransformerWrapper(model, device=device, prompt=args.prompt)
    for cluster in clusters:
        #make directory for results
        model_name = model.split("/")[-1]
        result_dir = f"results/{experiment}/{model_name}"
        if args.probe !="ridge":
            raise Exception("Invalid probe",args.probe)
        #only run the cluster if it is not already done
        if os.path.exists(f"{result_dir}/ablation/text_activation_{cluster.decode()}.json"):
            print(f"[INFO] Skipping cluster {cluster.decode()} for model {model} as results already exist.")
            continue
        print(f"[INFO] Starting cluster {cluster.decode()} for model {model}")
        start_time = time.time()

        def get_cluster_idx(data,cluster,sample_size=None): 
            idx = np.where(data["clusters"][:]==cluster)[0]
            if sample_size is not None or sample_size<idx.shape[0]:
                np.random.seed(42)
                idx=np.random.choice(idx, sample_size, replace=False)
            return idx

        def get_labels_idx(data,cluster_indexes):
            latitudes = data["latitudes"][:][cluster_indexes]
            longitudes = data["longitudes"][:][cluster_indexes]
            return np.concat([latitudes.reshape(-1,1),longitudes.reshape(-1,1)],axis=1)

        sample_size=5000
        cluster_indexes = get_cluster_idx(h5_file,cluster,sample_size)
        y = get_labels_idx(h5_file,cluster_indexes)
        
        #load the dataset
        dataset = HDF5ImageDataset("../vit/geocells_clusters.hdf5", transform=None, custom_indexes=cluster_indexes, max_size=sample_size)
        dataloader = DataLoader(dataset, batch_size = 1, num_workers = 4, pin_memory = True, prefetch_factor = 4, persistent_workers = True, shuffle = False, drop_last = False)


        mode = wrapper.get_modality()
        
        _, activations = get_model_traces(wrapper,dataloader,aggregation=args.aggregation)
        y = y[:len(activations)]
        results = get_ablation_regression(activations,y,bool(args.bias),result_dir,f"activation_{cluster.decode()}.json")
        

        del activations
        #get the text results if they exist
        if mode=="text":
            _,text_activations = get_model_traces(wrapper,dataloader,mode,aggregation=args.aggregation)
            print(f"[INFO] Starting Text")
            results = get_ablation_regression(text_activations,y,bool(args.bias),result_dir,f"text_activation_{cluster.decode()}.json")
            del text_activations
        print(f"[INFO] Elapsed time:{(time.time()-start_time)/60:.2f} minutes. Finished running {model} on {cluster.decode()}")
        torch.cuda.empty_cache()

    #################################################################LANDMARK####################################################################################
    
    print(f"[INFO] Starting landmarks for model {model}")
    if os.path.exists(f"{result_dir}/ablation/landmark_activation.json"):
        print(f"[INFO] Skipping model {model} as results already exist.")
        return
    start_time = time.time()
    def get_labels_idx(data):
        latitudes = data["latitudes"][:]
        longitudes = data["longitudes"][:]
        return np.concat([latitudes.reshape(-1,1),longitudes.reshape(-1,1)],axis=1)

    data_file="../vit/filtered_landmarks.hdf5" 
    h5_file = h5py.File(data_file, 'r')
    #_,indices = np.unique(h5_file["landmark_ids"],return_index=True)
    y = get_labels_idx(h5_file)
    
    #load the dataset
    dataset = HDF5ImageDataset("../vit/filtered_landmarks.hdf5", transform=None)
    dataloader = DataLoader(dataset, batch_size = 1, num_workers = 4, pin_memory = True, prefetch_factor = 4, persistent_workers = True, shuffle = False, drop_last = False)


    mode = wrapper.get_modality()
    _, activations = get_model_traces(wrapper,dataloader,aggregation=args.aggregation)
    y = y[:len(activations)]
    results = get_ablation_regression(activations,y,bool(args.bias),result_dir,f"landmark_activation.json")
    

    if mode =="text":
        print(f"[INFO] Starting Text")
        _,text_activations = get_model_traces(wrapper,dataloader,mode,aggregation=args.aggregation)
        results = get_ablation_regression(text_activations,y,bool(args.bias),result_dir,f"text_landmark_activation.json")
#################################################################WIKI####################################################################################
    
    print(f"[INFO] Elapsed time:{(time.time()-start_time)/60:.2f} minutes. Finished running {model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--model','-m',type=str,required=True)
    parser.add_argument('--aggregation',type=str,required=True,default="cls")
    parser.add_argument('--bias',type=int,required=True,default=0)
    parser.add_argument('--probe',type=str,default="ridge")
    parser.add_argument('--prompt',type=str,default=None)
    args = parser.parse_args()
    main(args.experiment,args.model,args)


    

import torch
import torch.nn as nn
from torchvision.transforms.functional import resize
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CLIPVisionModel, CLIPImageProcessor,
    ViTModel, ViTForImageClassification, ViTImageProcessor,
    ViTMAEForPreTraining, AutoImageProcessor,
    AutoProcessor, LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,
    SamModel, SamProcessor, AutoModel, Gemma3ForConditionalGeneration,AutoConfig
)
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from baukit import Trace, TraceDict
from einops import rearrange
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import os
#from qwen_vl_utils import process_vision_info


class HDF5ImageDataset(Dataset):
    def __init__(self, h5_path, transform=None, max_size=None, custom_indexes=None):
        self.h5_path = h5_path
        self.transform = transform
        self.max_size = max_size
        self.custom_indexes = custom_indexes
        self._init_file()

    def _init_file(self):
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.images = self.h5_file["images"]
        if self.custom_indexes is not None:
            self.length = len(self.custom_indexes)
        else:
            self.length = len(self.images)
        if self.max_size is not None:
            self.length = min(self.max_size, self.length)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.custom_indexes is not None:
            idx = self.custom_indexes[idx]
        img_bytes = self.images[idx]
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL-like format (HWC) then apply torchvision transforms
        if self.transform:
            img = self.transform(img)

        return img

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()

## Wrapper core class for all models
class VisionTransformerWrapper:
    def __init__(self, model_name: str, device: str = 'cpu',prompt=None):
        self.model_name = model_name.lower()
        self.device = device
        self.model, self.processor = self._load_model_and_processor(model_name)
        if isinstance(self.model, torch.nn.Module):
            self.model.to(device)
        self.prompt=prompt

        #RANDOMIZING PROMPT CODE
        # self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", trust_remote_code=True)
        # special_tokens = set([t.content for t in self.tokenizer.added_tokens_decoder.values()])
        # self.token_dist = [t for t in self.tokenizer.vocab if t not in special_tokens]
        # self.seed_counter = 0

    def _load_model_and_processor(self, name: str):
        if "clip" in name.lower():
            model = CLIPVisionModel.from_pretrained(name)
            processor = CLIPImageProcessor.from_pretrained(name)
        elif "dinov3" in name.lower():
            processor = AutoImageProcessor.from_pretrained(name)
            model = AutoModel.from_pretrained(name)
        elif "sam" in name.lower():
            model = SamModel.from_pretrained(name)
            processor = SamProcessor.from_pretrained(name)
        elif "vit-mae" in name.lower():
            model =  ViTMAEForPreTraining.from_pretrained(name)
            processor = ViTImageProcessor.from_pretrained(name)
        elif "webssl-mae" in name.lower():
            model =  ViTModel.from_pretrained(name)
            processor = AutoImageProcessor.from_pretrained(name)
        elif "vit" in name.lower():
            model = ViTForImageClassification.from_pretrained(name)
            processor = ViTImageProcessor.from_pretrained(name)
        elif "dino" in name.lower():
            processor = AutoImageProcessor.from_pretrained(name)
            model = AutoModel.from_pretrained(name)
        elif "llava" in name.lower():
            model = LlavaForConditionalGeneration.from_pretrained(name,torch_dtype=torch.float16, low_cpu_mem_usage=True,)
            processor = AutoProcessor.from_pretrained(name)
        elif "qwen" in name.lower():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                name,
                torch_dtype=torch.bfloat16,
            )
            processor = AutoProcessor.from_pretrained(name)
        elif "gemma" in name.lower():
            model = Gemma3ForConditionalGeneration.from_pretrained(name, torch_dtype=torch.bfloat16).eval()
            processor = AutoProcessor.from_pretrained(name)
        else:
            raise ValueError(f"Unknown model name: {name}")
        return model, processor

    def get_attention_modules(self,mode="vision"):
        if mode=="text":
            return self.get_text_attention_modules()
        if "clip" in self.model_name:
            return [f"vision_model.encoder.layers.{i}.self_attn" for i in range(self.model.config.num_hidden_layers)]
        elif "sam" in self.model_name:
            return [f"vision_encoder.layers.{i}.attn" for i in range(self.model.config.vision_config.num_hidden_layers)]
        elif "vit" in self.model_name:
            return [f"vit.encoder.layer.{i}.attention" for i in range(self.model.config.num_hidden_layers)]
        elif "webssl-mae" in self.model_name:
            return [f"encoder.layer.{i}.attention" for i in range(self.model.config.num_hidden_layers)]
        elif "dino" in self.model_name:
            return [f"encoder.layer.{i}.attention" for i in range(self.model.config.num_hidden_layers)]
        elif "llava" in self.model_name:
            return [f"vision_tower.vision_model.encoder.layers.{i}.self_attn" for i in range(self.model.config.vision_config.num_hidden_layers)] 
                    #+[f"language_model.model.layers.{i}.self_attn" for i in range(self.model.config.text_config.num_hidden_layers)])
        elif "qwen" in self.model_name:
            return [f"visual.blocks.{i}.attn" for i in range(self.model.config.vision_config.depth)]
        elif "gemma" in self.model_name:
            return [f"vision_tower.vision_model.encoder.layers.{i}.self_attn" for i in range(self.model.config.vision_config.num_hidden_layers)]
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
    
    def __get_text_residuals(self):
        if "llava" in self.model_name:
            return [f"language_model.model.layers.{i}" for i in range(self.model.config.text_config.num_hidden_layers)]
        elif "qwen" in self.model_name:
            return [f"model.layers.{i}" for i in range(self.model.config.num_hidden_layers)]
        elif "gemma" in self.model_name:
            return [f"language_model.model.layers.{i}" for i in range(self.model.config.text_config.num_hidden_layers)]
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
            
    def __get_residuals(self,mode="vision"):
        if mode=="text":
            return self.__get_text_residuals()
        if "clip" in self.model_name:
            return [f"vision_model.encoder.layers.{i}" for i in range(self.model.config.num_hidden_layers)]
        elif "dinov3" in self.model_name:
            return [f"model.layer.{i}" for i in range(self.model.config.num_hidden_layers)]
        elif "sam" in self.model_name:
            return [f"vision_encoder.layers.{i}" for i in range(self.model.config.vision_config.num_hidden_layers)]
        elif "vit" in self.model_name:
            return [f"vit.encoder.layer.{i}" for i in range(self.model.config.num_hidden_layers)]
        elif "webssl-mae" in self.model_name:
            return [f"encoder.layer.{i}" for i in range(self.model.config.num_hidden_layers)]
        elif "dino" in self.model_name:
            return [f"encoder.layer.{i}" for i in range(self.model.config.num_hidden_layers)]
        elif "llava" in self.model_name:
            return [f"vision_tower.vision_model.encoder.layers.{i}" for i in range(self.model.config.vision_config.num_hidden_layers)]
                    #+[f"language_model.model.layers.{i}.mlp" for i in range(self.model.config.text_config.num_hidden_layers)])
        elif "qwen" in self.model_name:
            return [f"visual.blocks.{i}" for i in range(self.model.config.vision_config.depth)]
        elif "gemma" in self.model_name:
            return [f"vision_tower.vision_model.encoder.layers.{i}" for i in range(self.model.config.vision_config.num_hidden_layers)]
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
    
    def get_activation_outputs(self,mode="vision",residuals=True):
        if residuals:
            return self.__get_residuals(mode)
        if mode=="text":
            return self.get_text_activation_outputs()
        if "clip" in self.model_name:
            return [f"vision_model.encoder.layers.{i}.mlp" for i in range(self.model.config.num_hidden_layers)]
        elif "sam" in self.model_name:
            return [f"vision_encoder.layers.{i}.mlp" for i in range(self.model.config.vision_config.num_hidden_layers)]
        elif "vit" in self.model_name:
            return [f"vit.encoder.layer.{i}.output.dense" for i in range(self.model.config.num_hidden_layers)]
        elif "dino" in self.model_name:
            return [f"encoder.layer.{i}.mlp" for i in range(self.model.config.num_hidden_layers)]
        elif "llava" in self.model_name:
            return [f"vision_tower.vision_model.encoder.layers.{i}.mlp" for i in range(self.model.config.vision_config.num_hidden_layers)]
                    #+[f"language_model.model.layers.{i}.mlp" for i in range(self.model.config.text_config.num_hidden_layers)])
        elif "qwen" in self.model_name:
            return [f"visual.blocks.{i}.mlp" for i in range(self.model.config.vision_config.depth)]
        elif "gemma" in self.model_name:
            return [f"vision_tower.vision_model.encoder.layers.{i}.mlp" for i in range(self.model.config.vision_config.num_hidden_layers)]
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def get_text_attention_modules(self):
        if "llava" in self.model_name:
            return [f"language_model.model.layers.{i}.self_attn" for i in range(self.model.config.text_config.num_hidden_layers)]
        elif "qwen" in self.model_name:
            return [f"model.layers.{i}.self_attn" for i in range(self.model.config.num_hidden_layers)]
        elif "gemma" in self.model_name:
            return [f"language_model.model.layers.{i}.self_attn" for i in range(self.model.config.text_config.num_hidden_layers)]
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def get_text_activation_outputs(self):
        if "llava" in self.model_name:
            return [f"language_model.model.layers.{i}.mlp" for i in range(self.model.config.text_config.num_hidden_layers)]
        elif "qwen" in self.model_name:
            return [f"model.layers.{i}.mlp" for i in range(self.model.config.num_hidden_layers)]
        elif "gemma" in self.model_name:
            return [f"language_model.model.layers.{i}.mlp" for i in range(self.model.config.text_config.num_hidden_layers)]
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
    
    def process_image(self,image):
        if "llava" in self.model_name:
            messages = [
                {
            
                  "role": "user",
                  "content": [
                      {"type": "image"},
                    ],
                },
            ]
            if self.prompt:
                messages[0]["content"].append({"type":"text","text":self.prompt})
            messages = self.processor.apply_chat_template(messages, add_generation_prompt=True)   
            return self.processor(images=image, text=messages, return_tensors="pt").to(self.device)
            
        if "qwen" in self.model_name:
            if (image.shape[0] < 28) or (image.shape[1] < 28):
                aux_img = image.permute(2,0,1)
                aux_img = resize(aux_img, (max(aux_img.shape[-2],28), max(aux_img.shape[-1],28))) 
                aux_img = aux_img.permute(1,2,0)
                image = aux_img
            if (image.shape[0] > 800) or (image.shape[1] > 800):
                aux_img = image.permute(2,0,1)
                aux_img = resize(aux_img, (min(aux_img.shape[-2],800), min(aux_img.shape[-1],800))) 
                aux_img = aux_img.permute(1,2,0)
                image = aux_img
            messages =[
            {
        
              "role": "user",
              "content": [
                  {"type": "image"},
                ],
            },
            ]
            if self.prompt:
                messages[0]["content"].append({"type":"text","text":self.prompt})
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=image.permute(2,0,1),
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            return inputs
        elif "gemma" in self.model_name:
            prompt = "<start_of_image>"
            if self.prompt:
                prompt += " " + self.prompt
            return self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        return self.processor(images=image, return_tensors="pt").to(self.device)

    def get_cls_id(self,mode="vision"):
        if mode=="text" or ("qwen" in self.model_name) or ("gemma" in self.model_name):
            return -1
        return 0
    
    def get_num_heads(self,mode="vision"):
        if "llava" in self.model_name and mode=="vision":
            heads= self.model.config.vision_config.num_attention_heads
        elif "llava" in self.model_name:
            heads= self.model.config.text_config.num_attention_heads
        elif "qwen" in self.model_name and mode =="vision":
            heads= self.model.config.vision_config.num_heads
        elif "gemma" in self.model_name:
            if mode=="text":
                heads= self.model.config.text_config.num_attention_heads
            else:
                heads= self.model.config.vision_config.num_attention_heads
        else:
            heads = self.model.config.num_attention_heads
        return heads 
    
    def forward(self, image, randomize_prompt=False):
        if randomize_prompt:
            rng = np.random.default_rng(seed=self.seed_counter)
            arr = rng.choice(self.token_dist,size=20,replace=True)
            self.prompt = ''.join(arr)
            self.seed_counter+=1
        with torch.no_grad():
            inputs = self.process_image(image)
        #if "clip" in self.model_name:
            #outputs = self.model.get_image_features(**inputs)
            if "llava" in self.model_name or "qwen" in self.model_name or "gemma" in self.model_name:
                outputs = self.model.generate(**inputs,max_new_tokens=1)
            else:
                outputs = self.model(**inputs)
        return outputs
    def get_modality(self):
        if "llava" in self.model_name or "qwen" in self.model_name or "gemma" in self.model_name:
            return "text"
        return "vision"

def check_trace_output(tensor):
    if isinstance(tensor, tuple):
        return True
    elif isinstance(tensor,torch.Tensor):
        return tensor.dim()>2
    else:
        raise Exception("Unexpected output for trace dict",type(tensor))


#Getting attentions was removed from this function, but it can be added back if needed. Just make sure to add the attention modules to the trace dict and extract them in the same way as the activations.
#Also, getting text activations and vision activations together was removed due to memory constraints.
def get_model_traces(wrapper,dataloader,mode="vision",aggregation="cls"):
    ACTIVATIONS = wrapper.get_activation_outputs()
    if mode=="text":
        TEXT_ACTIVATIONS = wrapper.get_activation_outputs(mode)
    else:
        TEXT_ACTIVATIONS=[]
    cls_activation_list = []
    cls_text_activation_list=[]
    for image in tqdm(dataloader):
        image = image.squeeze()
        with torch.no_grad():
            with TraceDict(wrapper.model, ACTIVATIONS+TEXT_ACTIVATIONS) as ret:
                outputs = wrapper.forward(image)

                if mode == "vision":
                    #get the activation outputs from the trace dict
                    activations = [ret[activation].output[0].squeeze().detach().cpu() if check_trace_output(ret[activation].output)  else ret[activation].output.squeeze().detach().cpu() for activation in ACTIVATIONS]
                    activations = torch.stack(activations, dim = 0).squeeze().float()
                    if aggregation == "cls":
                        #extract the cls token from the token activations
                        cls_activation_list.append(activations.numpy()[:, wrapper.get_cls_id(), :].copy())
                    elif aggregation == "maxmin":
                        activation_max = torch.max(activations,dim=1)[0]
                        activation_min = torch.min(activations,dim=1)[0]
                        cls_activation_list.append(torch.concat((activation_max,activation_min),dim=1).numpy().copy())

                #extract the LM activations
                if mode=="text":

                    #get the head outputs from the trace dict
                    activations = [ret[activation].output[0].squeeze().detach().cpu()  if check_trace_output(ret[activation].output) else ret[activation].output.squeeze().detach().cpu() for activation in TEXT_ACTIVATIONS]
                    activations = torch.stack(activations, dim = 0).squeeze().float()
                    if aggregation == "cls":
                        #extract the cls token from the token activations
                        cls_text_activation_list.append(activations.numpy()[:, wrapper.get_cls_id(mode), :].copy())
                    elif aggregation=="maxmin":
                        activation_max = torch.max(activations,dim=1)[0]
                        activation_min = torch.min(activations,dim=1)[0]
                        cls_text_activation_list.append(torch.concat((activation_max,activation_min),dim=1).numpy().copy())
                

    if mode=="text":
        text_activation_representations = np.stack(cls_text_activation_list)
        return [], text_activation_representations
    else:    
        #format activations into a examples x layers x hidden_dim array
        activation_representations = np.stack(cls_activation_list)
        return [], activation_representations
    return [],activation_representations,[],text_activation_representations


def train_ridge(X,y,results, k=5, bias=False):
    seed = int(os.environ.get("MY_EXP_SEED",42))
    skf = KFold(n_splits=k, shuffle=True, random_state=seed)

    mses, rmses, maes, r2s, spr,stds = [], [], [], [], [], []
    models = []
    y_hat = []
    ids = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RidgeCV(alphas= np.logspace(-1, 4, 12), fit_intercept=bias)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(model.alpha_)

        mses.append(mean_squared_error(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        maes.append(mean_absolute_error(y_test, y_pred))
        r2s.append(r2_score(y_test, y_pred))
        spr.append(spearmanr(y_test, y_pred).statistic)
        models.append(model)
        y_hat.extend(y_pred)
        ids.extend(test_idx)
        stds.append(np.std(X_train,axis=0))
    best_idx = np.argmax(r2s)
    model = models[best_idx]
    y_hat=np.stack(y_hat)
    return mses,rmses,maes,r2s,spr,y_hat,ids,model, stds[best_idx]

def train_mlp(X,y,results, k=5, bias=False):
    skf = KFold(n_splits=k, shuffle=True, random_state=42)

    mses, rmses, maes, r2s, spr, stds = [], [], [], [], [], []
    models = []
    y_hat = []
    ids = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = MLPRegressor(
            hidden_layer_sizes=(256,),
            learning_rate_init=1e-3,
            alpha=0.01,
            solver="adam",
            max_iter=100,
            early_stopping=True
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mses.append(mean_squared_error(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        maes.append(mean_absolute_error(y_test, y_pred))
        r2s.append(r2_score(y_test, y_pred))
        spr.append(spearmanr(y_test, y_pred).statistic)
        models.append(model)
        y_hat.extend(y_pred)
        ids.extend(test_idx)
        stds.append(np.std(X_train,axis=0))
    best_idx = np.argmax(r2s)
    model = models[best_idx]
    std = stds[best_idx]
    y_hat=np.stack(y_hat)
    return mses,rmses,maes,r2s,spr,y_hat,ids,model, std

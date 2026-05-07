
from transformers import AutoImageProcessor
from datasets import Dataset
import json
import argparse
from PIL import Image
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
from utils import HDF5ImageDataset
import torch
import numpy as np
from evaluate import load
from transformers import ViTForImageClassification, AutoModelForImageClassification
from transformers import TrainingArguments
from transformers import Trainer

learning_rates = [1e-5,2e-5,5e-5,1e-4,2e-4,5e-4]

def collate_fn(batch):
    return {
        "pixel_values": torch.tensor(np.stack([x["pixel_values"] for x in batch])),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }

metric = load("accuracy", keep_in_memory=True)
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    
def main(model):
    model_name_or_path = model
    processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    
    def format_data(image,label,processor):
        inputs = processor(image, return_tensors="np")  # numpy, not pt
        return {
            "pixel_values": inputs["pixel_values"].squeeze(),  # remove batch dim
            "labels": id2label[label],
        }


    def get_labels_idx(data):
        return [country.decode() for country in data["countries"][:]]
    data_file="../vit/balanced_landmarks.h5" 
    h5_file = h5py.File(data_file, 'r')
    #_,indices = np.unique(h5_file["landmark_ids"],return_index=True)
    y = get_labels_idx(h5_file)
    
    #load the dataset
    dataset = HDF5ImageDataset(data_file, transform=None)
    country, counts = np.unique(y,return_counts=True)
    
    id2label= {c:i for i,c in enumerate(country)}
    indices = [i for i in range(len(y)) if y[i] in country]
    data = [format_data(dataset[i],y[i],processor) for i in indices]
    train_dataset,test_dataset = train_test_split(data,train_size=0.7,random_state=42,stratify=[d["labels"] for d in data])
    val_dataset,test_dataset = train_test_split(test_dataset,train_size=0.66,random_state=42)
    labels =len(id2label)

    for i,learning_rate in enumerate(learning_rates):
        if "mae" in model_name_or_path:
            model = ViTForImageClassification.from_pretrained(
                model_name_or_path,
                num_labels=labels,
                id2label=id2label,
                label2id={c: str(i) for c, i in id2label.items()}
            )
        else:
            model = AutoModelForImageClassification.from_pretrained(
                model_name_or_path,
                num_labels=labels,
                id2label=id2label,
                label2id={c: str(i) for c, i in id2label.items()}
            )
        save_dir = f"./{model_name_or_path.split("/")[1]}_{i}"
        training_args = TrainingArguments(
          output_dir=save_dir,
          per_device_train_batch_size=16,
          eval_strategy="steps",
          num_train_epochs=5,
          save_steps=120,
          eval_steps=40,
          logging_steps=20,
          learning_rate=2e-5,
          save_total_limit=2,
          remove_unused_columns=False,
          push_to_hub=False,
          load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=processor,
        )
        trainer.train()
        trainer.save_model()
        json.dump(trainer.state.log_history, open(f"./logs/{model_name_or_path.split("/")[1]}_{i}.json","w"))






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-m',type=str,required=True)
    args = parser.parse_args()
    main(args.model)

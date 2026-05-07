This is the repository for the paper **Textual Supervision Enhances Geospatial Representations in Vision-Language Models** to be published at ICML 2026.

# Instructions

EvaluateAttacks.ipynb: contains the code for the embedding swapping results and evaluation

ICLR_plots.ipynb and Results.ipynb: contains the code for generating the figures in the paper

run_landscape.py: file responsible for running the landscape dataset analysis

run_probing.py: file responsible for running the clustered dataset analysis

utils.py: file that has most of the actual code, including dataset reader, model wrapper (and all required model information), and code for extracting the activations and attention heads, if you want to add a model, you need to update this code.

finetune.py: contains the code for the finetuning for the country classification task used in Section 4.6 in the paper

## To run

All scripts assume the existence of the landmarks and yfcc100m datasets, which we can't share here, so download and subsample those (as per our appendix) before running the script.
Alternatively, you may use any dataset together with the probe_landmarks.py script as long as it is an h5 file with attributes "images", "latitudes", and "longitudes". 
The run_probing.py script additionally requires a "clusters" attribute, but functionality is otherwise the same.

Simply use the slurm parallel_probing.sh script for the vision-only models, and the language_parallel_probing.sh script for the vision-language models.
They will automatically generate most of the main paper results in the results folder.
For identical results to the paper, use the arguments CLS, 1 and ridge.

Script usage:

```
sbatch parallel_probing.sh EXPERIMENT_NAME AGGREGATION(CLS or MAXMIN) BIAS(0 or 1) REGRESSION(ridge or mlp)
```

example: 
```
cd ./slurm
sbatch parallel_probing.sh EXPERIMENT_NAME CLS 1 ridge
```


To run the finetune experiments, use the script finetune.sh, which requires a subset of the landmarks dataset that has a balanced distribution of countries.

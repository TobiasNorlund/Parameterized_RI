import sys

import embedding
import model
import dataset

emb_str = sys.argv[1]
model_str = sys.argv[2]
dataset_str = sys.argv[3]

## --- DATASET ---

if dataset_str == "PL05":
    ds = dataset.PL05()
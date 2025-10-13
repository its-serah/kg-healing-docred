# === CONFIG ===
import os, json, numpy as np, pandas as pd, networkx as nx
from datetime import datetime

DATA_DIR   = "data"                             # where *_revised.json live
EMB_CACHE  = "embeddings/transformer"           # unzipped entity_{i}.npy per doc
OUT_ROOT   = "experiments"                      # all outputs go here
SPLIT      = "dev"                              # train|dev|test
N_DOCS     = 25                                 # dev subset size (e.g., 25)
ROTATE_NPY = None                               # path to RotatE .npy or None

os.makedirs(OUT_ROOT, exist_ok=True)

# If you already loaded these earlier, skip:
def _load_json(fp):
    with open(fp, "r") as f: return json.load(f)

try:
    train_data, dev_data, test_data
except NameError:
    train_data = _load_json(f"{DATA_DIR}/train_revised.json")
    dev_data   = _load_json(f"{DATA_DIR}/dev_revised.json")
    test_data  = _load_json(f"{DATA_DIR}/test_revised.json")

split_map = {"train": train_data, "dev": dev_data, "test": test_data}
DOCS = split_map[SPLIT][:N_DOCS]

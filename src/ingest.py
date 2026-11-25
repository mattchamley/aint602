# AINT602 ML Pipeline Source Ingestion File
# Matt Chamley 26 November 2026

import hashlib
import pandas as pd
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(path):
    return pd.read_csv(path)


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Karl Backend Package

import os


os.environ.setdefault("HF_HOME", "/tmp/karl-hf-home")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/karl-hf-home/transformers")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

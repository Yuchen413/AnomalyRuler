# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from PIL import Image
from tqdm import tqdm
import torch
from utils import *

np.random.seed(2024)
torch.manual_seed(2024)

device = "cuda" if torch.cuda.is_available() else "cpu"
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")
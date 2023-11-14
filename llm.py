# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
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

def lamma2():
    model_path = "meta-llama/Llama-2-70b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,  low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map ='auto')
    txt_path = 'SHTech/object_data/test_50_0_owlvit.txt'
    objects = read_txt_to_list(txt_path)
    inputs = []
    for obj in objects[0:2]:
        input_text = (
                    f"""
                    [INST]\n As a surveillance monitor for urban safety using the ShanghaiTech dataset, my job is to analyze
                    the video footage and identify anything that could be considered abnormal behavior or activity. 
                    I will be given a rule that indicates normal or anomaly, for example:
                    
                    Rules for normal:
        
                    1. [Person, Walking, Paved path] = normal, because it is common for people to walk on walkways.
                    2. [Person, Standing walking, Roadside] = normal, because people often stand or walk by the roadside.
                    3. [Multiple people, Walking, open area] = normal, because groups commonly walk together in open areas.
                    4. [Multiple people, Walking, Paved path] = normal, because walkways are designed for pedestrian traffic.
                    5. [Multiple people, Walking, Roadside] = normal, because walking alongside roads is typical pedestrian behavior.
        
                    Rules for anomaly:
        
                    1. [Person, Not walking, Paved walkway] = anomaly, because walkways are meant for walking, not loitering or unusual behavior.
                    2. [Person, Not standing/walking, Roadside] = anomaly, because people are expected to be either standing or moving by the roadside, not engaging in other non-pedestrian activities.
                    3. [Multiple people, Not walking, Spacious paved area] = anomaly, because in such areas, it's unusual to see people completely stationary or performing non-walking related activities.
                    4. [Multiple people, Crowded or disorganized, Paved walkway] = anomaly, because walkways are meant for orderly pedestrian flow.
                    5. [Multiple people, Walking in the road, Roadside] = anomaly, because pedestrians should be walking on the walkway, not on the road.
        
                    Question: now I am monitoring the campus and I'm given:{obj}, is it normal or anomaly?' \n[\INST]\n\n
                    """)

        inputs.append(input_text)

    input_ids = tokenizer.encode(inputs, return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=500)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

def solar():
    txt_path = 'SHTech/object_data/test_50_0_owlvit.txt'
    objects = read_txt_to_list(txt_path)
    tokenizer = AutoTokenizer.from_pretrained("upstage/Llama-2-70b-instruct-v2")
    model = AutoModelForCausalLM.from_pretrained(
        "upstage/Llama-2-70b-instruct-v2",
        low_cpu_mem_usage=True,
        device_map="auto",
        torch_dtype=torch.float16,
        rope_scaling={"type": "dynamic", "factor": 2}  # allows handling of longer inputs
    )

    system = (
        f"""
                As a surveillance monitor for urban safety using the ShanghaiTech dataset, my job is to analyze
                the video footage and identify anything that could be considered abnormal behavior or activity. 
                I will be given a rule that indicates normal or anomaly, for example:

                Rules for normal:

                1. [Person, Walking, Paved path] = normal, because it is common for people to walk on walkways.
                2. [Person, Standing walking, Roadside] = normal, because people often stand or walk by the roadside.
                3. [Multiple people, Walking, open area] = normal, because groups commonly walk together in open areas.
                4. [Multiple people, Walking, Paved path] = normal, because walkways are designed for pedestrian traffic.
                5. [Multiple people, Walking, Roadside] = normal, because walking alongside roads is typical pedestrian behavior.

                Rules for anomaly:

                1. [Person, Not walking, Paved walkway] = anomaly, because walkways are meant for walking, not loitering or unusual behavior.
                2. [Person, Not standing/walking, Roadside] = anomaly, because people are expected to be either standing or moving by the roadside, not engaging in other non-pedestrian activities.
                3. [Multiple people, Not walking, Spacious paved area] = anomaly, because in such areas, it's unusual to see people completely stationary or performing non-walking related activities.
                4. [Multiple people, Crowded or disorganized, Paved walkway] = anomaly, because walkways are meant for orderly pedestrian flow.
                5. [Multiple people, Walking in the road, Roadside] = anomaly, because pedestrians should be walking on the walkway, not on the road.
                """)

    for obj in objects[0:2]:
        prompt = f"### System:\n{system}\n\n### User:\n now I am monitoring the campus and I'm given:{obj}, is it normal or anomaly? Why' \n\n### Assistant:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        output = model.generate(**inputs, use_cache=True, max_new_tokens=float('inf'))
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(output_text)

    # output = model.generate(**inputs, streamer=streamer, use_cache=True, max_new_tokens=float('inf'))
    # output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(output_text)

solar()
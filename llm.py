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


def mixtral_induct(desc_path='SHTech/object_data/train_100_0_vicuna-7b-v1.5_act+env.txt'):
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                 device_map='auto').eval()
    answers = []
    filename = f"rule/{desc_path.split('/')[-1].split('.')[0]}_mistral7B.txt"
    if os.path.exists(filename):
        os.remove(filename)
        print(f"File {filename} has been deleted.")
    else:
        print(f"The file {filename} does not exist.")

    objects = read_txt_to_one_list(desc_path)
    print(objects)
    # for obj in objects:
    text = f'''Given the description, your task is to summarize the normal activities or normal objects.
        Now you are given the description {objects}, 
        Answer:
        1.'''
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=1000)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answers.append(answer)
    # print(f'-----------------------------------------------------{count}-----------------------------------------------------------------------')
    print(answer)
    # with open(filename, 'a') as file:
    #     for answer in answers:
    #         file.write(
    #             f'-----------------------------------------------------{count}-----------------------------------------------------------------------' + '\n')
    #         file.write(answer + '\n')
    return

def mixtral_deduct(desc_path_n, desc_path_a,  rule_path):
    # model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map ='auto').eval()

    desc_paths = [desc_path_n,desc_path_a]
    answers = []
    preds = []
    labels = [0]*50 + [1]*50

    filename = f"results/{rule_path.split('/')[1].split('.')[0]}_mistral7B_{desc_path_n.split('/')[-1].replace('test_50_0_','')}"
    if os.path.exists(filename):
        os.remove(filename)
        print(f"File {filename} has been deleted.")
    else:
        print(f"The file {filename} does not exist.")

    count = 0
    for desc_path in desc_paths:
        objects = read_line(desc_path)
        rule = open(rule_path, "r").read()

        for obj in objects:
            count+=1
            text = f'''You are monitoring the campus, you task is to detect the anomaly based on the given rules. The rules are: 
            {rule}. 
            Now you are given {obj}. Is it normal or anomaly? Answer Anomaly even if only one Anomaly exists. Otherwise answer Normal. 
                       
            Answer:'''
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=1000)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            preds.append(post_process(answer))
            answers.append(answer)
            print(f'-----------------------------------------------------{count}-----------------------------------------------------------------------')
            print(answer)
            with open(filename, 'a') as file:
                for answer in answers:
                    file.write(f'-----------------------------------------------------{count}-----------------------------------------------------------------------' + '\n')
                    file.write(answer + '\n')
    scores = [0.9 if x == 1 else 0.1 if x == 0 else 0.5 for x in preds]
    print(preds)
    print(f'ACC: {accuracy_score(labels, preds)}')
    print(f'Precision: {precision_score(labels, preds)}')
    print(f'Recall: {recall_score(labels, preds)}')
    print(f'AUC: {roc_auc_score(labels, scores)}')

mixtral_deduct('SHTech/object_data/test_50_0_vicuna-7b-v1.5_act+env.txt', 'SHTech/object_data/test_50_1_vicuna-7b-v1.5_act+env.txt', 'rule/train_100_0_vicuna-7b-v1.5_act+env.txt')

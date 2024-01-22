# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from PIL import Image
from tqdm import tqdm
import torch
from utils import *
from collections import Counter
from openai_api import llm_rule_correction
from image2text import cogvlm


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


def mixtral_induct(desc_path='SHTech/object_data/train_5_0_cogvlm.txt'):
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # model_id = "mistralai/Mistral-7B-Instruct-v0.2"
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

    objects = read_line(desc_path)
    print(objects)
    # for obj in objects:
    text = f'''As a surveillance monitor for urban safety, my job is to analyze the video footage and identify anything that could be 
               considered abnormal behavior or environment. Given the description, your task is to derive rules for anomaly detection.
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


def mixtral_verifier(cog_model, tokenizer, model, rule):
    # model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    # model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16,
    #                                              device_map='auto').eval()

    # cog_model = AutoModelForCausalLM.from_pretrained(
    #     'THUDM/cogvlm-chat-hf',
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     device_map='auto',
    #     trust_remote_code=True
    # ).eval()

    # filename = 'rule/rule_gpt4_wrong.txt'
    # if os.path.exists(filename):
    #     os.remove(filename)
    #     print(f"File {filename} has been deleted.")
    # else:
    #     print(f"The file {filename} does not exist.")
    answers = []
    preds = []
    labels = [0]*25
    count = 0
    # objects = read_line(desc_path)
    # rule = open(rule_path, "r").read()
    rule_number_used = []
    for i in range(5):
        selected_image_paths = random_select_data_without_copy(path='SHTech/train.csv', num=5, label=0)
        objects = cogvlm(model= cog_model, mode='chat', image_paths=selected_image_paths)
        wrong_answer = []
        for obj in objects:
            print(f'-----------------------------------------------------{count}-----------------------------------------------------------------------')
            print(obj)
            count += 1
            text = f'''You are monitoring the campus, you task is to detect the anomaly based on the given rules. The rules are: 
            {rule}. 
            Now you are given {[obj]}. Which rule you are using? Answer the rule number. Is it normal or anomaly? Answer Anomaly even if only one anomaly rule matches, otherwise answer Normal. Think step by step.
            Answer:'''
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=1000)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            preds.append(post_process(answer))
            answers.append(answer)
            if post_process(answer) == 1:
                print('==>Get Wrong')
                wrong_answer.append(answer)
            else:
                print('==>Count #Rule')
                rule_num = list(set(re.findall(r'\d', str(re.split(r'Answer:', answer)[1:]))))
                print(rule_num)
                rule_number_used += rule_num
                # print(re.findall(r'\b\d+\b', str(re.split(r'Rule used:|Rule:|Rule|rule', answer)[1:])))
                # with open(filename, 'a') as file:
                #     file.write(
                #         f'-----------------------------------------------------{count}-----------------------------------------------------------------------' + '\n')
                #     file.write(answer + '\n')
            print(answer)
        # if len(wrong_answer) > 0:
        #     rule = llm_rule_correction(wrong_answer)
    print(preds)
    print(f'ACC: {accuracy_score(labels, preds)}')
    print(f'Precision: {precision_score(labels, preds)}')
    print(f'Recall: {recall_score(labels, preds)}')
    print(Counter(rule_number_used))

# mixtral_verifier('rule/rule_gpt4_both.txt')
EXPRESSION_LIST = [
    "certain", "almost certain", "highly likely", "very good chance",
    "we believe", "probably", "probable", "likely", "better than even",
    "about even", "probably not", "we doubt", "unlikely", "little chance",
    "chances are slight", "improbable", "highly unlikely", "almost no chance",
    "impossible"
]

def mixtral_deduct(desc_path, rule_path, tokenizer, model, labels):
    preds = []
    probs = []
    scores = []
    saved_result = pd.DataFrame(columns=['answer', 'label', 'pred', 'probability', 'score'])
    rule = open(rule_path, "r").read()
    objects_list = read_line(desc_path)
    for index, obj in enumerate(objects_list):
        text = f'''You will be given an description of scene, you task is to detect the anomaly based on the rules. The rules are:
                    {rule}\n\n
                    First, if human activity present, which rule is matching? List the rule category, e.g., normal or anomaly, with number.\n\n
                    Second, if non-human object present, which rule is matching? List the rule category, e.g., normal or anomaly, with number.\n\n
                    Third, are the human activities or non-human objects anomaly? Answer: anomaly, if ANY anomaly rule (even if only one, no matter human activities or non-human objects) matches, otherwise answer: normal.\n\n
                    Fourth, describe how likely it is that your best answer is correct as one of the following expressions: ${EXPRESSION_LIST}. \nConfidence: <description of confidence, without any extra commentary whatsoever; just a short phrase!>\n\n
                    Now you are given the scene {obj}, think step by step.'''
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=4000)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print_out = str(obj)+find_text_after(answer, 'think step by step.')
        pred = post_process(print_out)
        prob = get_anomaly_score(print_out)
        preds.append(pred)
        probs.append(prob)
        if pred == 1:
            score = prob
        else:
            score = float("{:.2f}".format(1-prob))
        scores.append(score)
        print(f'----------------------------------------------------------------------------------------------------------------------------')
        print(pred)
        print(score)
        print(print_out)
        saved_result = saved_result._append({'answer': print_out,
                    'label': labels[index],
                    'pred': pred,
                    'probability': prob,
                    'score': score},
                     ignore_index=True)
    saved_result.to_csv(f"results/SH/{desc_path.split('/')[-1].split('.')[0]}.csv", index=False)
    print(f'Frequency of Probabilities: {Counter(probs)}')
    print(f'Frequency of Anomaly scores: {Counter(scores)}')
    print(f'ACC: {accuracy_score(labels, preds)}')
    print(f'Precision: {precision_score(labels, preds)}')
    print(f'Recall: {recall_score(labels, preds)}')
    print(f'AUC: {roc_auc_score(labels, scores)}')
    return preds, scores, probs


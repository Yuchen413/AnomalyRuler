# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from utils import *
from collections import Counter
from openai import OpenAI
from majority_smooth import cluster_keyword, anomaly_keywords


np.random.seed(2024)
torch.manual_seed(2024)

# OpenAI API Key
key = ""


device = "cuda" if torch.cuda.is_available() else "cpu"
# for i in range(torch.cuda.device_count()):
#     print(f"Device {i}: {torch.cuda.get_device_name(i)}")

EXPRESSION_LIST = [
    "certain", "almost certain", "highly likely", "very good chance",
    "we believe", "probably", "probable", "likely", "better than even",
    "about even", "probably not", "we doubt", "unlikely", "little chance",
    "chances are slight", "improbable", "highly unlikely", "almost no chance",
    "impossible"
]


def gpt_induction(objects,data_full_name):
    client = OpenAI(api_key=key)
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4" ]
    model = model_list[3]
    # objects = read_line(txt_path)
    # objects = read_txt_to_list(txt_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": f'''As a surveillance monitor for urban safety using the {data_full_name} dataset, my job is derive rules for detect abnormal human activities or environmental object.'''},
            {"role": "user", "content": "Based on the assumption that the given frame description are normal, "
                                        "Please derive rules for normal, start from an abstract concept and then generalize to concrete activities (e.g., walking) or objects."},
            {"role": "assistant", "content": '''
                                        **Rules for Normal Human Activities:
                                        1. 
                                        **Rules for Normal Environmental Objects:
                                        1. 
                                        '''},
            {"role": "user",
             "content": "Compared with the above rules for normal, can you provide potential rules for anomaly? Please start from an abstract concept then generalize to concrete activities (e.g., non-walking such as riding a bicycle, scooting, skateboarding.) or objects, compared with normal ones."},
            {"role": "assistant", "content": '''
                                        **Rules for Anomaly Human Activities:
                                        1. 
                                        **Rules for Anomaly Environmental Objects:
                                        1. 
                                        '''},

            {"role": "user",
             "content": f"Now you are given {objects}. What are the Normal and Anomaly rules you got? Think step by step. Reply following the above format, start from an abstract concept and then generalize to concrete activities or objects. List them using short terms, not an entire sentence."},
        ]
    )
    print('=====> Rule Generation:')
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def gpt_rule_correction(objects, n, data_full_name):
    client = OpenAI(api_key=key)
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4" ]
    model = model_list[3]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": f'''As a surveillance monitor for urban safety using the {data_full_name} dataset, my job is to organize rules for detect abnormal activities and objeacts.'''},
            {"role": "user", "content": f"You are given {n} independent sets of rules for Normal and Anomaly. "
                                        f"For the organized normal Rules, list the given normal rules with high-frenquency elements"
                                        f"For the organized anomaly Rules, list all the given anomaly rules"},
            {"role": "assistant", "content": '''
                                                **Rules for Anomaly Human Activities:
                                                1. Non-walking movement: riding a bicycle, scooting, skateboarding.
                                                2.
                                                **Rules for Anomaly Environmental Objects:
                                                1. Ground transportation: vehicles, motorcycles.
                                                2.
                                                **Rules for Normal Human Activities:
                                                1. Walking with common objects: a backpack, bag, umbrella.
                                                2.
                                                **Rules for Normal Environmental Objects:
                                                1. Architectural structures: building, bridges.
                                                2.
                                                '''},
            {"role": "user",
             "content": f"Now you are given {n} independent sets of rules as the sublists of {objects}. What rules for Anomaly and Normal do you get? Think step by step, reply following the above examples."},
        ]
    )
    # ', start from an abstract concept and then generalize to concrete activities or objects.'
    print('=====> Organized Rules:')
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def mixtral_deduct(data, desc_path, rule_path, tokenizer, model, labels):
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
    saved_result.to_csv(f"results/{data}/{desc_path.split('/')[-1].split('.')[0]}.csv", index=False)
    print(f'Frequency of Probabilities: {Counter(probs)}')
    print(f'Frequency of Anomaly scores: {Counter(scores)}')
    print(f'ACC: {accuracy_score(labels, preds)}')
    print(f'Precision: {precision_score(labels, preds)}')
    print(f'Recall: {recall_score(labels, preds)}')
    try:
        print(f'AUC: {roc_auc_score(labels, scores)}')
    except Exception as e:
        print(f"An error occurred: {e}. Setting AUC to 0.0.")
    return preds, scores, probs


def mixtral_double_deduct(data, desc_path, rule_path, tokenizer, model, labels):
    preds = []
    labels = labels
    saved_result = pd.DataFrame(columns=['answer', 'label', 'pred', 'dummy_pred'])
    rule = open(rule_path, "r").read()
    objects_list = read_line(desc_path)
    if os.path.exists('rule/rule_SHTech.npy'):
        # This is generated and saved from the Step3
        anomaly_from_rule = np.load('rule/rule_SHTech.npy', allow_pickle=True)
    else:
        anomaly_from_rule = anomaly_keywords()
    for index, obj in enumerate(objects_list):
        ini_pred, _, anomaly_keyword = cluster_keyword(obj, anomaly_from_rule)
        if ini_pred[0] == 1:
            ini_answer = f'Anomaly, since I found {anomaly_keyword[0]}.'
            text = f'''You will be given an description of scene, you task is to double check my initial anomaly detection result based on the rules. The rules are:
                        {rule}\n\n
                        My initial result is {ini_answer}\n
                        First, if human activity present, which rule is matching? List the rule category, e.g., normal or anomaly, with number.\n\n
                        Second, if environmental object present, which rule is matching? List the rule category, e.g., normal or anomaly, with number.\n\n
                        Third, are the human activities or environmental objects anomaly? Answer: anomaly, if you also find {anomaly_keyword[0]} or ANY anomaly rule (even if only one, no matter human activities or environmental objects) matches, otherwise answer: normal.\n\n
                        Now you are given the scene {obj}, think step by step.'''

        else:
            ini_answer = f"Normal."
            text = f'''You will be given an description of scene, you task is to double check my initial anomaly detection result based on the rules. The rules are:
                        {rule}\n\n
                        My initial result is {ini_answer}\n
                        First, if human activity present, which rule is matching? List the rule category, e.g., normal or anomaly, with number.\n\n
                        Second, if environmental object present, which rule is matching? List the rule category, e.g., normal or anomaly, with number.\n\n
                        Third, are the human activities or environmental objects anomaly? Answer: anomaly, if ANY anomaly rule (even if only one, no matter human activities or environmental objects) matches, otherwise answer: normal.\n\n
                        Now you are given the scene {obj}, think step by step.'''

        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=4000)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print_out = str(obj)+find_text_after(answer, 'think step by step.')
        pred = 1 if ini_pred[0] == 1 else post_process(print_out)
        preds.append(pred)
        print(f'----------------------------------------------------------------------------------------------------------------------------')
        print(pred)
        print(print_out)
        saved_result = saved_result._append({'answer': print_out,
                    'label': labels[index],
                    'pred': pred,
                    'dummy_pred': ini_pred[0]},
                     ignore_index=True)
    saved_result.to_csv(f"results/{data}/{desc_path.split('/')[-1].split('.')[0]}.csv", index=False)
    print(f'ACC: {accuracy_score(labels, preds)}')
    print(f'Precision: {precision_score(labels, preds)}')
    print(f'Recall: {recall_score(labels, preds)}')
    return preds


def gpt_double_deduction_demo(data, desc_path, rule_path):
    '''
    This is just an example for using GPT in deduction, have not tested on large scale dataset
    '''
    client = OpenAI(api_key=key)
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4" ]
    model = model_list[3]
    rule = open(rule_path, "r").read()
    objects_list = read_line(desc_path)
    anomaly_from_rule = anomaly_keywords()
    for index, obj in enumerate(objects_list):
        ini_pred, _, anomaly_keyword = cluster_keyword(obj, anomaly_from_rule)
        if ini_pred[0] == 1:
            ini_answer = f'Anomaly, since I found {anomaly_keyword[0]}.'
            text = f'''You will be given an description of scene, you task is to double check my initial anomaly detection result based on the rules. The rules are:
                        {rule}\n\n
                        My initial result is {ini_answer}\n
                        First, if human activity present, which rule is matching? List the rule category, e.g., normal or anomaly, with number.\n\n
                        Second, if environmental object present, which rule is matching? List the rule category, e.g., normal or anomaly, with number.\n\n
                        Third, are the human activities or environmental objects anomaly? Answer: anomaly, if you also find {anomaly_keyword[0]} or ANY anomaly rule (even if only one, no matter human activities or environmental objects) matches, otherwise answer: normal.\n\n
                        Now you are given the scene {obj}, think step by step.'''

        else:
            ini_answer = f"Normal."
            text = f'''You will be given an description of scene, you task is to double check my initial anomaly detection result based on the rules. The rules are:
                        {rule}\n\n
                        My initial result is {ini_answer}\n
                        First, if human activity present, which rule is matching? List the rule category, e.g., normal or anomaly, with number.\n\n
                        Second, if environmental object present, which rule is matching? List the rule category, e.g., normal or anomaly, with number.\n\n
                        Third, are the human activities or environmental objects anomaly? Answer: anomaly, if ANY anomaly rule (even if only one, no matter human activities or environmental objects) matches, otherwise answer: normal.\n\n
                        Now you are given the scene {obj}, think step by step.'''
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": f'''As a surveillance monitor for urban safety using the {data} dataset, my job to detect abnormal human activities or environmental object based on the provided rules'''},
                {"role": "user",
                 "content": f"{text}"},
            ]
        )
        print_out = response.choices[0].message.content
        print('=====> Results:')
        print(print_out)

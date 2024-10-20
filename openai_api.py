from openai import OpenAI
from utils import *
import base64
import re
import json
import inflect
p = inflect.engine()

# OpenAI API Key
key = ""


def extract_words_from_normal(rules_text, start_marker, end_marker="**Rules for"):
    start_index = rules_text.find(start_marker)
    if start_index != -1:
        start_index += len(start_marker)
        end_index = rules_text.find(end_marker, start_index)
        if end_index == -1:
            end_index = len(rules_text)  # If no end_marker is found, go to the end of the text
        extracted_text = rules_text[start_index:end_index].strip()
    else:
        extracted_text = ""
    words_in_text = re.findall(r'\b\w+\b', extracted_text.lower())
    return words_in_text



def keyword_extract(rule_path):
    client = OpenAI(api_key=key)
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4"]
    model = model_list[3]
    with open(rule_path, 'r', encoding='utf-8') as file:
        rules = file.read()
    # rules = read_txt_to_one_list(rule_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": f'''You will be given a set of rules for detecting abnormal activities; Please:
             1. Extract the "ing" verbs.
             2. Remove verbs that do not represent a specific physical activity or motion, such as 'engaging', 'moving', 'working'...
             3. Remove verbs that are listed as Normal Human Activities.
             4. Add the synonyms for the left "ing" verbs.
             The output should be in the format: [anomaly activity1, anomaly activity2, ...].
             Provide a combined Python list with each represented by a single word.'''},
            {"role": "user",
             "content": f'''Now you are given {rules}, please ONLY output the Python list without ANY additional descriptions. Think step by step.'''},
        ]
    )

    words_in_text = extract_words_from_normal(rules, "**Rules for Normal Human Activities:**") + extract_words_from_normal(rules, "**Rules for Normal Environmental Objects:**")
    raw_response = response.choices[0].message.content
    print(
        '==> Info: If you encounter anomaly results, such as 0 precision or recall, please check the raw_response from GPT within function keyword_extract(). In case the output is not formatted as a Python list, you may need to manually select the keywords and format them as a list.')
    # print(raw_response)
    all_lists = re.findall(r"\[.*?\]", raw_response, re.DOTALL)
    if all_lists:
        # Extract the last list from the matches
        last_list_string = all_lists[-1]  # Get the last match
        try:
            # Replace single quotes with double quotes if necessary
            python_list = json.loads(last_list_string.replace("'", '"'))  # Convert to Python list
        except json.JSONDecodeError:
            python_list = []
            print("Error: The extracted list is not valid JSON.")
    else:
        print("No list found in the raw response.")
        python_list = []
    customized_anomaly = ["bicycle"]  # This list is for manually adding specific anomaly keywords that you consider important. It helps make the results more stable and controllable by ensuring certain critical keywords are always included in the output.
    words_in_text = [item.lower() for item in words_in_text]
    python_list = [item.lower() for item in python_list]
    post_response = [item for item in python_list if item not in words_in_text]
    post_response = [p.singular_noun(i) or i for i in post_response] + customized_anomaly
    return post_response


def baseline():
    client = OpenAI(api_key=key)
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo"]
    model = model_list[0]
    txt_path = 'SHTech/object_data/test_50_1_vicuna-7b-v1.5_baseline.txt'
    objects = read_txt_to_list(txt_path)
    results = []
    for obj in objects:
        prompt = (
            "As a surveillance monitor for urban safety using the ShanghaiTech dataset, my job is to analyze the video footage and identify "
            "anything that could be considered abnormal behavior or activity, which might indicate a potential safety issue or incident. "
            "For each individual or object that the camera observes, I will reason about whether it constitutes a normal observation or an anomaly. "
            "Normal observations do not suggest any potential safety threats, whereas anomalies might. Finally, I will classify whether the overall "
            "scene is normal or abnormal. For example,\n"
            "\"The surveillance camera is monitoring an area of the campus and observes:\n"
            "[walking, sitting, running fast]\n"
            "walking:\n"
            "1. Is this common to see in this environment?\n"
            "Yes, walking is the most common mode of transportation around the campus.\n"
            "2. Can this influence the environment’s safety? Generally, no, walking does not pose a safety risk if individuals are following the "
            "campus pathways and rules.\n"
            "3. Is this behavior safe under normal circumstances? Yes, walking is safe and expected behavior on campus.\n\n"
            "Classification: Normal.\n"
            "sitting:\n"
            "1. Is this common to see in this environment?\n"
            "Yes, it is normal to see individuals sitting on campus, whether for resting, socializing, or studying.\n"
            "2. Can this influence the environment’s safety? No, sitting is a benign activity and is expected in areas like benches, lawns, or steps.\n"
            "3. Is this behavior safe under normal circumstances? Yes, sitting is considered a normal and safe activity.\n\n"
            "Classification: Normal.\n"
            "running fast:\n"
            "1. Is this common to see in this environment?\n"
            "Running fast is less common and may depend on context, such as being late for class or exercising.\n"
            "2. Can this influence the environment’s safety? It could, especially if the individual is running in a crowded area or if their "
            "behavior suggests panic or emergency.\n"
            "3. Is this behavior safe under normal circumstances? Running fast could be unsafe if it leads to collisions with others or if it "
            "induces panic.\n"
            "Classification: Anomaly. \n"
            "Overall Scenario Classification: Anomaly. \"\n\n"
            f"\"I am monitoring the campus and I see:{obj}\n"
        )

        response = client.completions.create(
            model=model,
            prompt = prompt,
            max_tokens = 1000
        )

        print(response.choices[0].text)
        results.append(response.choices[0].text)

    filename = f"results/{model}_{txt_path.split('/')[-1]}"
    count_0 = 0
    count_1 = 0
    with open(filename, 'w') as file:
        for string in results:
            file.write(string + '\n')
            lines = string.split('\n')
            last_line = lines[-1]
            if 'Normal' in last_line:
                count_0 += 1
            elif 'Anomaly' in last_line:
                count_1 +=1

    print(count_0)
    print(count_1)
    if filename.split('_')[-2] == '0':
        print(f'Acc:{count_0/len(results)}')
    elif filename.split('_')[-2] == '1':
        print(f'Acc:{count_1/len(results)}')

    # with open(filename, 'r') as file:
    #     # Read the entire file content
    #     file_content = file.read()
    # original_list = file_content.strip().split('\n\n')

# baseline()

def llm_deduction(txt_path, rule, rule_name):
    client = OpenAI(api_key=key)
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4"]
    model = model_list[2]
    objects = read_line(txt_path)
    count = 0
    filename = f"results/{rule_name}_{model}_{txt_path.split('/')[-1]}"
    if os.path.exists(filename):
        os.remove(filename)
        print(f"File {filename} has been deleted.")
    else:
        print(f"The file {filename} does not exist.")

    for obj in objects:
        count += 1
        print(f'================================{count}=============================')
        with open(filename, 'a') as file:
            file.write(f'============================{count}======================' + '\n')
        for item in obj:
            print(item)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"You are monitoring the campus, you task is to detect the anomaly based on the given rules. The rules are: {rule}"},
                    {"role": "user", "content": "What are in the test_frame_description_local, as detail as possible? Reply following the template."},
                    {"role": "assistant", "content": '''Image Summary:
                                                        Object: People, a group of people, skateboarder
                                                        Action: Walking, standing, riding'''},
                    {"role": "user", "content": "Is it normal or anomaly? Reply following the template."},
                    {"role": "assistant", "content": '''Anomaly Detection:
                                                        [A group pf people, walking] = Normal based on rule 3, because walking is a common activity in a park or on a walkway in an urban area.
                                                        [People, standing] = Normal based one rule 1,  because it is common for people to stop and stand in urban areas, possibly to rest or wait.
                                                        [skateboarder, riding] = Anomaly, because it is unusually to see a skateboarder to be riding on a road.
                                                        
                                                        Overall, its Anomaly. Since there is anomaly above. '''},
                    {"role": "user", "content": f"Now you are given {item}. What is in the test_frame_description_local? Reply following the template. Is it normal or anomaly? Reply following the template."},
                ]
            )
            print(response.choices[0].message.content)
            print('--------------------------------------')
            with open(filename, 'a') as file:
                file.write(f'----------------------{count}-----------------------' + '\n')
                file.write(response.choices[0].message.content + '\n')

def gpt4v_induction():
    import base64
    import requests

    # OpenAI API Key
    api_key = key

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    image_paths = get_all_paths("SHTech/train_10_0")
    base64_images = [encode_image(i) for i in image_paths]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {"role": "system",
             "content": f'''As a surveillance monitor for urban safety using the ShanghaiTech dataset, my job is derive rules for detect abnormal behavior or activity from the given images.'''},
            {"role": "user", "content": "What in this image?"
                                        "Please reply question for each image"},
            {"role": "assistant", "content": '''Image 1:Individuals walking calmly on the sidewalk.
                                                Image 2:
                                                 '''},
            {"role": "user", "content": "All the images are labeled as Normal, what are the Rules for Normal?"},
            {"role": "assistant", "content": '''Rules for Normal:
                                                1. Individuals walking calmly on the sidewalk is normal, because .
                                                2.
                                         '''},
            {"role": "user", "content": "Now based on the Normal rules you have, what are the Rules for Anomaly? Please consider the objects or activities that leads to safety risk."},
            {"role": "assistant", "content": ''' Rules for Anomaly:
                                        1.
                                        2.
                                 '''},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Now you are given 10 images, for each image, please answer: What is in the image? What are Rules for Normal? What are Rules for Anomaly? Please think step by step"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images[0]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images[1]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images[2]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images[3]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images[4]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images[5]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images[6]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images[7]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images[8]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images[9]}"
                        },
                    }
                ]
            }
        ],
        "max_tokens": 5000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response)
    print(response.json()['choices'][0]['message']['content'])
def gpt_text2object():
    client = OpenAI(api_key=key)
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo"]
    model = model_list[1]
    txt_path = 'SHTech/object_data/test_50_1_instructblip.txt'

    with open(txt_path, 'r') as file:
        file_content = file.read()
    objects = file_content.strip().split('\n')
    results = []
    for obj in objects:
        prompt = (
            "Please summarize the given sentence into one list with several words, for example, [person, walking, street, cycling].\n"
            f"Now you are given{obj}"
        )
        response = client.completions.create(
            model=model,
            prompt = prompt,
            max_tokens = 100
        )

        print(response.choices[0].text)
        results.append(','.join(re.findall(r'\b\w+\b', response.choices[0].text)))
    with open(f'{txt_path.split(".")[0]}object.txt', 'w') as file:
        for inner_list in results:
            file.write(inner_list + '\n')

def gpt4v_deduction(rule_name, prompt, image_root = "SHTech/test_50_0"):
    import base64
    import requests

    # OpenAI API Key
    model = "gpt-4-vision-preview"
    api_key = key

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    image_paths = sorted(get_all_paths(image_root))
    base64_images = [encode_image(i) for i in image_paths][:20]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = prompt
    results = []
    count_0 = 0
    count_1 = 0
    for i in range(len(base64_images)):
        print(f"=============={i}===============")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_images[i]}"
                            },

                        }
                    ]
                }
            ],
            "max_tokens": 800
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(response.json())
        print(response.json()['choices'][0]['message']['content'])
        filename = f"results/{rule_name}_{model}_{image_root.split('/')[-1]}"
        with open(filename, 'a') as file:
            file.write(f'============================{i}======================'+'\n')
            file.write(response.json()['choices'][0]['message']['content'] + '\n')
    #
    # if image_root.split('_')[-1] == '0':
    #     print(count_0)
    #     print(f'Acc:{count_0 / len(results)}')
    # elif image_root.split('_')[-1] == '1':
    #     print(count_1)
    #     print(f'Acc:{count_1 / len(results)}')

def llm_induction(objects):
    client = OpenAI(api_key=key)
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4-1106-preview" ]
    model = model_list[3]
    # objects = read_line(txt_path)
    # objects = read_txt_to_list(txt_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": f'''As a surveillance monitor for urban safety using the ShanghaiTech dataset, my job is derive rules for detect abnormal activities or object.'''},
            {"role": "user", "content": "Based on the assumption that the given test_frame_description_local are normal, "
                                        "Please derive rules for normal considering the specific and concrete activities or objects."},
            {"role": "assistant", "content": '''
                                        **Rules for Normal Human Activities:
                                        1.
                                        **Rules for Normal Non-Human Objects:
                                        1.
                                        '''},
            {"role": "user",
             "content": "Compared with the above normal rules, can you provide potential anomaly rules? Please consider the specific and concrete activities or objects compared with normal ones."},
            {"role": "assistant", "content": '''**Rules for Anomaly Human Activities:
                                        1. 
                                        **Rules for Anomaly Non-Human Objects:
                                        1.
                                        '''},

            {"role": "user",
             "content": f"Now you are given {objects}. What are the Normal and Anomaly rules you got? Think step by step. Reply following the above format, aim for concrete activities/objects rather than being too abstract. List them using short terms, not an entire sentence. "},
        ]
    )
    print('=====> Rule Generation:')
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def llm_rule_correction(objects, n, data_full_name):
    client = OpenAI(api_key=key)
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4-1106-preview" ]
    model = model_list[3]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": f'''As a surveillance monitor for urban safety using the {data_full_name} dataset, my job is to organize rules for detect abnormal activities and objeacts.'''},
            {"role": "user", "content": f"You are given {n} independent sets of rules for Normal and Anomaly. "
                                        f"For the organized normal Rules, list the given normal rules with high-frenquency elements"
                                        f"For the organized anomaly Rules, list all the given anomaly rules"},
            {"role": "assistant", "content": '''**Rules for Normal Human Activities:
                                                1. 
                                                **Rules for Normal Non-Human Objects:
                                                1.
                                                **Rules for Anomaly Human Activities:
                                                1. 
                                                **Rules for Anomaly Non-Human Objects:
                                                1.
                                                '''},
            {"role": "user",
             "content": f"Now you are given {n} independent sets of rules as the sublists of {objects}. What are the Normal and Anomaly rules do you get? Think step by step, reply following the above format, aim for concrete activities/objects rather than being too abstract."},
        ]
    )
    print('=====> Organized Rules:')
    print(response.choices[0].message.content)
    return response.choices[0].message.content


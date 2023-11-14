from openai import OpenAI
from utils import *
import base64
import re



def baseline():
    client = OpenAI(api_key="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")
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo"]
    model = model_list[0]
    txt_path = 'SHTech/object_data/test_50_0_owlvit.txt'
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



def llm_deduction(txt_path, rule):
    client = OpenAI(api_key="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")
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo"]
    model = model_list[1]
    objects = read_txt_to_list(txt_path)
    print(objects)
    results = []
    count_0 = 0
    count_1 = 0
    count = 0

    for obj in objects:
        count+=1
        print(f'================================{count}=============================')
        prompt = rule + '\n' + f'''Now I am monitoring the campus and I'm given:{obj}, 
        First, I will summarize the above description into [object, action, environment],
        Second, reply [object, action, environment] = Normal, or Anomaly because, 
        Finally, summarize the above and reply 'Overall, it is Normal, or Anomaly'
        '''

        response = client.completions.create(
            model=model,
            prompt = prompt,
            max_tokens = 500
        )
        print(response.choices[0].text)

        if 'Normal' in response.choices[0].text:
            count_0 += 1
        elif 'Anomaly' in response.choices[0].text:
            count_1 += 1
        results.append(response.choices[0].text)

        filename = f"results/rule_v4_{model}_{txt_path.split('/')[-1]}"
        with open(filename, 'w') as file:
            for string in results:
                file.write(str(count) + '\n')
                file.write(string + '\n')


    if txt_path.split('_')[-2] == '0':
        print(count_0)
        print(f'Acc:{count_0 / len(results)}')
    elif txt_path.split('_')[-2] == '1':
        print(count_1)
        print(f'Acc:{count_1 / len(results)}')





def gpt4v_induction():
    import base64
    import requests

    # OpenAI API Key
    api_key = "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"

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    image_paths = get_all_paths("SHTech/train_5_0")
    base64_images = [encode_image(i) for i in image_paths]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Your task is to derive rules for anomaly detection, you will be given a bunch of images labeled as 'normal'"
                                "First describe each from perspective of [object, action, environment], as detailed as possible, and try use words or short terms "
                                "Second, derive rules used for normal from your description you got for the first step, using the template '[object, action, environment] = normal, because', use short terms."
                                "Third, derive rules used for anomaly, from the normal rules you got, consider what objects is potential anormaly, and what action is anormaly, using the template '[object, action, environment] = anomaly, because', use short terms."
                                "Reply following the format:"
                                "Rules for normal:"
                                "Rules for anomaly:"
                                "Summary of rules:"
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
                    }
                ]
            }
        ],
        "max_tokens": 500
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.json()['choices'][0]['message']['content'])
def gpt_text2object():
    client = OpenAI(api_key="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")
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
    api_key = "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"

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    image_paths = sorted(get_all_paths(image_root))
    base64_images = [encode_image(i) for i in image_paths][10:20]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = prompt
    results = []
    count_0 = 0
    count_1 = 0
    # for i in range(0,5):
    #     print(f"=============={i}===============")
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

                    },

                ]
            }
        ],
        "max_tokens": 500
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.json())
    print(response.json()['choices'][0]['message']['content'])

    if 'Anomaly' in response.json()['choices'][0]['message']['content']:
        count_1 += 1
    else:
        count_0 += 1

    results.append(response.json()['choices'][0]['message']['content'])

    filename = f"results/{rule_name}_{model}_{image_root.split('/')[-1]}"
    with open(filename, 'w') as file:
        for string in results:
            # file.write(str(i) + '\n')
            file.write(string + '\n')

    if image_root.split('_')[-1] == '0':
        print(count_0)
        print(f'Acc:{count_0 / len(results)}')
    elif image_root.split('_')[-1] == '1':
        print(count_1)
        print(f'Acc:{count_1 / len(results)}')

def llm_induction_1(txt_path, prompt):
    client = OpenAI(api_key="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")
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo"]
    model = model_list[1]
    objects = read_txt_to_list(txt_path)

    prompt = prompt + '\n' + f'''Now I am monitoring the campus and I'm given a description:{objects}, '''

    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=500
    )
    print(response.choices[0].text)
    return response.choices[0].text


def llm_induction_2(rule_stage_1):
    client = OpenAI(api_key="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")
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo"]
    model = model_list[1]
    prompt = f'''As a surveillance monitor for urban safety using the ShanghaiTech dataset, my job is detect abnormal behavior or activity. 
                 I will be given a bunch of rules indicate normal.
                 
                 Given {rule_stage_1}, derive anomaly rules, by considering what objects is potential anomaly, 
                 and what action is anomaly
                 
                 Reply using the template:
                 
                 **Rules for Anomaly:
                 1. [object, action, environment] = Anomaly, because
                 2.
                 
                 '''

    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=500
    )
    print(response.choices[0].text)
    return response.choices[0].text




# baseline()
# baseline_with_rule()
# gpt4v()
# gpt_text2object()

base = '''As a surveillance monitor for urban safety using the ShanghaiTech dataset, 
                    my job is to analyze the video footage and identify anything that could be 
                    considered abnormal behavior or activity, which might indicate a potential 
                    safety issue or incident. For each individual or object that the camera observes, 
                    I will reason about whether it constitutes a normal observation or an anomaly. 
                    Normal observations do not suggest any potential safety threats, whereas anomalies
                    might. Finally, I will classify whether the overall scene is normal or abnormal. 
                    
                    For example，
                    "The surveillance camera is monitoring an area of the campus and observes:
                    - Walking
                    - Sitting
                    - Running fast
                    
                    Walking:
                    1. Is this common to see in this environment?
                    Yes, walking is the most common mode of transportation around the campus.
                    2. Can this influence the environment’s safety? Generally, no, walking does not pose a safety risk if individuals are following the campus pathways and rules.
                    3. Is this behavior safe under normal circumstances? Yes, walking is safe and expected behavior on campus.
                    Classification: Normal.
                    
                    Sitting:
                    1. Is this common to see in this environment?
                    Yes, it is normal to see individuals sitting on campus, whether for resting, socializing, or studying.
                    2. Can this influence the environment’s safety? No, sitting is a benign activity and is expected in areas like benches, lawns, or steps.
                    3. Is this behavior safe under normal circumstances? Yes, sitting is considered a normal and safe activity.
                    Classification: Normal.
                    
                    Running fast:
                    1. Is this common to see in this environment?
                    Running fast is less common and may depend on context, such as being late for class or exercising.
                    2. Can this influence the environment’s safety? It could, especially if the individual is running in a crowded area or if their behavior suggests panic or emergency.
                    3. Is this behavior safe under normal circumstances? Running fast could be unsafe if it leads to collisions with others or if it induces panic.
                    Classification: Anomaly.
                    
                    Overall Scenario Classification: Anomaly
                    "
                    
                    I am monitoring the campus and I see:'''

rule_v3 = """
As a surveillance monitor for urban safety using the ShanghaiTech dataset, my job is to analyze the video footage and identify anything that could be considered abnormal behavior or activity. 
I will first describe the image as detailed as possible, including all the objects, their actions, and the environment in the image
I will then be given a rule that indicates normal or anomaly based on my description, for example:

Rules for normal:
1. [Person, walking, path] = normal, because walking on paths is expected.
2. [Persons, walking, beside road] = normal, because walking beside a road on the sidewalk is common behavior.
3. [Persons, walking, open area] = normal, because people walking through open areas is typical.
4. [Person, standing, open area] = normal, because an individual may stand anywhere in an open pedestrian area.
5. [Multiple Persons, walking, road] = normal, because walking cautiously on roads with less traffic is common practice.     

Rules for anomaly:
1. [Person, lying down, paved path] = anomaly, because lying down on paths disrupts flow of traffic.
2. [Person, cycling, path] = anomaly, because sidewalks are typically for pedestrian use not bicycles.
3. [Vehicles, driving, open area] = anomaly, because vehicles are not typically allowed in areas designated for pedestrians.
4. [Person, scooting, paved path] = anomaly, because scooting may not be allowed on certain paths.
5. [Person, non-walking, road] = anomaly, because walkways are meant for walking, but not other unusual actions

Now I am monitoring the campus and I'm given a {image}, following the rules, reply {image description} = Normal or {image description} = Anomaly, because
"""

rule_v4 = f"""
As a surveillance monitor for urban safety using the ShanghaiTech dataset, my job is to analyze the video footage and identify anything that could be considered abnormal behavior or activity. I will be given a rule that indicates normal or anomaly, for example:

**Rules for Normal:
1. [people, walking, street] = Normal, because it is common for people to walk down the street in a group.
2. [backpack, carrying, group] = Normal, because individuals often carry bags or backpacks while walking in larger groups.
3. [people, walking, sidewalk] = Normal, because walking on a sidewalk is a common activity that people engage in.
4. [people, walking, street] = Normal, because it is common to see people walking down a pedestrian-friendly street.
5. [pedestrian walkway, walking, outdoors] = Normal, because it is expected for people to walk on pedestrian walkways and enjoy the outdoors.


**Rules for Anomaly:
1. [person, crawling, public space] = Anomaly, because crawling is an unusual form of movement in a public space.
2. [backpack, dragging, non-pedestrian area] = Anomaly, because dragging a backpack in a non-pedestrian area may indicate a potential threat or suspicious behavior.
3. [person, sprinting, crowded area] = Anomaly, because sprinting in a crowded area can cause disturbance or be indicative of a potential threat.
4. [vehicle, driving, pedestrian walkway] = Anomaly, because driving a vehicle on a pedestrian walkway can cause harm to pedestrians and is not a normal use of the space.
5. [person, climbing, non-climbable structure] = Anomaly, because climbing a non-climbable structure is not a common behavior and may indicate malicious intent or danger.
"""

induce_rule = """
        As a surveillance monitor for urban safety using the ShanghaiTech dataset, my job is detect abnormal behavior or activity. 
        I will be given a description of scene, and I will induce the rule for future anomaly detection. 

        First, I will summarize the description into [object, action, environment]
        
        Second, for each description, I will derive rules based on my knowledge and reply following the template:
        
        **Rules for Normal:
        1. [object, action, environment] = Normal, because
        2. 
        """

gpt4v_deduction(rule_name='rule_v3',prompt=rule_v3, image_root="SHTech/test_50_1")
# rule_stage_1 = llm_induction_1('SHTech/train_5_0_instructblip.txt', prompt=induce_rule)
# rule_stage_2 = llm_induction_2(rule_stage_1)



# llm_deduction('SHTech/test_50_1_instructblip.txt', rule=rule_v4)

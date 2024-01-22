from openai import OpenAI
from utils import *
import base64
import re

def baseline():
    client = OpenAI(api_key="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")
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
    client = OpenAI(api_key="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")
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
    api_key = "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"

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

def llm_induction_1(objects):
    client = OpenAI(api_key="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")
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
             "content": f"Now you are given {objects}. What are the Normal and Anomaly rules you got? Think step by step. Reply following the above format, aim for concrete activities/objects rather than being too abstract. List them using short terms, not an entire sentence."},
        ]
    )
    print('=====> Rule Generation:')
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def llm_rule_correction(objects, n):
    client = OpenAI(api_key="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")
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4-1106-preview" ]
    model = model_list[3]
    # objects = read_line(txt_path)
    # objects = read_txt_to_list(txt_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": f'''As a surveillance monitor for urban safety using the ShanghaiTech dataset, my job is to organize rules for detect abnormal activities and objeacts.'''},
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


def llm_induction_2(rule_stage_1):
    client = OpenAI(api_key="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")
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4-1106-preview"]
    model = model_list[3]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": f'''I will be given a bunch of normal rules, I need to derive anomaly rules based on them, by considering what behavior or objects is potential anomaly'''},
            {"role": "user", "content": f"Given normal rules {rule_stage_1}, what are the anomaly rules? Reply following the template and use terms as short as possible"},
            # {"role": "assistant", "content": '''**Rules for Anomaly:
            #                                     1.[Any object, action, environment you think might cause safety issue]= Anomaly, because
            #                                     2.
            #                                     '''},
        ]
    )
    print(response.choices[0].message.content)

    # response = client.completions.create(
    #     model=model,
    #     prompt=prompt,
    #     max_tokens=500
    # )
    # print(response.choices[0].text)
    # return response.choices[0].text


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

rule_v3_enhanced = """
To perform the task of anomaly detection, I will first describe the contents of the image based on the [object, action, environment] framework:

Object: I'll identify the main subjects or objects in the image.
Action: I'll describe what these subjects or objects are doing.
Environment: I'll describe the setting or location where the action is taking place.

I will follow the rules below:

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
5. [Person, running, road] = anomaly, because running may be dangerous

For each image, my reply will be like the following example:

Image Summary:
Object: People
Action: Walking, standing
Environment: Park, walkway, urban area

Anomaly Detection:
[People, walking, park/walkway] = Normal based on rule x, because walking is a common activity in a park or on a walkway in an urban area.
[People, standing, urban area] = Normal based one rule y,  because it is common for people to stop and stand in urban areas, possibly to rest or wait.
"""
rule_v5_enhanced = '''
**Rules for Normal:
1. [people, walking/standing, street] = Normal
2. [people, walking, sidewalk] = Normal
3. [a group of people, walking] = Normal
4. [person, walking, carrying object] = Normal
5. [person, sitting, on bench] = Normal
6. [people, walking, in park] = Normal
7. [person, talking, on cell phone] = Normal 
8. [person, walking, near building] = Normal
9. No human-related actions or behaviors =  Normal


**Rules for Anomaly:
1. [person, running, street/sidewalk] = Anomaly, because it is unusual for a person to be moving at a faster pace on a regular street/sidewalk.
2. [skateboarder, riding, road] = Anomaly, because it is unusual for a skateboarder to be riding on a road with pedestrians.
3. [bike/motorcycle, riding, sidewalk] = Anomaly, because it is unusual for a non-pedestrian vehicle to be riding.
4. [person, laying down, sidewalk] = Anomaly, because it is uncommon for a person to be laying down on a sidewalk.
5. [Vehicles, driving, open area] = Anomaly, because vehicles are not typically allowed in areas designated for pedestrians.
6. [people, fighting/pushing, anywhere] = Anomaly, because violent activity are not permitted on campus.
7. [person, riding, street/sidewalk] = Anomaly, because it is unusual for a non-pedestrian vehicle to be riding.
8. [person, jumping] = Anomaly, because it is unusual for a person to be jumping on street.
'''
rule_v5_cogvlm_normal = '''
**Rules for Normal:** 
1. [people, walking, urban area] = Normal, because people walking in an urban area is a common and expected behavior.
2. [people, standing, urban area] = Normal, as people standing in an urban area can be attributed to waiting or observing their surroundings.
3. [people, walking with bags, urban area] = Normal, since people carrying bags while walking in an urban area is a common occurrence.
4. [people, walking together, urban area] = Normal, because people walking together in an urban area, especially engaged in conversation, is a typical social behavior.
5. No human-related actions or behaviors =  Normal
**Rules for Anomaly:**
Any object, activities that are not covered by Normal = Anomaly

'''
rule_v6_with_wrong = '''
**Normal Rules**
1. [people, walking, sidewalk] = Normal, because it is a common activity in urban areas.
2. [people, sitting/standing, open area] = Normal, as it is a common activity in open area
3. [woman, walking, sidewalk, carrying handbag] = Normal, as it represents a regular pedestrian activity.
4. [group of people, walking, sidewalk, bike path] = Normal, as it signifies a group activity in a common urban area.
5. [group of people, skateboarding, sidewalk] = Anomaly, because skateboarding on the sidewalk is not a common or safe activity.
6. No human-related activities = Normal
7. Any other object, action, environment that is safe = Normal


**Potentially Wrong Rules:**
Rule 5 - [group of people, skateboarding, sidewalk] = Anomaly

**Cause for the Potentially Wrong Rule:**
The potentially wrong rule could be due to misclassifying the activity of skateboarding on the sidewalk as an anomaly instead of recognizing it as a normal activity. It might be possible that the surveillance monitor has misinterpreted the behavior as unusual or potentially unsafe. Alternatively, if the surveillance monitor has been specifically trained to consider skateboarding on the sidewalk as an anomaly, then the rule may not be wrong but rather intentional based on pre-defined criteria.

**Anormaly Rules**
1. [group of people, skateboarding, sidewalk] = Anomaly, because skateboarding on the sidewalk poses a safety risk to pedestrians and is not allowed in most urban areas.
2. [people, bicycling, sidewalk or road] = Anomaly, because riding on the sidewalk poses a safety risk to pedestrians and is not allowed in most urban areas.
3. [person, loitering, near parked cars] = Anomaly, because loitering near parked cars can indicate suspicious or potentially criminal behavior.
4. [person, throwing objects, public space] = Anomaly, because throwing objects in a public space can be dangerous and may cause harm to individuals or property.
5. [person, climbing, building facade] = Anomaly, because climbing a building facade is a dangerous and unauthorized activity that can lead to accidents or property damage.
6. [vehicle, driving in the wrong direction, one-way street] = Anomaly, because driving in the wrong direction on a one-way street is a traffic violation and poses a risk to other vehicles and pedestrians.
7. Any other object, action, environment that is unsafe = Anomaly'''
rule_v7_noenv = '''
**Rules for Normal:
1. [people, walking] = Normal
2. [people, standing] = Normal
3. [a group of people, walking] = Normal
4. [person, walking, carrying object] = Normal
5. [person, sitting] = Normal
6. [person, talking, on cell phone] = Normal 
7. No human-related actions or behaviors =  Normal


**Rules for Anomaly:
1. [person, running] = Anomaly
2. [skateboarder, riding] = Anomaly
3. [bike/motorcycle, riding] = Anomaly
4. [person, laying down] = Anomaly
5. [Vehicles/Van, driving] = Anomaly
6. [people, fighting/pushing] = Anomaly
7. [person, riding] = Anomaly
8. [person, jumping] = Anomaly
9. [Vehicles/Van, people] = Anomaly
10. [person, Not standing/Not walking/Not sitting] = Anomaly
'''


# gpt4v_deduction(rule_name='rule_v5_enhanced',prompt=rule_v5_enhanced, image_root="SHTech/test_50_1")
# rule_stage_1 = llm_induction_1('SHTech/object_data/train_5_0_vicuna-7b-v1.5.txt')
# rule_stage_2 = llm_induction_2(rule_stage_1)

# rule_stage_1 = llm_induction_1('SHTech/object_data/train_5_0_cogvlm.txt')
# rule_stage_2 = llm_induction_2(rule_stage_1)

# llm_rule_correction('/home/yyang/PycharmProjects/anomaly_detection/rule/rule_gpt4_wrong.txt')


# llm_deduction('SHTech/object_data/test_50_1_vicuna-7b-v1.5.txt', rule_v5_cogvlm_normal, 'rule_v5_cogvlm_normal')

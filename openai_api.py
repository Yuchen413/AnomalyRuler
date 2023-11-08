from openai import OpenAI
from utils import *
import base64
import re



def baseline():
    client = OpenAI(api_key="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")
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo"]
    model = model_list[0]
    txt_path = 'SHTech/test_50_0_owlvit.txt'
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



def baseline_with_rule():
    client = OpenAI(api_key="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")
    model_list = ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo"]
    model = model_list[0]
    txt_path = 'SHTech/test_50_1_instructblipobject.txt'
    objects = read_txt_to_list(txt_path)
    results = []
    count_0 = 0
    count_1 = 0
    for obj in objects:
        prompt = (
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

            Summary of rules:

            - Individuals or groups walking in designated pedestrian areas, like walkways and spacious paved areas, are considered normal.
            - Loitering, performing non-walking-related activities, or walking in undesignated areas, like roads with traffic, are considered anomalies.

            Now I am monitoring the campus and I'm given:{obj}, reply '{obj} = Normal' or '{obj} = Anomaly, because'
            """
        )
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

    if txt_path.split('_')[-2] == '0':
        print(count_0)
        print(f'Acc:{count_0 / len(results)}')
    elif txt_path.split('_')[-2] == '1':
        print(count_1)
        print(f'Acc:{count_1 / len(results)}')

    filename = f"results/rule_v2_{model}_{txt_path.split('/')[-1]}"
    with open(filename, 'w') as file:
        for string in results:
            file.write(string + '\n')

def gpt4v():
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
                                "First describe each from perspective of [object, action, environment], as simple as possible, e.g. use words or short sentences. "
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
    txt_path = 'SHTech/test_50_1_instructblip.txt'

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


# baseline()
baseline_with_rule()
# gpt4v()
# gpt_text2object()
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration,AutoProcessor, AutoModelForCausalLM, OwlViTProcessor, OwlViTForObjectDetection,Owlv2Processor, Owlv2ForObjectDetection, InstructBlipProcessor, InstructBlipForConditionalGeneration
from tqdm import tqdm
import torch
from utils import *

np.random.seed(2024)
torch.manual_seed(2024)
torch.cuda.memory_summary(device=None, abbreviated=False)

import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")



def blip(model_path, image_path = "SHTech/train_5_0"):
    image_paths = sorted(get_all_paths(image_path))
    batch_images = [Image.open(p) for p in image_paths]
    if model_path.startswith('Salesforce/blip2'):
        processor = Blip2Processor.from_pretrained(model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        inputs = processor(images=batch_images, return_tensors="pt").to(device, torch.float16)

        model.to(device)

    else:
        processor = InstructBlipProcessor.from_pretrained(model_path)
        model = InstructBlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model.to(device)

        prompt = "describe the image, including each object in it with its actions"
        prompts = []
        for i in range(len(batch_images)):
            prompts.append(prompt)
        inputs = processor(images=batch_images, text= prompts,return_tensors="pt").to(device, torch.float16)



    generated_ids = model.generate(**inputs, max_new_tokens=10000)
    generated_text = [i.strip() for i in processor.batch_decode(generated_ids, skip_special_tokens=True)]
    with open(f'{image_path}_{model_path.split("/")[1].split("-")[0]}.txt', 'w') as file:
        for i in generated_text:
            file.write(i + '\n')
    print(generated_text)


def git(model_path = "microsoft/git-large", image_path = "SHTech/train_50_0"):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    image_paths = sorted(get_all_paths(image_path))
    batch_images = [Image.open(p) for p in image_paths]

    pixel_values = processor(images=batch_images, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=1000)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    dict(zip(image_paths, generated_text))
    print(dict)


def owlvit(model_path = "google/owlv2-large-patch14", image_path = "SHTech/test_50_0"):
    # processor = OwlViTProcessor.from_pretrained(model_path)
    # model = OwlViTForObjectDetection.from_pretrained(model_path,torch_dtype=torch.float16, device_map ='auto')
    processor = Owlv2Processor.from_pretrained(model_path)
    model = Owlv2ForObjectDetection.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')
    image_paths = sorted(get_all_paths(image_path))
    print(len(image_paths))
    batch_images = [Image.open(p) for p in image_paths]
    texts = [["walking", "sitting", "groups", "running fast", "laying down", "cycling", "fighting", "vehicles", "gun", "person", "path", "open area", "road"]]
    objects = []
    for i in tqdm(range(len(batch_images))):
        inputs = processor(text=texts, images=batch_images[i], return_tensors="pt")
        outputs = model(**inputs)
        results = processor.post_process_object_detection(outputs=outputs, threshold=0.15)
        object = []
        for label in results[0]["labels"]:
            object.append(texts[0][label])
        print(list(set(object)))
        objects.append(list(set(object)))
    with open(f'{image_path.split("/")[0]}/object_data/{image_path.split("/")[1]}_{model_path.split("/")[1]}.txt', 'w') as file:
        for inner_list in objects:
            file.write(','.join(map(str, inner_list)) + '\n')
# owlvit()


blip('Salesforce/instructblip-flan-t5-xl', "SHTech/test_50_1")

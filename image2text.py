import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from transformers import Blip2Processor, Blip2ForConditionalGeneration,AutoProcessor, AutoModelForCausalLM, OwlViTProcessor, OwlViTForObjectDetection, InstructBlipProcessor, InstructBlipForConditionalGeneration
from tqdm import tqdm
from utils import *
from LLaVA.llava.eval.run_llava import *
from LLaVA.llava.mm_utils import get_model_name_from_path

np.random.seed(2024)
torch.manual_seed(2024)
torch.cuda.memory_summary(device=None, abbreviated=False)

import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

cropper = AllCrop(size=(224,224), stride=(256, 210)) #ori size:480x856 (68, 104) (128, 158) (256, 210)

def llava(model_path = 'liuhaotian/llava-v1.5-13b', image_path = 'SHTech/test_5_1', crop = False):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    image_paths = sorted(get_all_paths(image_path))
    if crop == True:
        images = [cropper(load_image(img_path)) for img_path in image_paths]
        num_crop = len(images[0])
        images = [item for sublist in images for item in sublist]
    else:
        images = [load_image(img_path) for img_path in image_paths]
        num_crop = 0

    text = []
    for image in images:
        args = type('Args', (), {
            "model_base": None,
            "model_name": model_name,
            "query": 'What are in the image? Use one short sentence to describe the objects and their actions',
            "conv_mode": None,
        })()

        qs = args.query
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode,
                                                                                                                  args.conv_mode,
                                                                                                               args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print(outputs)
        text.append(outputs)
    if crop == True:
        text = list(np.array(text).reshape((len(image_paths), num_crop)))
        with open(f'{image_path.split("/")[0]}/object_data/{image_path.split("/")[1]}_{model_path.split("/")[1]}.txt','w') as file:
            for inner_list in text:
                file.write(','.join(map(str, inner_list)) + '\n')
    else:
        with open(f'{image_path.split("/")[0]}/object_data/{image_path.split("/")[1]}_{model_path.split("/")[1]}.txt',
                  'w') as file:
            for inner_list in text:
                file.write(inner_list + '\n')



def blip(model_path, image_path = "SHTech/test_5_1"):
    image_paths = sorted(get_all_paths(image_path))
    images = [cropper(Image.open(p)) for p in image_paths]
    num_crop = len(images[0])
    images = [item for sublist in images for item in sublist]
    bs = num_crop
    print(bs)
    batch_images = split_list(images, bs)

    text = []
    if model_path.startswith('Salesforce/blip2'):
        processor = Blip2Processor.from_pretrained(model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model.to(device)
        for i in range(len(batch_images)):
            inputs = processor(images=batch_images[i], return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs)
            generated_text = [i.strip() for i in processor.batch_decode(generated_ids, skip_special_tokens=True)]
            text.append(generated_text)

    else:
        processor = InstructBlipProcessor.from_pretrained(model_path)
        model = InstructBlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model.to(device)
        prompt = "Detect the objects"
        for i in range(len(batch_images)):
            prompts = [prompt]*len(batch_images[i])
            inputs = processor(images=batch_images[i], text= prompts,return_tensors="pt").to(device,torch.float16)
            generated_ids = model.generate(**inputs)
            generated_text = [i.strip() for i in processor.batch_decode(generated_ids, skip_special_tokens=True)]
            text.append(generated_text)
    text = list(np.array(text).reshape((len(image_paths), num_crop)))
    with open(f'{image_path.split("/")[0]}/object_data/{image_path.split("/")[1]}_{model_path.split("/")[1]}.txt', 'w') as file:
        for inner_list in text:
            file.write(','.join(map(str, inner_list)) + '\n')
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
def owlvit(model_path = "google/owlvit-large-patch14", image_path = "SHTech/test_50_0"):
    processor = OwlViTProcessor.from_pretrained(model_path)
    model = OwlViTForObjectDetection.from_pretrained(model_path,torch_dtype=torch.float16, device_map ='auto')
    image_paths = sorted(get_all_paths(image_path))
    batch_images = [cropper(Image.open(p)) for p in image_paths]
    num_crop = len(batch_images[0])
    batch_images = [item for sublist in batch_images for item in sublist]
    texts = [["walking", "sitting", "groups", "running fast", "laying down", "cycling", "attack", "vehicles", "gun", "person", "path", "open area", "road", "motorbike","van", "park"]]
    objects = []
    for i in tqdm(range(len(batch_images))):
        inputs = processor(text=texts, images=batch_images[i], return_tensors="pt")
        outputs = model(**inputs)
        results = processor.post_process_object_detection(outputs=outputs, threshold=0.19)
        obj = []
        for label in results[0]["labels"]:
            obj.append(texts[0][label])
        objects.append(list(set(obj)))
        print(list(set(obj)))
    objects = split_list(objects, num_crop)
    with open(f'{image_path.split("/")[0]}/object_data/{image_path.split("/")[1]}_{model_path.split("/")[1]}.txt', 'w') as file:
        for inner_list in objects:
            file.write(','.join(map(str, inner_list)) + '\n')

# owlvit(model_path = "google/owlvit-large-patch14", image_path = "SHTech/train_5_0")

# blip('Salesforce/blip2-flan-t5-xl', "SHTech/test_50_0")

llava(model_path = 'liuhaotian/llava-v1.5-13b', image_path = 'SHTech/test_50_0', crop = True)
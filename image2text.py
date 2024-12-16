import torch
from transformers import LlamaTokenizer, AutoProcessor, AutoModelForCausalLM, OwlViTProcessor, OwlViTForObjectDetection,InstructBlipProcessor, InstructBlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
from utils import *
import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

cropper = AllCrop(size=(224,224), stride=(256, 210)) #ori size:480x856 (68, 104) (128, 158) (256, 210)


def cogvlm(model, image_paths, mode = 'chat', root_path = None, model_path = 'lmsys/vicuna-7b-v1.5'):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    query= 'How many people are in the image and what is each of them doing? What are in the images other than people? Think step by step'
    if root_path != None:
        image_paths = sorted(get_all_paths(root_path))

    batch_images = [Image.open(p) for p in image_paths]
    description = []

    # for count, query in enumerate(queries):
    for image in batch_images:
        if mode == 'chat':
            inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # vqa mode
        else:
            inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image],
                                                        template_version='vqa')  # vqa mode

        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }

        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            description.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return description

def continue_frame(data_root_name):
    file_path = f'{data_root_name}/test.csv'  # Replace with your actual file path
    df = pd.read_csv(file_path)
    image_path_sample = df['image_path'][0]
    last_slash_pos = image_path_sample.rfind('/')
    second_last_slash_pos = image_path_sample.rfind('/', 0, last_slash_pos)

    # Extract all unique number segments from the paths
    unique_segments = df.iloc[:, 0].apply(lambda x: x[second_last_slash_pos+1:last_slash_pos].split('/')[0]).unique()
    print(unique_segments)
    if not os.path.exists(f'{data_root_name}/test_frame'):
        os.makedirs(f'{data_root_name}/test_frame')
    for i in unique_segments:
        filtered_df = df[df.iloc[:, 0].apply(lambda x: x[second_last_slash_pos+1:last_slash_pos].split('/')[0]== i)]
        filtered_df.to_csv(f'{data_root_name}/test_frame/test_{i}.csv', index=False)

def get_description_frame(data_root_name):
    cog_model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True
    ).eval()
    all_video_csv_paths = get_all_paths(f'{data_root_name}/test_frame')
    print(all_video_csv_paths)
    for video_csv_path in all_video_csv_paths[31:]:   #:30 31:(test_abnormal_scene_4_scenario_7)
        name = video_csv_path.split('/')[-1].split('.')[0]
        print(name)
        df = pd.read_csv(video_csv_path)
        img_paths_per_video = df.iloc[:, 0].tolist()
        descriptions_per_video = cogvlm(model=cog_model, mode='chat', image_paths=img_paths_per_video)
        if not os.path.exists(f'{data_root_name}/test_frame_description'):
            os.makedirs(f'{data_root_name}/test_frame_description')
        with open(f'{data_root_name}/test_frame_description/{name}.txt', 'w') as file:
            for inner_list in descriptions_per_video:
                file.write(inner_list + '\n')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='SHTech',
                        choices=['SHTech', 'avenue', 'ped2', 'UBNormal'])
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    data_name = args.data
    continue_frame(data_name)
    get_description_frame(data_name)


if __name__ == "__main__":
    main()




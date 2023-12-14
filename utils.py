import numpy as np
import pandas as pd
import os
import shutil
import torchvision.transforms.functional as TF
import torch.nn as nn
from IPython.display import display
from PIL import Image
import re
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

np.random.seed(2024)

def split_list(data, size):
    # Splits 'data' into sublists of length 'size'
    return [data[i:i + size] for i in range(0, len(data), size)]
def display_images_in_one_row(images):
    # Calculate total width and maximum height
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new blank image with the calculated size
    composite_image = Image.new('RGB', (total_width, max_height))

    # Paste each image into the composite image
    x_offset = 0
    for img in images:
        composite_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Show the composite image
    composite_image.show()


class AllCrop(nn.Module):
    def __init__(self, size=(224, 224), stride=(128, 158)):
        super(AllCrop, self).__init__()

        self.height, self.width = size
        self.stride_h, self.stride_w = stride

    def forward(self, input):
        # Assuming 'input' is a PIL Image
        image_width, image_height = input.size

        all_crop = []
        for h in range(0, image_height - self.height + 1, self.stride_h):
            for w in range(0, image_width - self.width + 1, self.stride_w):
                crop = TF.crop(input, h, w, self.height, self.width)
                all_crop.append(crop)
        # Optional: Convert crops to tensors if needed
        # all_crop = [TF.to_tensor(c) for c in all_crop]
        # display_images_in_one_row(all_crop)
        return all_crop
def create_csv():
    source_directory = 'SHTech/train'
    file_paths = []
    for root, _, files in os.walk(source_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    # labels = list(np.load('SHTech/frame_labels_shanghai.npy'))
    labels = list(np.zeros(len(file_paths), dtype=int))

    data = {'image_path': sorted(file_paths), 'label': labels}
    df = pd.DataFrame(data)
    csv_file_path = 'SHTech/train.csv'
    df.to_csv(csv_file_path, index=False)


def random_select_data(path = 'SHTech/train.csv', num = 100, label = 0):
    df = pd.read_csv(path)
    image_file_paths = list(df.loc[df['label'] == label, 'image_path'].values)
    selected_image_paths = np.random.choice(image_file_paths, num, replace=False)
    print(len(set(selected_image_paths)))
    destination_directory = path[:-4]+'_'+str(num)+'_'+str(label)
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    else:
        shutil.rmtree(destination_directory)  # Removes all the subdirectories!
        os.makedirs(destination_directory)
    for image_path in selected_image_paths:
        path_parts = image_path.split('/')
        image_filename= '_'.join(path_parts[-2:])
        destination_path = os.path.join(destination_directory, image_filename)
        shutil.copy2(image_path, destination_path)

def get_all_paths(directory):
    """Get all file paths under a directory and return them as a list."""
    file_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))

    return file_paths

def read_txt_to_list(path = 'SHTech/test_owlvit.txt'):
    list_of_lists = []
    with open(path, 'r') as file:
        for line in file:
            inner_list = line.strip().split(',')
            inner_list = [item for item in inner_list]
            list_of_lists.append(inner_list)
    return list_of_lists

def read_txt_to_one_list(path = 'SHTech/test_owlvit.txt'):
    lines = []
    with open(path, 'r') as file:
        for line in file:
            lines += line.strip().split(',')
    return lines

def read_line(path = 'SHTech/test_owlvit.txt'):
    with open(path, 'r') as file:
        lines = [line.strip().split('\n') for line in file]
    return lines


def post_process(text):
    key_phrases = ['Normal', 'Anomaly']
    answer_index = text.find('Answer:')
    if answer_index != -1:
        # Extract the substring starting from 'Answer:'
        substring = text[answer_index + len('Answer:'):]
        # Split the substring into words and return the first one that contains a key phrase
        words = substring.split('.')[0]
        for phrase in key_phrases:
            if phrase in words:
                if phrase=='Anomaly' : return 1
                if phrase=='Normal': return 0
    print("Neither 'Normal' nor 'Anomaly' found in the specified locations.")
    return -1

def read_and_process_file(file_path = 'SHTech/object_data/train_100_0_vicuna-7b-v1.5_act+env.txt'):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    # Splitting based on numbers or commas
    split_content = re.split(r'\d+\.|,', content)
    # Removing non-English parts
    # Assuming non-English parts can be identified, for example, by being enclosed in parentheses
    # This step will depend on how the non-English text is formatted
    processed_content = [part for part in split_content if not re.search(r'[^\x00-\x7F]+', part) and not None]
    cleaned_content = [re.sub(r'[\d\W]+', ' ', part) for part in processed_content]

    # Removing redundant parts
    # This step will also depend on what is considered redundant in your context
    unique_content = list(set(cleaned_content))

    output_path = f"rule/{file_path.split('/')[-1]}"

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for line in unique_content:
            output_file.write(line.strip() + '\n')

    return unique_content

read_and_process_file()
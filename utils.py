import numpy as np
import pandas as pd
import os
import shutil
import torchvision.transforms.functional as TF
import torch.nn as nn

np.random.seed(2024)


class AllCrop(nn.Module):
    def __init__(self, size=(224, 224), stride=(128, 158)):
        super(AllCrop, self).__init__()

        self.height, self.width = size
        self.stride_h, self.stride_w = stride

    def forward(self, input):

        _, image_height, image_width = TF.to_tensor(input).size()

        all_crop = []
        for h in range(0, image_height - self.height + 1, self.stride_h):
            for w in range(0, image_width - self.width + 1, self.stride_w):
                all_crop.append(TF.crop(input, h, w, self.height, self.width))
        print(all_crop)
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


def random_select_data(path = 'SHTech/test.csv', num = 5, label = 1):
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




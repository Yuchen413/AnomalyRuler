import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPModel
# from utils import *
import random
import torch
import numpy as np
import cv2
import glob
import os
import scipy.io as scio
from torch.utils.data import Dataset
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

import os
import shutil

class VideoOrganizer:
    def __init__(self, data_folder, train_folder, test_normal_folder, test_abnormal_folder):
        self.data_folder = data_folder
        self.train_folder = train_folder
        self.test_normal_folder = test_normal_folder
        self.test_abnormal_folder = test_abnormal_folder
        # Create destination folders if they do not exist
        # os.makedirs(self.train_folder, exist_ok=True)
        # os.makedirs(self.test_normal_folder, exist_ok=True)
        # os.makedirs(self.test_abnormal_folder, exist_ok=True)

    def read_video_list(self, file_path):
        with open(file_path, 'r') as file:
            return [line.strip() for line in file]

    def copy_videos(self, video_list, destination_folder):
        for scene_folder in os.listdir(self.data_folder):
            scene_path = os.path.join(self.data_folder, scene_folder)
            if os.path.isdir(scene_path):
                for video in os.listdir(scene_path):
                    src = os.path.join(scene_path, video)
                    dst = os.path.join(destination_folder, video)
                    # Check if the file name without extension is in the list and if src and dst are not the same
                    if os.path.splitext(video)[0] in video_list and src != dst:
                        print(f"Copying {video} to {destination_folder}")
                        shutil.copy2(src, dst)
                    else:
                        print(f"Skipping {video}")

    def organize_videos(self, normal_train_file, normal_test_file, abnormal_test_file):
        normal_train_videos = self.read_video_list(normal_train_file)
        normal_test_videos = self.read_video_list(normal_test_file)
        abnormal_test_videos = self.read_video_list(abnormal_test_file)

        self.copy_videos(normal_train_videos, self.train_folder)
        self.copy_videos(normal_test_videos, self.test_normal_folder)
        self.copy_videos(abnormal_test_videos, self.test_abnormal_folder)



# organizer = VideoOrganizer('data', 'data/train', 'data/test/normal', 'data/test/abnormal')
# organizer.organize_videos('normal_training_video_names.txt', 'normal_validation_video_names.txt', 'abnormal_validation_video_names.txt')


import cv2
import os

class VideoFrameExtractor:
    def __init__(self, base_folders):
        """
        Initialize the VideoFrameExtractor with base folders.
        :param base_folders: List of folders containing videos to process.
        """
        self.base_folders = base_folders

    @staticmethod
    def extract_frames(video_path, destination_folder):
        """
        Extract frames from a video and save them in a specified folder.
        :param video_path: Path to the video file.
        :param destination_folder: Folder where extracted frames will be saved.
        """
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_path = os.path.join(destination_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        cap.release()

    def process_folder(self, folder_path):
        """
        Process all videos in a folder.
        :param folder_path: Path to the folder containing videos.
        """
        for video in os.listdir(folder_path):
            if video.endswith('.mp4'):
                video_path = os.path.join(folder_path, video)
                frame_folder = os.path.join(folder_path, os.path.splitext(video)[0])  # Remove .mp4
                self.extract_frames(video_path, frame_folder)

    def process_all_folders(self):
        """ Process all base folders. """
        for folder in self.base_folders:
            self.process_folder(folder)

# Usage
# base_folders = ['data/test/normal', 'data/test/abnormal', 'data/train']
# extractor = VideoFrameExtractor(base_folders)
# extractor.process_all_folders()


class Label_loader_save:
    def __init__(self,name):
        self.name = name
        self.frame_path = f'{name}/test'
        self.mat_path = f'{name}/{name}.mat'
        video_folders = os.listdir(self.frame_path)
        video_folders.sort()
        self.video_folders = [os.path.join(self.frame_path, aa) for aa in video_folders]

    def __call__(self):
        data = self.load_ucsd_avenue()
        df = pd.DataFrame(data, columns=['image_path', 'label'])
        df.to_csv(f'{self.name}/test.csv', index=False)

    def load_ucsd_avenue(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']
        all_data = []
        for i, folder in enumerate(self.video_folders):
            frame_files = sorted(os.listdir(folder))
            length = len(frame_files)
            sub_video_gt = np.zeros((length,), dtype=np.int8)

            one_abnormal = abnormal_events[i]
            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

            for j in range(one_abnormal.shape[1]):
                start = one_abnormal[0, j] - 1
                end = one_abnormal[1, j]
                sub_video_gt[start: end] = 1

            for frame, label in zip(frame_files, sub_video_gt):
                all_data.append([os.path.join(folder, frame), label])

        return all_data

# gt_loader = Label_loader_save('ped2')  # Get gt labels.
# gt = gt_loader()

def create_train_csv(data_name):
    source_directory = f'{data_name}/train'
    file_paths = []
    for root, _, files in os.walk(source_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    # labels = list(np.load('SHTech/frame_labels_shanghai.npy'))
    labels = list(np.zeros(len(file_paths), dtype=int))

    data = {'image_path': sorted(file_paths), 'label': labels}
    df = pd.DataFrame(data)
    csv_file_path = f'{data_name}/train.csv'
    df.to_csv(csv_file_path, index=False)

# create_train_csv('UBNormal')

def create_test_UBNormal_csv(data_name):
    source_directory = f'{data_name}/test'
    file_paths = []
    for root, _, files in os.walk(source_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    # labels = list(np.load('SHTech/frame_labels_shanghai.npy'))
    labels = [1 if 'abnormal' in name else 0 for name in sorted(file_paths)]

    data = {'image_path': sorted(file_paths), 'label': labels}
    df = pd.DataFrame(data)
    csv_file_path = f'{data_name}/test.csv'
    df.to_csv(csv_file_path, index=False)

# create_test_UBNormal_csv('UBNormal')

class TrainDataset(Dataset):
    def __init__(self, path = 'SHTech/train_1000_0.pt'):
        self.x = torch.load(path)
        self.x = self.x.reshape(2000, 2*768)
        self.y = torch.cat((torch.ones(1000, dtype=torch.long), torch.zeros(1000, dtype=torch.long)), dim=0)

    def __len__(self):
        # Return the total number of samples
        return len(self.x)

    def __getitem__(self, idx):
        # Return a single sample and its label
        return self.x[idx], self.y[idx]

class TestDataset(Dataset):
    def __init__(self, path = 'SHTech/test_100.pt'):
        self.x = torch.load(path)
        self.x = self.x.reshape(200, 2*768)
        self.y = torch.cat((torch.ones(50, dtype=torch.long), torch.zeros(50, dtype=torch.long),torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)), dim=0)

    def __len__(self):
        # Return the total number of samples
        return len(self.x)

    def __getitem__(self, idx):
        # Return a single sample and its label
        return self.x[idx], self.y[idx]



def clip_feature_extractor():
    # Initialize CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16,
                                      low_cpu_mem_usage=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.to(device)

    # Load an image
    image_path = 'SHTech/test_50_1'
    normal_rule = read_txt('rule/normal_rule_3.txt')
    anomaly_rule = read_txt('rule/anomaly_rule_3.txt')
    image_paths = sorted(get_all_paths(image_path))
    images = [Image.open(p) for p in image_paths]
    train_data = torch.zeros((100, 2, 768))
    for i in range(len(images)):
        with torch.no_grad():
            inputs_n = processor(text=normal_rule, images=images[i], return_tensors="pt", padding=True)
            inputs_a = processor(text=anomaly_rule, images=images[i], return_tensors="pt", padding=True)
            inputs_n['pixel_values'] = inputs_n['pixel_values'].half()
            inputs_a['pixel_values'] = inputs_a['pixel_values'].half()
            inputs_a.to(device)
            inputs_n.to(device)
            outputs_n = model(**inputs_n)
            outputs_a = model(**inputs_a)
            image_embedding = outputs_n.image_embeds
            max_index_n = int(torch.sort(outputs_n['logits_per_image'][0], descending=True)[1][0])
            max_index_a = int(torch.sort(outputs_a['logits_per_image'][0], descending=True)[1][0])
            text_embeddings_n = outputs_n.text_embeds[max_index_n].reshape(image_embedding.shape)
            text_embeddings_a = outputs_a.text_embeds[max_index_a].reshape(image_embedding.shape)
            image_text_n = torch.cat((image_embedding, text_embeddings_n), 0)
            image_text_a = torch.cat((image_embedding, text_embeddings_a), 0)
            train_data[i] = image_text_n
            train_data[i + 50] = image_text_a
    torch.save(train_data, 'SHTech/test_50_1.pt')

# clip_feature_extractor()
# normal = torch.load('SHTech/test_50_0.pt')
# anomal = torch.load('SHTech/test_50_1.pt')
# test = torch.cat((normal,anomal),dim=0)
# torch.save(test, 'SHTech/test_100.pt')

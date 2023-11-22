import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPModel
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
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

from transformers import CLIPProcessor, CLIPModel
import torch
from utils import *

np.random.seed(2024)
torch.manual_seed(2024)
torch.cuda.memory_summary(device=None, abbreviated=False)


device = "cuda" if torch.cuda.is_available() else "cpu"
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",torch_dtype=torch.float16, low_cpu_mem_usage=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.to(device)

# Load an image
image_path = 'SHTech/test_50_0'
normal_rule = read_txt('rule/normal_rule_3.txt')
anomaly_rule = read_txt('rule/anomaly_rule_3.txt')
image_paths = sorted(get_all_paths(image_path))
images = [Image.open(p) for p in image_paths]

count_n_mean = 0
count_a_mean = 0
count_n_max = 0
count_a_max = 0
count_n_top = 0
count_a_top = 0

for i in images:
    with torch.no_grad():
        inputs_n = processor(text=normal_rule, images=i, return_tensors="pt", padding=True)
        inputs_a = processor(text=anomaly_rule, images=i, return_tensors="pt", padding=True)
        inputs_n['pixel_values'] = inputs_n['pixel_values'].half()
        inputs_a['pixel_values'] = inputs_a['pixel_values'].half()
        inputs_a.to(device)
        inputs_n.to(device)
        outputs_n = model(**inputs_n)
        outputs_a = model(**inputs_a)

        max_cosine_n = torch.max(outputs_n['logits_per_image'][0].to('cpu'))
        max_cosine_a = torch.max(outputs_a['logits_per_image'][0].to('cpu'))

        if max_cosine_a > max_cosine_n:
            count_a_max += 1
        else: count_n_max += 1

    # mean_cosine_n = torch.mean(outputs_n['logits_per_image'][0])
    # mean_cosine_a = torch.mean(outputs_a['logits_per_image'][0])
    # if mean_cosine_a > mean_cosine_n:
    #     count_a_mean += 1
    # else: count_n_mean += 1
    # topk_mean_cosine_n = torch.mean(torch.sort(outputs_n['logits_per_image'][0], descending=True)[0][:3])
    # topk_mean_cosine_a = torch.mean(torch.sort(outputs_a['logits_per_image'][0], descending=True)[0][:3])
    # if topk_mean_cosine_a  > topk_mean_cosine_n:
    #     count_a_top += 1
    # else: count_n_top += 1

print(count_n_max, count_a_max)
# print(max_n/1000.0)
# print(max_a/1000.0)


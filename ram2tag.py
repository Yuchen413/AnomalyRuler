import argparse
import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
from utils import get_all_paths
import numpy as np

np.random.seed(2024)
torch.manual_seed(2024)
torch.cuda.memory_summary(device=None, abbreviated=False)


device = "cuda" if torch.cuda.is_available() else "cpu"
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")


parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='SHTech/train_5_0/train_01_054_0237.jpg')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='recognize-anything/pretrained/ram_plus_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')


if __name__ == "__main__":

    args = parser.parse_args()

    transform = get_transform(image_size=args.image_size)
    image_path = 'SHTech/test_5_0'
    image_paths = sorted(get_all_paths(image_path))
    # batch_images = [Image.open(p) for p in image_paths]
    model = ram_plus(pretrained=args.pretrained,
                             image_size=args.image_size,
                             vit='swin_l')
    model.eval()
    model = model.to(device)
    for image_path in image_paths:
        image = transform(Image.open(image_path)).unsqueeze(0).to(device)
        res = inference(image, model)
        print("Image Tags: ", res[0])
        # print("图像标签: ", res[1])

import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
from utils import *
from PIL import Image
import pytorch_ood
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

np.random.seed(2024)
torch.manual_seed(2024)
torch.cuda.memory_summary(device=None, abbreviated=False)

class CLIPVerifier:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")

        # Initialize CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)

    @staticmethod
    def read_txt(file_path):
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file]
        return lines

    @staticmethod
    def read_csv(file_path):
        df = pd.read_csv(file_path)
        return df['image_path'].values, df['label'].values


    @staticmethod
    def normalize(lst):
        minimum = min(lst)
        maximum = max(lst)
        normalized_list = [(x - minimum) / (maximum - minimum) for x in lst]
        return normalized_list

    def get_scored_rule(self, image_path, rule_path):
        rules = self.read_txt(rule_path)
        image_paths = sorted(get_all_paths(image_path))
        images = [Image.open(p) for p in image_paths]
        rule_scores_max = {rule: [] for rule in rules}
        rule_scores_min= {rule: [] for rule in rules}

        for i in images:
            with torch.no_grad():
                inputs = self.processor(text=rules, images=i, return_tensors="pt", padding=True)
                inputs['pixel_values'] = inputs['pixel_values'].half()
                inputs.to(self.device)
                outputs = self.model(**inputs)
                sorted_values, sorted_indices = torch.sort(outputs['logits_per_image'][0].to('cpu'),descending=True)
                max_value = sorted_values[0].item()
                max_value_index = sorted_indices[0].item()
                min_value = sorted_values[-1].item()
                min_value_index = sorted_indices[-1].item()
                rule_scores_max[rules[max_value_index]].append(max_value)
                rule_scores_min[rules[min_value_index]].append(min_value)

        return rule_scores_max, rule_scores_min

    def rule_verifier(self, image_path, normal_rule_path, anomaly_rule_path):
        max_rule_scores_normal, _ = self.get_scored_rule(image_path, normal_rule_path)
        max_rule_scores_anomaly, _ = self.get_scored_rule(image_path, anomaly_rule_path)
        _, min_rule_scores_anomaly = self.get_scored_rule(image_path, anomaly_rule_path)
        _, min_rule_scores_normal = self.get_scored_rule(image_path, normal_rule_path)
        verified_normal_rule = [key for key, value in max_rule_scores_normal.items() if len(value)>0]
        verified_anomaly_rule = [key for key, value in max_rule_scores_anomaly.items() if len(value)>0]
        print(f'Normal rules: {verified_normal_rule}')
        print(f'Anomaly rules: {verified_anomaly_rule}')
        return verified_normal_rule, verified_anomaly_rule

    def get_score(self, score, softmax, image_path, rule_path, rules=None):
        to_np = lambda x: x.data.cpu().numpy()
        if rules == None:
            rules = self.read_txt(rule_path)
        image_paths = sorted(get_all_paths(image_path))
        images = [Image.open(p) for p in image_paths]
        _score = []
        for i in images:
            with torch.no_grad():
                inputs = self.processor(text=rules, images=i, return_tensors="pt", padding=True)
                inputs['pixel_values'] = inputs['pixel_values'].half()
                inputs.to(self.device)
                output = self.model(**inputs)['logits_per_image'][0].to(dtype=torch.float32).unsqueeze(0)
                if softmax:
                    smax = to_np(F.softmax(output / 1, dim=1))
                else:
                    smax = to_np(output / 1)
                if score == 'energy':
                    # Energy = - T * logsumexp(logit_k / T), by default T = 1 in https://arxiv.org/pdf/2010.03759.pdf
                    _score.append(-to_np((1 * torch.logsumexp(output / 1, dim=1))))  # energy score is expected to be smaller for ID
                elif score == 'entropy':
                    _score.append(entropy(smax, axis=1))
                    # _score.append(filtered)
                elif score == 'var':
                    _score.append(-np.var(smax, axis=1))
                elif score == 'MCM':
                    _score.append(-np.max(smax, axis=1))
                elif score == 'max':
                    _score.append(np.max(smax))
                elif score == 'mean':
                    _score.append(np.mean(smax))
        scores_sorted = sorted(_score, reverse=True)
        index = max(int(len(scores_sorted) * 0.95) - 1, 0)
        threshold = scores_sorted[index]
        return _score, threshold

    def deduct(self,score, softmax, image_normal_path, image_anomaly_path, normal_rule_path, anomaly_rule_path, normal_rule = None, anomaly_rule = None):

        image_paths = [image_normal_path, image_anomaly_path]
        if_correct_normal = []
        if_correct_anomaly = []
        label = []
        norm_scores = []

        for image_path in image_paths:
            scores_normal, _ = self.get_score(score, softmax,image_path, normal_rule_path, normal_rule)
            scores_anomaly, _ = self.get_score(score, softmax,image_path, anomaly_rule_path, anomaly_rule)
            scores = [scores_anomaly[i] - scores_normal[i] for i in range(len(scores_normal))]
            norm_scores += self.normalize(scores)

            if image_path.split('_')[-1] == '0':
                label += [0]*len(scores)
                if_correct_normal = [scores[i] <= 0 for i in range(len(scores))]
            else:
                label += [1]*len(scores)
                if_correct_anomaly = [scores[i] > 0 for i in range(len(scores))]
        print('=================================================')
        print(sum(if_correct_normal))
        print(sum(if_correct_anomaly))
        print(f'ACC: {(sum(if_correct_normal) + sum(if_correct_anomaly)) / 100}')
        print(f'Precision: {(sum(if_correct_anomaly)) / ((sum(if_correct_anomaly)) + (50-sum(if_correct_normal)))}')
        print(f'Recall: {sum(if_correct_anomaly) / 50}')
        print(f'AUC: {roc_auc_score(label, norm_scores)}')

    def deduct_normal(self, score, softmax, threshold, image_normal_path, image_anomaly_path, normal_rule_path, normal_rule = None):
        scores_normal, _ = self.get_score(score, softmax,image_normal_path, normal_rule_path, normal_rule)
        scores_anomaly, _ = self.get_score(score, softmax, image_anomaly_path, normal_rule_path, normal_rule)
        if_correct_normal = [scores_normal[i] >= threshold for i in range(len(scores_normal))]
        if_correct_anomaly = [scores_anomaly[i] < threshold for i in range(len(scores_normal))]
        print('=================================================')
        print(sum(if_correct_normal))
        print(sum(if_correct_anomaly))
        print(f'ACC: {(sum(if_correct_normal) + sum(if_correct_anomaly)) / 100}')
        print(f'Precision: {(sum(if_correct_anomaly)) / ((sum(if_correct_anomaly)) + (50-sum(if_correct_normal)))}')
        print(f'Recall: {sum(if_correct_anomaly) / 50}')


    def get_score_all(self, score, softmax, image_paths, rule_path, rules=None):
        to_np = lambda x: x.data.cpu().numpy()
        if rules == None:
            rules = self.read_txt(rule_path)
        images = [Image.open(p) for p in image_paths]
        _score = []
        for i in tqdm(images):
            with torch.no_grad():
                inputs = self.processor(text=rules, images=i, return_tensors="pt", padding=True)
                inputs['pixel_values'] = inputs['pixel_values'].half()
                inputs.to(self.device)
                output = self.model(**inputs)['logits_per_image'][0].to(dtype=torch.float32).unsqueeze(0)
                if softmax:
                    smax = to_np(F.softmax(output / 1, dim=1))
                else:
                    smax = to_np(output / 1)
                if score == 'energy':
                    # Energy = - T * logsumexp(logit_k / T), by default T = 1 in https://arxiv.org/pdf/2010.03759.pdf
                    _score.append(-to_np((1 * torch.logsumexp(output / 1, dim=1))))  # energy score is expected to be smaller for ID
                elif score == 'entropy':
                    _score.append(entropy(smax, axis=1))
                    # _score.append(filtered)
                elif score == 'var':
                    _score.append(-np.var(smax, axis=1))
                elif score == 'MCM':
                    _score.append(-np.max(smax, axis=1))
                elif score == 'max':
                    _score.append(np.max(smax))
                elif score == 'mean':
                    _score.append(np.mean(smax))
        scores_sorted = sorted(_score, reverse=True)
        index = max(int(len(scores_sorted) * 0.95) - 1, 0)
        threshold = scores_sorted[index]
        return _score, threshold

    def deduct_all(self,score, softmax, image_label_path, normal_rule_path, anomaly_rule_path, normal_rule = None, anomaly_rule = None):

        image_paths, label = self.read_csv(image_label_path)
        print(len(label))
        scores_normal, _ = self.get_score_all(score, softmax, image_paths, normal_rule_path, normal_rule)
        scores_anomaly, _ = self.get_score_all(score, softmax, image_paths, anomaly_rule_path, anomaly_rule)
        scores = [scores_anomaly[i] - scores_normal[i] for i in range(len(scores_normal))]
        norm_scores = self.normalize(scores)
        predict = [0 if i <= 0 else 1 for i in scores]

        print('=================================================')
        print(f'ACC: {accuracy_score(label, predict)}')
        print(f'Precision: {precision_score(label,predict)}')
        print(f'Recall: {recall_score(label,predict)}')
        print(f'AUC: {roc_auc_score(label, norm_scores)}')




# Usage
n_rule_path = 'rule/normal_rule_3.txt'
a_rule_path = 'rule/anomaly_rule_3.txt'
clip_processor = CLIPVerifier()
verified_normal_rule, verified_anomaly_rule = clip_processor.rule_verifier('SHTech/train_5_0', n_rule_path, a_rule_path)
score_type = ['entropy', 'var', 'MCM', 'max','mean']
score = score_type[3]
softmax = False
'''
Use both normal/anomaly rules
'''
clip_processor.deduct_all(score, softmax,'SHTech/test.csv', n_rule_path, a_rule_path)
clip_processor.deduct_all(score, softmax,'SHTech/test.csv', n_rule_path, a_rule_path, verified_normal_rule,verified_anomaly_rule)
# clip_processor.deduct(score, softmax,'SHTech/test_50_0', 'SHTech/test_50_1', n_rule_path, a_rule_path)
# clip_processor.deduct(score, softmax,'SHTech/test_50_0', 'SHTech/test_50_1', n_rule_path, a_rule_path, verified_normal_rule,verified_anomaly_rule)

'''
Only use normal rules
'''
# _, threshold = clip_processor.get_score(score, softmax,'SHTech/train_5_0', n_rule_path)
# clip_processor.deduct_normal(score, softmax, threshold, 'SHTech/test_50_0', 'SHTech/test_50_1',  n_rule_path)
# _, threshold = clip_processor.get_score(score, softmax,'SHTech/train_5_0', n_rule_path, verified_normal_rule)
# clip_processor.deduct_normal(score, softmax,threshold,'SHTech/test_50_0', 'SHTech/test_50_1', n_rule_path, verified_normal_rule)


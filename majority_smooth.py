from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from utils import *
import re
from collections import Counter

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def cluster_kmeans(sentences, num_clusters=2):
    # Vectorize the sentences using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_

def clsuter_keyword(text_lines):
    anomaly_from_rule = [
    "trolley",
    "cart",
    "luggage",
    "bicycle",
    "skateboard",
    "scooter",
    "vehicles",
    "vans",
    "running",
    "riding",
    "skateboarding",
    "scooting",
    "lying",
    "bending",
    "fighting",
    "loitering",
    "climbing",
    "tampering",
    "lingering"]
    preds = []
    for line in text_lines:
        found_anomaly = False
        for anomaly in anomaly_from_rule:
            if anomaly in line:
                found_anomaly = True
        preds.append(1 if found_anomaly else 0)
    return preds, anomaly_from_rule

def old_custom_smooth(data, window_size=20):
    # Ensure the window size is odd to have a central element
    if window_size % 2 == 0:
        window_size += 1

    # Pad the data at the beginning and end to handle edge cases
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='edge')

    # Initialize the smoothed data array
    smoothed_data = np.copy(data)

    # Iterate through the data and smooth
    for i in range(len(data)):
        # Determine the start and end indices of the window
        start = i
        end = i + window_size

        # Slice the window and count the occurrences
        window = padded_data[start:end]
        ones_count = np.sum(window)
        zeros_count = window_size - ones_count
        # Assign new value based on the majority
        smoothed_data[i] = 1 if ones_count > zeros_count else 0
    return smoothed_data

def majority_smooth(data, window_size=20, edge_region_size=None):
    # Adjust window size to be odd
    if window_size % 2 == 0:
        window_size += 1
    pad_size = window_size // 2

    # Determine edge region size if not specified
    if edge_region_size is None:
        edge_region_size = pad_size

    padded_data = np.pad(data, pad_size, mode='edge')
    smoothed_data = np.copy(data)

    for i in range(len(data)):
        # Apply different rule for edge regions
        if i < edge_region_size or i >= len(data) - edge_region_size:
            # For edge data, consider only the previous pad_size values
            start = max(0, i - pad_size)
            end = i + 1  # Include the current point
            window = padded_data[start:end]
        else:
            # Regular processing for central part
            start = i
            end = i + window_size
            window = padded_data[start:end]

        # Apply majority rule
        ones_count = np.sum(window)
        zeros_count = len(window) - ones_count
        smoothed_data[i] = 1 if ones_count > zeros_count else 0

    return smoothed_data

def find_most_frequent_keyword(text, keyword_list):
    words = re.findall(r'\b\w+\b', text)
    keyword_freq = Counter(word for word in words if word in keyword_list)
    if keyword_freq:
        return keyword_freq.most_common(1)[0][0]
    return None

def remove_sentences_with_keywords(text, keyword_list):
    # Splitting the text into partial sentences
    partial_sentences = re.split(r',|\.', text)
    # Keeping sentences that do not contain any of the keywords
    return '. '.join(sentence for sentence in partial_sentences if not any(keyword in sentence for keyword in keyword_list))
def modify_text(preds, s_preds, keyword_list, text_list, window_size):
    if window_size % 2 != 0:
        window_size += 1
    window_size = int(window_size/2)
    modified_text_list = []

    for i, (pred, s_pred, text) in enumerate(zip(preds, s_preds, text_list)):
        # Condition when original label is 1 and new label is 0
        if s_pred == 0:
            if pred == 1:
                text = remove_sentences_with_keywords(text, keyword_list)
        # Condition when new label is 1
        elif s_pred == 1:
            # Extracting the window of text
            start_index = max(0, i - window_size)
            end_index = min(len(text_list), i + window_size + 1)
            window_text = ' '.join(text_list[start_index:end_index])
            most_freq_keyword = find_most_frequent_keyword(window_text, keyword_list)
            if most_freq_keyword:
                if most_freq_keyword.endswith('ing'):
                    addition = f'there are people {most_freq_keyword}'
                else:
                    addition = f'there is a {most_freq_keyword}'
                text = re.sub(r'\.', f'. {addition}', text, 1)
        modified_text_list.append(text)

    return modified_text_list


def evaluate(file_path, labels, output_file_path, save_modified):
    # Initial labels using keyword in rules
    text_lines = read_file(file_path)
    preds, keyword_list = clsuter_keyword(text_lines)

    # First-time EMA to smooth the preds with a more sensitive way
    ema_smoothed_data = pd.Series(preds).ewm(span=5, adjust=True).mean()
    threshold = ema_smoothed_data.mean()
    ema_preds = (ema_smoothed_data > threshold).astype(int)

    # Then majority_smooth to adjust the general trends
    s_preds = majority_smooth(ema_preds, window_size=20, edge_region_size=None)

    # Second-time EMA to get the auc score
    scores = pd.Series(s_preds).ewm(alpha = threshold, adjust=True).mean()

    if save_modified == True:
        modified_texts = modify_text(preds, s_preds, keyword_list, text_lines, window_size=20)
        with open(output_file_path, 'w') as file:
            for inner_list in modified_texts:
                file.write(inner_list + '\n')
    print(f"======================{file_path.split('/')[-1].split('.')[0]}========================>  ")
    print(f'Ori ACC: {accuracy_score(labels, preds)}')
    print(f'Ori Precision: {precision_score(labels, preds)}')
    print(f'Ori Recall: {recall_score(labels, preds)}')
    print(f'Soomth ACC: {accuracy_score(labels, s_preds)}')
    print(f'Soomth Precision: {precision_score(labels, s_preds)}')
    print(f'Soomth Recall: {recall_score(labels, s_preds)}')
    return preds, list(s_preds), list(scores)


def main():
    entries = os.listdir('SHTech/test_frame_description')
    all_preds = []
    all_labels = []
    all_spreds = []
    all_scores = []
    for item in entries:
        name = item.split('.')[0]
        input_file_path = f'SHTech/test_frame_description/{name}.txt'  # Path to your input text file
        output_file_path = f'SHTech/modified_test_frame_description/{name}.txt'  # Path for the new output text file
        if not os.path.exists(os.path.dirname(output_file_path)):
            os.makedirs(os.path.dirname(output_file_path))
        labels = pd.read_csv(f'SHTech/test_frame/{name}.csv').iloc[:, 1].tolist()
        preds, s_preds, scores = evaluate(input_file_path, labels, output_file_path, save_modified=True)
        all_labels += labels
        all_preds += preds
        all_spreds += s_preds
        all_scores += scores

    print(f"======================ALL DATA========================>  ")
    print(f'Ori ACC: {accuracy_score(all_labels, all_preds)}')
    print(f'Ori Precision: {precision_score(all_labels, all_preds)}')
    print(f'Ori Recall: {recall_score(all_labels, all_preds)}')
    print(f'Smooth ACC: {accuracy_score(all_labels, all_spreds)}')
    print(f'Smooth Precision: {precision_score(all_labels, all_spreds)}')
    print(f'Smooth Recall: {recall_score(all_labels, all_spreds)}')
    print(f'AUC: {roc_auc_score(all_labels, all_scores)}')

if __name__ == "__main__":
    main()


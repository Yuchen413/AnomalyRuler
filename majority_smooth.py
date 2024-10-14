from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from utils import *
import re
from collections import Counter
import argparse
from openai_api import keyword_extract


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

def anomaly_keywords(rule_path = 'rule/rule_SHTech.txt', regenerate_keyword = False):
    '''
    The below anomaly keywords are extracted once and used for the experiment in the paper,
    you can also extract from your rules with the below function.
    '''
    if regenerate_keyword == False:
        anomaly_from_rule = [
        "trolley",
        "cart",
        "luggage",
        "bicycle",
        "scooter",
        "vehicles",
        "vans",
        "accident",
        "running",
        "jumping",
        "riding",
        "skateboarding",
        "scooting",
        "lying",
        "falling",
        "bending",
        "fighting",
        "pushing",
        "loitering",
        "climbing",
        "tampering",
        "lingering"]
    else:
        anomaly_from_rule = keyword_extract(rule_path)
        print('Anomaly Keyword:', anomaly_from_rule)
    return anomaly_from_rule


def cluster_keyword(text_lines, anomaly_from_rule):
    preds = []
    anomaly_word = []
    for line in text_lines:
        found_anomaly = False
        for anomaly in anomaly_from_rule:
            if anomaly in line:
                found_anomaly = True
                anomaly_word.append(anomaly)
        preds.append(1 if found_anomaly else 0)
    return preds, anomaly_from_rule, anomaly_word

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

def ema_majority_smooth(ema_data, threshold, window_size=20, edge_region_size=None):
    # Adjust window size to be odd
    if window_size % 2 == 0:
        window_size += 1
    pad_size = window_size // 2

    # Determine edge region size if not specified
    if edge_region_size is None:
        edge_region_size = pad_size

    padded_data = np.pad(ema_data, pad_size, mode='edge')
    smoothed_data = np.zeros(len(ema_data), dtype=int)

    for i in range(len(ema_data)):
        # Apply different rule for edge regions
        if i < edge_region_size or i >= len(ema_data) - edge_region_size:
            # For edge data, consider only the previous pad_size values
            start = max(0, i - pad_size)
            end = i + 1  # Include the current point
            window = padded_data[start:end]
        else:
            # Regular processing for central part
            start = i
            end = i + window_size
            window = padded_data[start:end]

        # Apply majority rule based on threshold
        above_threshold_count = np.sum(window > threshold)
        below_threshold_count = len(window) - above_threshold_count
        smoothed_data[i] = 1 if above_threshold_count > below_threshold_count else 0

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
                    addition = f'{most_freq_keyword}'
                else:
                    addition = f'riding a {most_freq_keyword}'
                # text = re.sub(r'\.', f'. {addition}', text, 1)
                pattern = r"(the first person is)[^,]*"
                text = re.sub(pattern, r"\1 " + addition, text)

        modified_text_list.append(text)

    return modified_text_list


def evaluate(file_path, labels, output_file_path, save_modified,anomaly_from_rule):
    # Initial labels using keyword in rules
    text_lines = read_file(file_path)
    preds, keyword_list, _ = cluster_keyword(text_lines, anomaly_from_rule=anomaly_from_rule)

    # First-time EMA to smooth the preds with a more sensitive way
    ema_smoothed_data = pd.Series(preds).ewm(alpha = 0.33, adjust=True).mean()
    threshold = ema_smoothed_data.mean()
    s_preds = ema_majority_smooth(ema_smoothed_data, threshold, window_size=1)

    if threshold ==0:
        threshold += 0.0000001
    # Second-time EMA to get the auc score
    scores = pd.Series(s_preds).ewm(alpha = threshold, adjust=True).mean()

    if save_modified == True:
        modified_texts = modify_text(preds, s_preds, keyword_list, text_lines, window_size=1)
        with open(output_file_path, 'w') as file:
            for inner_list in modified_texts:
                file.write(inner_list + '\n')
    # print(f"======================{file_path.split('/')[-1].split('.')[0]}========================>  ")
    # print(f'Ori ACC: {accuracy_score(labels, preds)}')
    # print(f'Ori Precision: {precision_score(labels, preds)}')
    # print(f'Ori Recall: {recall_score(labels, preds)}')
    # print(f'Soomth ACC: {accuracy_score(labels, s_preds)}')
    # print(f'Soomth Precision: {precision_score(labels, s_preds)}')
    # print(f'Soomth Recall: {recall_score(labels, s_preds)}')
    return preds, list(s_preds), list(scores), list(ema_smoothed_data)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='SHTech',
                        choices=['SHTech', 'avenue', 'ped2', 'UBNormal'])
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    data_name = args.data
    entries = os.listdir(f'{data_name}/test_frame_description')
    all_preds = []
    all_labels = []
    all_spreds = []
    all_scores = []
    all_ori_scores = []
    anomaly_from_rule = anomaly_keywords(rule_path='rule/rule_SHTech.txt')
    for item in entries:
        name = item.split('.')[0]
        input_file_path = f'{data_name}/test_frame_description/{name}.txt'  # Path to your input text file
        output_file_path = f'{data_name}/modified_test_frame_description/{name}.txt'  # Path for the new output text file
        if not os.path.exists(os.path.dirname(output_file_path)):
            os.makedirs(os.path.dirname(output_file_path))
        labels = pd.read_csv(f'{data_name}/test_frame/{name}.csv').iloc[:, 1].tolist()
        preds, s_preds, scores, ori_scores = evaluate(input_file_path, labels, output_file_path, save_modified=False, anomaly_from_rule=anomaly_from_rule)
        all_labels += labels
        all_preds += preds
        all_spreds += s_preds
        all_scores += scores
        all_ori_scores += ori_scores

    print(f"======================ALL DATA========================>  ")
    print(f'Ori ACC: {accuracy_score(all_labels, all_preds)}')
    print(f'Ori Precision: {precision_score(all_labels, all_preds)}')
    print(f'Ori Recall: {recall_score(all_labels, all_preds)}')
    print(f'Ori AUC: {roc_auc_score(all_labels, all_ori_scores)}')
    print(f'Smooth ACC: {accuracy_score(all_labels, all_spreds)}')
    print(f'Smooth Precision: {precision_score(all_labels, all_spreds)}')
    print(f'Smooth Recall: {recall_score(all_labels, all_spreds)}')
    print(f'AUC: {roc_auc_score(all_labels, all_scores)}')

if __name__ == "__main__":
    main()


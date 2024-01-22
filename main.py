from llm import *
from openai_api import llm_induction_1
from utils import *


def main():
    # cog_model = AutoModelForCausalLM.from_pretrained(
    #     'THUDM/cogvlm-chat-hf',
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     device_map='auto',
    #     trust_remote_code=True
    # ).eval()

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                 device_map='auto').eval()

    # Rule generation:
    # objects_list = []
    # rule_list = []
    # t = 5
    # n = 1
    # for i in range(t):
    #     print('=====> Image Description:')
    #     selected_image_paths = random_select_data_without_copy(path='SHTech/train.csv', num=n, label=0)
    #     objects = cogvlm(model=cog_model, mode='chat', image_paths=selected_image_paths)
    #     objects_list.append(objects)
    #     rule_list.append(llm_induction_1(objects))
    # llm_rule_correction(rule_list, t)


    ## Deduction
    entries = os.listdir('SHTech/modified_test_frame_description')
    final_result = pd.DataFrame(columns=['file_name','labels','preds', 'scores', 'probs'])
    for item in entries:
        name = item.split('.')[0]
        labels = pd.read_csv(f'SHTech/test_frame/{name}.csv').iloc[:, 1].tolist()
        preds, scores, probs = mixtral_deduct(f'SHTech/modified_test_frame_description/{name}.txt',
                       'rule/rule_gpt4_both_5.txt', tokenizer, llm_model, labels=labels)
        final_result = final_result._append({'file_name': name,
                                            'labels': labels,
                                            'preds': preds,
                                            'scores': scores,
                                            'probs': probs},
                                           ignore_index=True)
    final_result.to_csv(f"results/SH/report_result.csv", index=False)
    # Initialize empty lists for each column
    labels_list = []
    preds_list = []
    scores_list = []
    probs_list = []

    # Iterate over each row in the DataFrame
    for index, row in final_result.iterrows():
        labels_list += row['labels']
        preds_list += row['preds']
        scores_list += row['scores']
        probs_list += row['probs']

    print(f"======================ALL DATA========================>  ")
    print(f'Frequency of Probabilities: {Counter(probs_list)}')
    print(f'Frequency of Anomaly scores: {Counter(scores_list)}')
    print(f'ACC: {accuracy_score(labels_list, preds_list)}')
    print(f'Precision: {precision_score(labels_list, preds_list)}')
    print(f'Recall: {recall_score(labels_list, preds_list)}')
    print(f'AUC: {roc_auc_score(labels_list, scores_list)}')

    with open('results/SH/report_result.txt', 'w') as file:
        file.write("======================ALL DATA========================>\n")
        file.write(f'Frequency of Probabilities: {Counter(probs_list)}\n')
        file.write(f'Frequency of Anomaly scores: {Counter(scores_list)}\n')
        file.write(f'ACC: {accuracy_score(labels_list, preds_list)}\n')
        file.write(f'Precision: {precision_score(labels_list, preds_list)}\n')
        file.write(f'Recall: {recall_score(labels_list, preds_list)}\n')
        file.write(f'AUC: {roc_auc_score(labels_list, scores_list)}\n')


if __name__ == "__main__":
    main()
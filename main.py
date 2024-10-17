from llm import *
from utils import *
# from image2text import cogvlm
import argparse
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='SHTech',
                        choices=['SHTech', 'avenue', 'ped2', 'UBNormal'])
    parser.add_argument('--induct', action='store_true')
    parser.add_argument('--deduct', action='store_true')
    parser.add_argument('--gpt_deduct_demo', action='store_true')
    parser.add_argument('--b', type = int, default=10)
    parser.add_argument('--bs', type = int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    data_name = args.data
    data_full_name = {'SHTech':'ShanghaiTech', 'avenue':'CUHK Avenue' , 'ped2': 'UCSD Ped2', 'UBNormal': 'UBNormal'}[data_name]
    print(args)

    if args.induct:

        cog_model = AutoModelForCausalLM.from_pretrained(
            'THUDM/cogvlm-chat-hf',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            # device_map='auto',
            trust_remote_code=True
        ).to(device).eval()

        #Rule generation:
        objects_list = []
        rule_list = []
        batch = args.b
        batch_size = args.bs
        for i in range(batch):
            print('=====> Image Description:')
            selected_image_paths = random_select_data_without_copy(path=f'{data_name}/train.csv', num=batch_size, label=0)
            print(selected_image_paths)
            objects = cogvlm(model=cog_model, mode='chat', image_paths=selected_image_paths)
            objects_list.append(objects)
            rule_list.append(gpt_induction(objects, data_full_name))
        gpt_rule_correction(rule_list, batch, data_full_name)

    if args.deduct:
        ## Deduction
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        llm_model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                     device_map='auto').eval()
        entries = os.listdir(f'{data_name}/modified_test_frame_description')
        final_result = pd.DataFrame(columns=['file_name','labels','preds', 'scores', 'probs'])
        for item in entries:
            print(item)
            name = item.split('.')[0]
            labels = pd.read_csv(f'{data_name}/test_frame/{name}.csv').iloc[:, 1].tolist()
            preds = mixtral_double_deduct(data_name,f'{data_name}/modified_test_frame_description/{name}.txt',
                                                  f'rule/rule_{data_name}.txt', tokenizer, llm_model, labels=labels)
            ### This is for 100 random test SHtech
            # labels = [1]*50 + [0]*50
            # preds, scores, probs = mixtral_deduct(f'SHTech/test_100_cogvlm_1_0.txt',
            #                'rule/rule_SHTech.txt', tokenizer, llm_model, labels=labels)

            scores = pd.Series(preds).ewm(alpha = 0.1, adjust=True).mean()
            final_result = final_result._append({'file_name': name,
                                                'labels': labels,
                                                'preds': preds,
                                                'scores': scores},
                                               ignore_index=True)
        final_result.to_csv(f"results/{data_name}/report_result.csv", index=False)
        # Initialize empty lists for each column
        labels_list = []
        preds_list = []
        scores_list = []

        # Iterate over each row in the DataFrame
        for index, row in final_result.iterrows():
            labels_list += row['labels']
            preds_list += row['preds']
            scores_list += row['scores']

        print(f"======================ALL DATA========================>  ")
        print(f'ACC: {accuracy_score(labels_list, preds_list)}')
        print(f'Precision: {precision_score(labels_list, preds_list)}')
        print(f'Recall: {recall_score(labels_list, preds_list)}')
        print(f'AUC: {roc_auc_score(labels_list, scores_list)}')

        with open(f'results/{data_name}/report_result.txt', 'w') as file:
            file.write("======================ALL DATA========================>\n")
            file.write(f'ACC: {accuracy_score(labels_list, preds_list)}\n')
            file.write(f'Precision: {precision_score(labels_list, preds_list)}\n')
            file.write(f'Recall: {recall_score(labels_list, preds_list)}\n')
            file.write(f'AUC: {roc_auc_score(labels_list, scores_list)}\n')

    if args.gpt_deduct_demo:
        entries = os.listdir(f'{data_name}/modified_test_frame_description')
        # entries[:1] try one file
        for item in entries[:1]:
            print(item)
            name = item.split('.')[0]
            gpt_double_deduction_demo(data_name, f'{data_name}/modified_test_frame_description/{name}.txt',
                                          f'rule/rule_{data_name}.txt')


if __name__ == "__main__":
    main()
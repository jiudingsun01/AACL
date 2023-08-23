import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def evaluate_re(df, label_verbalizer, pos_label_list, nota_eval=True, nota_eval_average='micro'):
    precision, recall, f1, support = precision_recall_fscore_support(y_pred=df['verbalized_pred'], y_true=df['verbalized_label'],
                                                labels=list(label_verbalizer), average='micro')
    return {'f1': f1, 'precision': precision, 'recall': recall}

csv_file_path = 'merged_and_verbalized_pred.csv'
data = pd.read_csv(csv_file_path)
label_verbalizer = set(data['verbalized_label'])
pos_label_list = {label for label in label_verbalizer if label != "no_relation"}

evaluation_results = evaluate_re(data, label_verbalizer, pos_label_list)
print(evaluation_results)
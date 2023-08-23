import csv
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# ------------Stage 1-------------
# 预测文件和标准答案文件
predict_file = 'Output_TACRED_QA'
golden_file = 'Golden_TACRED_QA'
data_file = ''

# 读取 output.txt 和 golden.txt 文件的数据
with open(predict_file, 'r') as output_file:
    output_data = output_file.readlines()

with open(golden_file, 'r') as golden_file:
    golden_data = golden_file.readlines()

# 创建一个列表来存储合并后的数据
merged_data = []

# 按行合并 output_data 和 golden_data
for idx, (output_line, golden_line) in enumerate(zip(output_data, golden_data)):
    output_line = output_line.strip().split('\t')[1].replace(".", "")
    golden_line = golden_line.strip().split('\t')[1]
    merged_data.append((idx, output_line, golden_line))

# 将合并后的数据保存为 CSV 文件
with open('merged_result.csv', 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Index', 'Output', 'Golden'])  # 写入标题行
    csv_writer.writerows(merged_data)  # 写入合并后的数据

# ------------Stage 2-------------

# 读取CSV文件
data = pd.read_csv(data_file)
merged_result_data = pd.read_csv('merged_result.csv')

if 'Index' in merged_result_data.columns:
    merged_result_data = merged_result_data.drop(columns=['Index'])

# 根据条件过滤数据
data['output_text_length'] = data['output_text'].apply(len)  # 添加一列记录output_text的长度
filtered_data = data[data['output_text_length'] == 1]  # 只保留长度为1的行

# 重置索引以确保对齐
merged_result_data.reset_index(drop=True, inplace=True)
data.reset_index(drop=True, inplace=True)

# 删除添加的长度列
filtered_data = filtered_data.drop(columns=['output_text_length'])

# 删除原文件中的binary_preds列
if 'binary_preds' in filtered_data.columns:
    filtered_data = filtered_data.drop(columns=['binary_preds'])

merged_data = pd.concat([merged_result_data, filtered_data], axis=1)

# 将处理后的数据保存到新的CSV文件中
# merged_data.to_csv('filtered_test.csv')
filtered_data.to_csv('filtered_data.csv', index=False)

# ------------Stage 3-------------

# 读取merged_result.csv的数据
merged_result_data = []
with open('merged_result.csv', 'r', newline='') as merged_result_file:
    reader = csv.reader(merged_result_file)
    for row in reader:
        merged_result_data.append(row)

# 读取filtered_data.csv的数据
filtered_data = []
with open('filtered_data.csv', 'r', newline='') as filter_data_file:
    reader = csv.reader(filter_data_file)
    for row in reader:
        filtered_data.append(row)

# 合并两个数据集并提取verbalized_pred
merged_data_with_verbalized_pred = []
for i in range(len(merged_result_data)):
    merged_row = merged_result_data[i] + filtered_data[i]
    if i != 0:
        label_verbalizer_dict = eval(merged_row[6])  # 将label_verbalizer列的字符串转换为字典
        output_key = merged_row[1]  # 获取Output列的值作为字典的键
        verbalized_pred = label_verbalizer_dict.get(output_key, "")  # 从字典中获取对应的值
        merged_row.append(verbalized_pred)  # 添加verbalized_pred到行末
        merged_data_with_verbalized_pred.append(merged_row)
    else:
        merged_row.append("verbalized_pred")
        merged_data_with_verbalized_pred.append(merged_row)

# 将合并后的数据保存到新的CSV文件中，包括新的列
with open('merged_and_verbalized_pred.csv', 'w', newline='') as merged_and_verbalized_pred_file:
    writer = csv.writer(merged_and_verbalized_pred_file)
    for row in merged_data_with_verbalized_pred:
        writer.writerow(row)

# ------------Stage 4-------------

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
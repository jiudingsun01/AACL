import csv

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
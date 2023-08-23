import pandas as pd

# 读取CSV文件
data = pd.read_csv('test.csv')
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
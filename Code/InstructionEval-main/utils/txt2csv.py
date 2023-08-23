import csv

# 预测文件和标准答案文件
predict_file = 'Output_TACRED_QA.txt'
golden_file = 'Golden_TACRED_QA.txt'

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


import pandas as pd
import csv
import os
import re

pattern = r".*vanilla\.csv"
files = [f for f in os.listdir('.') if os.path.isfile(
    os.path.join('.', f)) and re.match(pattern, f)]
# 一些可能的分隔符
delimiters = [' ', ',', '\t', ';', ':']
# files = ['semeval-text-davinci-002-vanilla.csv']
print(f"files={files}\n")
for f in files:
    df = pd.read_csv(os.path.join('.', f), sep='\t')
    print(f"{f}:{df.columns}")

    first_row = df.iloc[0].to_dict()
    # print(f"first_row={first_row}")
    if not 'final_input_prompts' in first_row:
        new_df = pd.read_csv(os.path.join('.', f), sep=',')
        new_first_row = new_df.iloc[0].to_dict()
        # print(f"new_first_row={new_first_row}")
        if 'final_input_prompts' in new_first_row:
            print(f"change to {f}")
            # new_df.to_csv(os.path.join('.', f), sep='\t', index=False)
        else:
            print(f"error")
            exit()

    # 用'\t'作为分隔符，重新写入文件
    # df.to_csv(os.path.join('.', f), sep='\t', index=False)

print("finish")
# # 使用检测到的分隔符读取文件
# df = pd.read_csv(filename, sep=sep)

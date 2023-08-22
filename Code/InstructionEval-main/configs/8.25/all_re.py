import os.path
import json
from configs.preprocessor import Preprocessor
import multiprocessing as mp
from datasets import Dataset
from configs.utils import load_BBL_file
import re
import pandas as pd
import ast
import sys


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):
    items, examples = [], []
    pattern = r".*vanilla\.csv"
    files = [f for f in os.listdir(input_dir) if os.path.isfile(
        os.path.join(input_dir, f)) and re.match(pattern, f)]
    print(f"{files}")
    # get final_input_prompts, all_indexes, correct_template_indexes
    tar_keys = ['input_text', 'output_text', 'label_space']
    src_keys = ['final_input_prompts', 'label', 'unique_labels']

    not_to_del_keys = ['final_input_prompts',
                       'label', 'unique_labels']
    for f in files:
        # print(f"file={f}")
        df = pd.read_csv(os.path.join(input_dir, f), sep='\t')
        for index, row in df.iterrows():
            row_dict = row.to_dict()

            # select keys to delete
            first_row = df.iloc[0].to_dict()
            del_keys = list(first_row.keys())
            for key in not_to_del_keys:
                if key in del_keys:
                    del_keys.remove(key)

            for i in range(len(tar_keys)):
                row_dict[tar_keys[i]] = row_dict.pop(src_keys[i])
            # convert to list
            row_dict['input_text'] = row_dict['input_text'].replace(
                "\n", " ").replace(r'`', r"'")
            row_dict['output_text'] = row_dict['output_text'].replace(
                "[", "").replace("]", "").replace(",", "").replace("'", "").replace(" ", "")
            row_dict['label_space'] = ast.literal_eval(row_dict['label_space'])
            # row_dict['input_text']=repr(row_dict['input_text'])
            for key in del_keys:
                del row_dict[key]

            # only single choose
            if len(row_dict['output_text']) == 1:
                items.append(row_dict)

    test_set = Dataset.from_list(items)

    for key in test_set[0].keys():
        print(
            f"key={key}\nvalue={test_set[0][key]}\ntype={type(test_set[0][key])}")
        print("\n")

    # exit()
    return test_set

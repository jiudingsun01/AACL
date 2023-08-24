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
    pattern = r".*davinci-003-vanilla\.csv"
    files = [f for f in os.listdir(input_dir) if os.path.isfile(
        os.path.join(input_dir, f)) and re.match(pattern, f)]
    print(f"{files}")
    # get final_input_prompts, all_indexes, correct_template_indexes
    tar_keys = ['input_text', 'output_text', 'label_space']
    src_keys = ['final_input_prompts', 'label', 'unique_labels']

    not_to_del_keys = ['final_input_prompts',
                       'label', 'unique_labels']

    labelSpace = set()

    for f in files:
        # print(f"file={f}")
        df = pd.read_csv(os.path.join(input_dir, f), sep='\t')
        for index, row in df.iterrows():
            row_dict = row.to_dict()

            # select keys to delete
            first_row = df.iloc[0].to_dict()
            del_keys = list(first_row.keys())

            for key in src_keys:
                del_keys.remove(key)

            for i in range(len(tar_keys)):
                row_dict[tar_keys[i]] = row_dict.pop(src_keys[i])
            # convert to list
            row_dict['input_text'] = row_dict['input_text'].replace(
                "\n", " ").replace(r'`', r"'")
            row_dict['output_text'] = row_dict['output_text'].replace(
                "[", "").replace("]", "").replace("'", "").replace(" ", "")
            row_dict['label_space'] = ast.literal_eval(row_dict['label_space'])

            for item in row_dict['label_space']:
                labelSpace.add(item)

            # row_dict['input_text']=repr(row_dict['input_text'])
            for key in del_keys:
                del row_dict[key]
            items.append(row_dict)

    generated_space = list()
    for item in labelSpace:
        generated_space.append(item)

    for row in items:
        row['label_space'] = generated_space

    if instruction is not None:
        for v in items:
            v['input_text'] = v['input_text'].replace(
                "Given a Sentence, and two Entities within the Sentence, classify the relationship between the two Entities based on the provided Sentence. All possible Relationships are listed below:", instruction, 1)

    test_set = Dataset.from_list(items)

    for key in test_set[0].keys():
        print(
            f"key={key}\nvalue={test_set[0][key]}\ntype={type(test_set[0][key])}")
        print("\n")

    # exit()
    return test_set

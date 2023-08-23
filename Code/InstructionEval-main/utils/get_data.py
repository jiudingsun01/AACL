import os
import re
import ast
import csv

input_dir = '.'
items = []
pattern = r".*QA4RE\.csv"
files = [f for f in os.listdir(input_dir) if os.path.isfile(
    os.path.join(input_dir, f)) and re.match(pattern, f)]

tar_keys = ['input_text', 'output_text', 'label_space']
src_keys = ['input_prompt', 'correct_template_indexes', 'all_indexes']
del_keys = ['Unnamed: 0', 'id', 'label', 'ent1_type', 'ent2_type', 'ent1', 'ent2',
            'sents', 'masked_sents', 'verbalized_label', 'final_input_prompts', 'index2rel',
            'test_ready_prompts', 'predictions', 'uncalibrated_predictions',
            'gpt3_output_predictions', 'cost', 'time',
            'rel_predictions']

for f in files:
    with open(os.path.join(input_dir, f), 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            row_dict = dict(row)
            for i in range(len(tar_keys)):
                row_dict[tar_keys[i]] = row_dict.pop(src_keys[i])
            row_dict['input_text'] = row_dict['input_text'].replace(
                "\n", " ").replace(r'`', r"'")
            row_dict['output_text'] = row_dict['output_text'].replace(
                "[", "").replace("]", "").replace(",", "").replace("'", "").replace(" ", "")
            row_dict['label_space'] = ast.literal_eval(row_dict['label_space'])
            for key in del_keys:
                if key in row_dict:
                    del row_dict[key]
            items.append(row_dict)

# Save to csv
with open('test.csv', 'w', newline='') as csvfile:
    fieldnames = items[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(items)
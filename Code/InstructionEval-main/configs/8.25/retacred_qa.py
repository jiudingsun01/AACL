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
    pattern = r".*QA4RE\.csv"
    files = [f for f in os.listdir(input_dir) if os.path.isfile(
        os.path.join(input_dir, f)) and re.match(pattern, f)]
    print(f"{files}")
    # get final_input_prompts, all_indexes, correct_template_indexes
    tar_keys = ['input_text', 'output_text', 'label_space']
    src_keys = ['input_prompt', 'correct_template_indexes', 'all_indexes']

    del_keys = ['Unnamed: 0', 'id', 'label', 'ent1_type', 'ent2_type', 'ent1', 'ent2', 'sents', 'masked_sents', 'verbalized_label',
                'index2rel', 'test_ready_prompts', 'predictions', 'uncalibrated_predictions', 'gpt3_output_predictions', 'final_input_prompts', 'cost', 'time', 'rel_predictions']
    for f in files:
        # print(f"file={f}")
        df = pd.read_csv(os.path.join(input_dir, f), sep='\t')
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            # print(f"{row_dict.keys()}")
            # exit()
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
            if len(row_dict['output_text']) == 1:
                items.append(row_dict)

    # print(f"\n{items[0]['output_text']}len={len(items[0]['output_text'])}\n")
    # items.append(dict())
    # # items[0]['input_text']="Answer the question about concepts combination. Specifically, you need to take contradictions, emergent properties, fanciful fictional combinations, homonyms, invented words, and surprising uncommon combinations into consideration. Concept: Impatient trees.  Question: Which of the following sentences best characterizes impatient trees? (A) Impatient trees sag when you make them wait. (B) Impatient trees are happy to wait. (C) Impatient trees prefer loamy soil. (D) Impatient trees take a long time to bloom."
    # items[0]['input_text']="Determine which option can be inferred from the given Sentence.  Sentence: Kercher 's body was found in a pool of blood with [her] throat slit on Nov. 2 , 2007 , in the bedroom of the house [she] shared with Knox . Options: A. [her] is the identity/pronoun of [she] B. [her] and [she] are the same person C. [her] is the parent of [she] D. [her] is the spouse of [she] E. [her] is the siblings of [she] F. [her] is the other family member of [she] G. [her] has the parent [she] H. [her] has no known relations to [she]  Which option can be inferred from the given Sentence? Option:"
    # items[0]['output_text']='AB'
    # items[0]['label_space']=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    # items=items[0:1]

    # items.append(dict())
    # # items[0]['input_text']="Answer the question about concepts combination. Specifically, you need to take contradictions, emergent properties, fanciful fictional combinations, homonyms, invented words, and surprising uncommon combinations into consideration. Concept: Impatient trees.  Question: Which of the following sentences best characterizes impatient trees? (A) Impatient trees sag when you make them wait. (B) Impatient trees are happy to wait. (C) Impatient trees prefer loamy soil. (D) Impatient trees take a long time to bloom."
    # items[0]['input_text']="Determine which option can be inferred from the given Sentence.  Sentence: And by the way , [One] of Alex Jones favorite '' alternative sources '' is the blatantly anti-semitic [American Free Press] , so I will have the gull to call him at the very least , ( coupled with his obvious hate for illegal immigrants ) a racist , hate monger . Options: A. [American Free Press] has the number of employees [One] B. [American Free Press] has the website [One] C. [American Free Press] has no known relations to [One]  Which option can be inferred from the given Sentence? Option:"
    # items[0]['output_text']='C'
    # items[0]['label_space']=['A', 'B', 'C']

    # print(items[0] == items[1])

    # print(f"items[0]={items[0]}\nlen={sys.getsizeof(items[0])}")
    # print(f"items[0]={items[1]}\nlen={sys.getsizeof(items[1])}")
    # items[1]['label_space'] = ['A', 'B', 'C', 'D']
    test_set = Dataset.from_list(items)

    for i in range(3):
        for key in test_set[i].keys():
            print(
                f"\n{i}:key={key}\nvalue={test_set[i][key]}\ntype={type(test_set[i][key])}\n")
            print("\n")

    print(items[0].keys())
    print(items[1].keys())
    print(items[2].keys())

    return test_set

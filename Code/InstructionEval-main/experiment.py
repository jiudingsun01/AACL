from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from metrics import LogitsMetric, OutputMetric
from pytorch_lightning import seed_everything
from inference_modules import LitDataModule
from lightning.fabric import Fabric
from tqdm import tqdm
import importlib
import argparse
import torch
import json
import time
import os
from sklearn.metrics import precision_recall_fscore_support

precisions_dict = {
    "fp16": (16, torch.float16),
    "fp32": (32, torch.float32),
    "bf16": ("bf16-mixed", torch.bfloat16),
}


DATASET2CONFIGS = {
    # data_dir, config_dir
    "MMLU_General": ("./data/MMLU", "./configs/MMLU/general.py",),
    "MMLU_Specific": ("./data/MMLU", "./configs/MMLU/specific.py",),
    "RETACRED_QA": (
        "../Data/RETACRED",
        "./configs/8.25/all_qa.py"
    ),
    "semeval_QA": (
        "../Data/semeval",
        "./configs/8.25/all_qa.py"
    ),
    "TACRED_QA": (
        "../Data/TACRED",
        "./configs/8.25/all_qa.py"
    ),
    "TACREV_QA": (
        "../Data/TACREV",
        "./configs/8.25/all_qa.py"
    ),
    "RETACRED_RE": (
        "../Data/RETACRED",
        "./configs/8.25/all_re.py"
    ),
    "semeval_RE": (
        "../Data/semeval",
        "./configs/8.25/all_re.py"
    ),
    "TACRED_RE": (
        "../Data/TACRED",
        "./configs/8.25/all_re.py"
    ),
    "TACREV_RE": (
        "../Data/TACREV",
        "./configs/8.25/all_re.py"
    ),
}


class Experiment:

    def __init__(self, model_name_or_path, devices, seed=42, precision="fp16"):

        if "alpaca" in model_name_or_path:
            ModelClass, TokenizerClass = AutoModelForCausalLM, LlamaTokenizer
        elif "vicuna" in model_name_or_path or "WizardLM" in model_name_or_path:
            ModelClass, TokenizerClass = AutoModelForCausalLM, AutoTokenizer
        else:
            ModelClass, TokenizerClass = AutoModelForSeq2SeqLM, AutoTokenizer

        seed_everything(seed)
        self.model_name_or_path = model_name_or_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_gpus = torch.cuda.device_count()
        print("Loading model...")
        if precision == "bf16":
            torch.set_float32_matmul_precision("high")
        fabric_precision, precision = precisions_dict[precision]

        self.model = ModelClass.from_pretrained(
            model_name_or_path, torch_dtype=precision)
        self.tokenizer = TokenizerClass.from_pretrained(model_name_or_path)
        strategy = "ddp" if len(devices) > 1 else "auto"
        self.fabric = Fabric(accelerator="cuda", devices=devices,
                             precision=fabric_precision, strategy=strategy)
        self.fabric.launch()

        self.model.eval()
        self.model = self.fabric.setup(self.model)

        self._tasks = []
        self.fabric.barrier()

    def add_tasks(
            self,
            input_dir: str,
            output_dir: str,
            config_dir: str,
            batch_size: str,
            instruction: str,
            shot_count: str,
            eval_by_logit: bool
    ) -> None:
        # if output_dir not in [task["output_dir"] for task in self._tasks]:
        self._tasks.append({
            "input_dir": input_dir,
            "output_dir": output_dir,
            "config_dir": config_dir,
            "batch_size": int(batch_size),
            "instruction": instruction,
            "shot_count": int(shot_count),
            "eval_by_logit": eval_by_logit
        })

    def add_tasks_by_name(self,
                          task_name: str,
                          output_dir: str,
                          batch_size: str,
                          instruction: str,
                          shot_count: str,
                          eval_by_logit: bool
                          ) -> None:
        if task_name not in DATASET2CONFIGS.keys():
            raise ValueError("Task name not found")
        else:
            input_dir, config_dir = DATASET2CONFIGS[task_name]
            self.add_tasks(input_dir, output_dir, config_dir,
                           batch_size, instruction, shot_count, eval_by_logit)

    def inference(self):

        for i, task in enumerate(self._tasks):

            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

            print("Inference on Task {}/{}...".format(i, len(self._tasks)))
            input_dir, output_dir, config_dir, batch_size, instruction, shot_count, eval_by_logits = list(
                task.values())
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            # try:
            spec = importlib.util.spec_from_file_location("config", config_dir)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            print("Loading datasets")
            test_set = config.load_data(
                input_dir, instruction, shot_count, eval_by_logits, self.tokenizer)
            print(f"1.test_set={test_set[0]}")
            example = test_set[0]["input_text"]
            """except Exception:
                print(instruction)
                print("Encountered Exception while loading config file from {}; continue...".format(config_dir))
                continue"""

            data_module = LitDataModule(batch_size, test_set, self.tokenizer)
            test_set = data_module.test_dataloader()
            test_set = self.fabric.setup_dataloaders(test_set)
            self.fabric.barrier()

            metric = LogitsMetric(
                self.fabric) if eval_by_logits else OutputMetric()

            all_classes, all_gold_classes = [], []
            all_pred, all_gold = [], []
            idx = 0
            with torch.no_grad():
                for batch in tqdm(test_set):

                    input_ids, attention_mask, labels, label_cls, label_spaces_ids, sample_to = batch.values()
                    choose_AB = False
                    if self.tokenizer.decode(labels.tolist()[0])[0] == '?':
                        choose_AB = True
                    # print(f"attention_mask={attention_mask}")
                    # exit()
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "max_new_tokens": 32
                    }

                    outputs = self.model.generate(
                        **inputs, return_dict_in_generate=True, output_scores=True)
                    scores = outputs.scores
                    logits = torch.stack(scores, dim=1)
                    if eval_by_logits:
                        classes = metric.classify(
                            logits, label_spaces_ids, sample_to)
                        # print("{logits}")
                        all_classes.extend(classes.cpu().numpy())
                        all_gold_classes.extend(label_cls.cpu().numpy())

                    pred_ids = outputs.sequences
                    all_pred.extend(pred_ids.cpu().numpy())
                    all_gold.extend(labels.cpu().numpy())

            assert len(all_pred) == len(all_gold) and len(
                all_classes) == len(all_gold_classes)
            preds = self.tokenizer.batch_decode(
                all_pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            golds = self.tokenizer.batch_decode(
                all_gold, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for i in range(len(preds)):
                preds[i] = preds[i].replace(".", "")
                # print(len(preds[i]))
                if golds[i] == '?' and (preds[i][0] == 'A' or preds[i][0] == 'B'):
                    preds[i] = preds[i].replace(preds[i][0], "?")
                #     print(f"change:{preds[i]}")
                # if i < 5:
                #     print(preds)
                # else:
                #     exit()

            label_verbalizer = set(golds)
            all_labels = {
                label for label in label_verbalizer if label != "no_relation"}
            precision, recall, f1, support = precision_recall_fscore_support(y_pred=preds, y_true=golds,
                                                                             labels=list(all_labels), average='micro')
            dataset_folder_name = os.path.basename(os.path.normpath(input_dir))
            config_file_dir = os.path.basename(os.path.normpath(config_dir))
            output_file = "output_{}_{}_{}.txt".format(dataset_folder_name.split('/')[-1], config_file_dir.split('.')[-2][-2:], self.model_name_or_path.split(
                '/')[-1])
            golden_file = "golden_{}_{}_{}.txt".format(dataset_folder_name.split('/')[-1], config_file_dir.split('.')[-2][-2:], self.model_name_or_path.split(
                '/')[-1])

            with open(os.path.join(output_dir, "result.txt"), "a") as r:
                r.write("{}_{}_{}\nPrecision: {}\tRecall: {}\tF1 Score: {}\n\n\n".format(dataset_folder_name.split('/')[-1], config_file_dir.split('.')[-2][-2:], self.model_name_or_path.split(
                    '/')[-1], precision, recall, f1))
            # 写文件
            with open(os.path.join(output_dir, output_file), "w") as f:

                for n, pred in enumerate(preds):
                    f.write(str(n) + "\t" + pred + "\n")
                f.close()
            f.close()
            with open(os.path.join(output_dir, golden_file), "w") as f:
                for n, gold in enumerate(golds):
                    f.write(str(n) + "\t" + gold + "\n")
                f.close()
        print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--tasks_dir', required=True)
    parser.add_argument('--precision', default="fp16",
                        choices=["fp16", "fp32", "bf16"], type=str)
    parser.add_argument('--devices', default=[0], type=int, nargs="+")

    args = parser.parse_args()

    experiment = Experiment(args.model_name_or_path,
                            devices=args.devices, precision=args.precision)
    print("tasks_dir is: {}".format(args.tasks_dir))
    tasks_args = json.load(open(args.tasks_dir, "r"))
    for args in tasks_args:
        experiment.add_tasks(**args)

    experiment.inference()


if __name__ == "__main__":
    main()

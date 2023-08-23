from experiment import Experiment


if __name__ == "__main__":
  # Flan-T5-3B Alpaca-7B Vicuna-7B T0++ 11B Flan-T5-11B WizardLM-13B
    # model_name = "google/flan-t5-xl"
    model_name = "ziqingyang/chinese-alpaca-2-7b"
    # model_name = "lmsys/vicuna-7b-v1.5"
    # model_name = "bigscience/T0pp"
    # model_name = "google/flan-t5-xxl"
    # model_name = "google/flan-t5-xxl"
    experiment = Experiment(model_name, devices=[0], precision="bf16")

    # Instructions
    #
    # QA:
    # 1. Origin
    # 2. You are given a multi-choice question answering task, given the question and mutliple answer options, select the option that can be reasonably derived from the sentence's context and content.
    # 3. Select the alternative that can be logically extrapolated from the information given in the sentence.
    # 4. Pinpoint the correct choice that can be inferred through logical analysis of the given statement.
    Ins_QA = [None, "You are given a multi-choice question answering task, given the question and mutliple answer options, select the option that can be reasonably derived from the sentence's context and content.",
              "Select the alternative that can be logically extrapolated from the information given in the sentence.", "Pinpoint the correct choice that can be inferred through logical analysis of the given statement."]
    # RE:
    # 1. Origin
    # 2. Involving a given sentence and two entities within it, your goal is to identify the relationship between the two entities using the provided sentence. Choose from the set of relationships provided below.
    # 3. You'll be working with a sentence that contains two entities. Your objective is to categorize the relationship between these entities using the provided sentence. Refer to the list of potential relationships provided.
    # 4. The challenge is to establish the relationship between two entities within a given sentence. Your task is to classify this relationship based on the sentence provided. Explore the list of possible relationships below.
    Ins_RE = [None, "Involving a given sentence and two entities within it, your goal is to identify the relationship between the two entities using the provided sentence. Choose from the set of relationships provided below.",
              "You'll be working with a sentence that contains two entities. Your objective is to categorize the relationship between these entities using the provided sentence. Refer to the list of potential relationships provided.", "The challenge is to establish the relationship between two entities within a given sentence. Your task is to classify this relationship based on the sentence provided. Explore the list of possible relationships below."]
    for i in range(len(Ins_QA)):
        experiment.add_tasks_by_name(
            task_name="RETACRED_QA",
            output_dir="./chinese-alpaca-2-7b/",
            batch_size="1",
            instruction=Ins_QA[i],
            shot_count="0",
            eval_by_logit=True,
        )
        experiment.add_tasks_by_name(
            task_name="semeval_QA",
            output_dir="./chinese-alpaca-2-7b/",
            batch_size="1",
            instruction=Ins_QA[i],
            shot_count="0",
            eval_by_logit=True,
        )
        experiment.add_tasks_by_name(
            task_name="TACRED_QA",
            output_dir="./chinese-alpaca-2-7b/",
            batch_size="1",
            instruction=Ins_QA[i],
            shot_count="0",
            eval_by_logit=True,
        )
        experiment.add_tasks_by_name(
            task_name="TACREV_QA",
            output_dir="./chinese-alpaca-2-7b/",
            batch_size="1",
            instruction=Ins_QA[i],
            shot_count="0",
            eval_by_logit=True,
        )

    for i in range(len(Ins_RE)):
        experiment.add_tasks_by_name(
            task_name="RETACRED_RE",
            output_dir="./chinese-alpaca-2-7b/",
            batch_size="1",
            instruction=Ins_RE[i],
            shot_count="0",
            eval_by_logit=True,
        )
        experiment.add_tasks_by_name(
            task_name="semeval_RE",
            output_dir="./chinese-alpaca-2-7b/",
            batch_size="1",
            instruction=Ins_RE[i],
            shot_count="0",
            eval_by_logit=True,
        )
        experiment.add_tasks_by_name(
            task_name="TACRED_RE",
            output_dir="./chinese-alpaca-2-7b/",
            batch_size="1",
            instruction=Ins_RE[i],
            shot_count="0",
            eval_by_logit=True,
        )
        experiment.add_tasks_by_name(
            task_name="TACREV_RE",
            output_dir="./chinese-alpaca-2-7b/",
            batch_size="1",
            instruction=Ins_RE[i],
            shot_count="0",
            eval_by_logit=True,
        )
    experiment.inference()

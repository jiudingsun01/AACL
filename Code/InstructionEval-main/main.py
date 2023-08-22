from experiment import Experiment


if __name__ == "__main__":
    model_name = "google/flan-t5-xl"
    # experiment = Experiment("google/flan-t5-xl", devices=[0], precision="bf16")
    experiment = Experiment(model_name, devices=[0], precision="bf16")
    # experiment.add_tasks_by_name(
    #     task_name="Conceptual_Combinations_Adversarial",
    #     output_dir="./test/",
    #     batch_size="8",
    #     instruction="BBL/Unobserved/4",
    #     shot_count="0",
    #     eval_by_logit=True,
    # )

    # instruction need to add '.' in the end to identify the end of a sentence
    # experiment.add_tasks_by_name(
    #     task_name="RETACRED_QA",
    #     output_dir="./test/",
    #     batch_size="1",
    #     instruction="test.",
    #     shot_count="0",
    #     eval_by_logit=True,
    # )
    # experiment.add_tasks_by_name(
    #     task_name="semeval_QA",
    #     output_dir="./test/",
    #     batch_size="1",
    #     instruction=None,
    #     shot_count="0",
    #     eval_by_logit=True,
    # )
    # experiment.add_tasks_by_name(
    #     task_name="TACRED_QA",
    #     output_dir="./test/",
    #     batch_size="1",
    #     instruction=None,
    #     shot_count="0",
    #     eval_by_logit=True,
    # )
    experiment.add_tasks_by_name(
        task_name="TACREV_QA",
        output_dir="./test/",
        batch_size="1",
        instruction=None,
        shot_count="0",
        eval_by_logit=True,
    )
    experiment.inference()

Experiments:
1. 验证模型对于Unseen tasks(RE)和Seen tasks(QA)的应对paraphrase的鲁棒性是否有区别，把datasets里的QA和RE拿出来分别修改instructions(paraphrase)在所有模型上跑一遍，对比性能变化
2. 分别做zero-shot和few-shot，观察in context learning是否能改善模型鲁棒性
3. 用Flan-T5验证模型的scaling(small～XXL)能否改善模型鲁棒性


References:
https://arxiv.org/pdf/2109.01652.pdf Fine-tuned Language Models Are Zero-Shot Learners

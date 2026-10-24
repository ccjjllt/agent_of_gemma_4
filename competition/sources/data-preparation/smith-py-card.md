---
language:
- en
license: mit
size_categories:
- 10K<n<100K
task_categories:
- text-generation
dataset_info:
  features:
  - name: instance_id
    dtype: string
  - name: patch
    dtype: string
  - name: FAIL_TO_PASS
    list: string
  - name: PASS_TO_PASS
    list: string
  - name: image_name
    dtype: string
  - name: repo
    dtype: string
  - name: problem_statement
    dtype: string
  splits:
  - name: train
    num_bytes: 5085837247
    num_examples: 50908
  download_size: 220645767
  dataset_size: 5085837247
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
tags:
- code
- agents
- software-engineering
---

<div align="center">
  <a href="https://swesmith.com/">
    <img src="https://avatars.githubusercontent.com/u/189315905?s=200&v=4" alt="Logo" width="200">
    <h1 align="center">SWE-smith Dataset</h1>
  </a>
</div>
<p align="center">
<a href="https://github.com/SWE-bench/SWE-smith">Code</a>
•
<a href="https://huggingface.co/papers/2504.21798">Paper</a>
•
<a href="https://swesmith.com/">Site</a>
</p>

As of 12/14/2025, **SWE-smith: Python** contains **50908** task instances from **131** GitHub repositories

The SWE-smith Dataset is the largest open source dataset for training software engineering agents.

All SWE-smith task instances come with an executable environment.
To learn more about how to use this dataset to train Language Models for Software Engineering, please refer to the [documentation](https://swesmith.com/docs).
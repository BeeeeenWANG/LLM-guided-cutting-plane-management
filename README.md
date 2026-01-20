# LLM-guided-cutting-plane-management

## Project Description
This repository contains the official implementation for the paper: **LLM-guided Cutting-plane Management for Mixed-integer Linear Programming**.  
We propose a novel method for adaptive separator configuration and cut selection in MILP solving using large language models (LLMs). Our approach leverages instance-specific structural characteristics to activate effective separators and select high-quality cuts, achieving significant performance improvements over heuristic-based and existing learning-based methods across five MILP problem classes.

## Requirements
- Python rely on
  - Python 3.8
  - tqdm
  - requests
- Solver dependencies
  - SCIP 8.0.0
  - PySCIPOpt 4.1.0 (DIY), please refer to https://gitee.com/wang-zhihai/py-scipopt_-hem_-iclr2023
- API Application
  - We have applied for large language model APIs on https://cloud.siliconflow.cn/ and https://openrouter.ai/. Please apply before running the code.  
 
## Running the Code
  ```bash
python main.py --dataset setcover --mode ai
```


## Datasets
The datasets can be found in https://drive.google.com/drive/folders/1LXLZ8vq3L7v00XH-Tx3U6hiTJ79sCzxY?usp=sharing

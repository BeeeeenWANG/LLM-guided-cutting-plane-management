from pyscipopt import SCIP_RESULT, Cutsel
# from test_api import llmClient
import logging
import re
import time
import random
import numpy as np
import requests
import json
import sys
import os
import pdb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import advanced_cut_feature_generator, compute_normalized_violation_scores

logger = logging.getLogger(__name__)

# 国内API接口
# API_KEY = "sk-aacsjwbslmzarphddbhhupmyxtmycrxnidyinvxoleybgcld"
API_KEY = "sk-ctoowrguzjggztifabtbbonkhjtyprzesyrsuwdxyoffpfyz"
MODEL_NAME = "Qwen/Qwen3-8B"  # "Qwen/Qwen3-30B-A3B"#"Qwen/Qwen3-8B"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 国外API接口
API_KEY_ABO = "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"
MODEL_NAME_ABO = "openai/gpt-4o"  # "google/gemini-2.5-flash"
API_URL_ABO = "https://openrouter.ai/api/v1/chat/completions"
global_llm_time, global_cut_num = [], []


class CutSelAgent(Cutsel):

    def __init__(self, scip_model, api_key, api_url, model_name, sel_cuts_percent=0.05, max_calls=1):
        super().__init__()
        self.scip_model = scip_model
        self.sel_cuts_percent = sel_cuts_percent
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        # 新增缓存控制变量
        self.llm_response_cache = None
        self.data = {}
        self.max_calls = max_calls
        self.call_count = 0  # 调用计数器

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        # 限制选择次数
        if self.call_count >= self.max_calls:
            # if not root:
            num = len(cuts)
            return {
                'cuts': cuts[:num],
                'nselectedcuts': num,
                'result': SCIP_RESULT.SUCCESS
            }
        self.call_count += 1
        logger.debug(f"Cut selection called ({self.call_count} times)")

        # 基础校验
        num_cuts = len(cuts)
        global_cut_num.append(num_cuts)
        if num_cuts <= 1:
            return self._base_return(cuts, 1)
        # 首次调用LLM
        start_time = time.time()
        try:
            # 特征生成
            raw_features = advanced_cut_feature_generator(self.scip_model, cuts)
            n_select = int(num_cuts * self.sel_cuts_percent)
            # 构建prompt
            prompt = self._build_prompt(raw_features, n_select)
            logger.debug(f"Prompt sent to LLM:\n{prompt}")

            # 调用国内大模型
            # gpt = llmClient(api_key=self.api_key,model_name=self.model_name ,api_url=self.api_url)
            # 调用国外大模型
            gpt = llmClient(api_key=self.api_key, model_name=self.model_name, api_url=self.api_url)
            self.llm_response_cache = gpt.get_content(gpt.getResponse(prompt))
            # print(self.llm_response_cache)
            llm_time = time.time() - start_time
            global_llm_time.append(llm_time)  # 记录本次LLM调用用时
            # print(f'llm-cut本次运行时间为: {llm_time:.4f}秒', f'cut-time列表内容为: {global_llm_time}')
            print(f'调用llm-cut平均运行时间为: {np.mean(global_llm_time):.4f}秒',
                  f'产生的cuts个数列表: {global_cut_num}')
            logger.info("LLM调用成功")

            # 解析结果
            selected_idx = self._parse_answer(self.llm_response_cache, num_cuts)
            logger.debug(f"Raw selected indices: {selected_idx}")

            # 结果校验
            valid_idx = self._validate_selection(selected_idx, num_cuts, maxnselectedcuts)

            # 构建返回结构
            sorted_cuts, remaining = self._sort_cuts(cuts, valid_idx)

            # 记录数据
            self.data = {
                "raw_features": raw_features,
                "selected_idx": valid_idx,
                "processing_time": time.time() - start_time,
                "llm_response": self.llm_response_cache,
                "llm_time": llm_time  # 记录本次LLM调用用时
            }
            return {
                'cuts': sorted_cuts + remaining,
                'nselectedcuts': len(valid_idx),
                'result': SCIP_RESULT.SUCCESS
            }

        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            print('Calling LLM failed.')
            return self._fallback_selection(cuts, maxnselectedcuts)

    def _build_prompt(self, features, n):
        # FEATURE_DESC = [
        #     "与目标函数的平行度（越大越好）",
        #     "有效性指标（越大越好）",
        #     "稀疏度指标（越小越好）",
        #     "整数支持度（越大越好）",
        #     "标准化违反度（越大越好）"
        # ]

        FEATURE_DESC = [
            "Parallelism with the objective function (the larger, the better)",
            "Efficacy metric (the larger, the better)",
            "Support metric (the smaller, the better)",
            "Integer support (the larger, the better)",
            "Normalized violation (the larger, the better)"
        ]

        # prompt = """
        # You are a mathematical optimization expert. Please analyze the following cutting-plane features and recommend the most effective subset:

        # Feature Description:

        # """

        # """
        # 你是一个数学优化专家，请分析以下切割平面特征，推荐最有效的子集：

        # 特征说明：
        # """
        # for i, desc in enumerate(FEATURE_DESC):
        #     prompt += f"{i}. {desc}\n"

        # # prompt += "\n当前切割特征值（原始数据）：\n"
        # # for idx, feat in enumerate(features):
        # #     values = ', '.join([f"{v:.4f}" for v in feat])
        # #     prompt += f"切割{idx}: [{values}]\n"

        # prompt += "\nCurrent cuts features：\n"
        # for idx, feat in enumerate(features):
        #     values = ', '.join([f"{v:.4f}" for v in feat])
        #     prompt += f"Cut{idx}: [{values}]\n"

        # prompt += """

        # Please make recommendations according to the following rules:
        # Give priority to cuts that are most aligned with the objective function direction which means the larger objective parallelism.
        # Next, consider cuts with a high degree of violation.
        # Finally, consider support and integer support.

        # Only output the indices of the selected cuts. Do not include any analysis or explanation. Please output strictly in the following format:
        # Recommended cut order: 0,2,5,6,8
        # """

        # """
        # 请按以下规则推荐：
        # 1. 优先选择与目标函数方向最接近的切割
        # 2. 其次考虑违反程度大的切割
        # 3. 最后考虑稀疏性和整数支持度

        # 请你只需要输出所选择切割的序号即可，不需要反馈给我分析过程，请用一段话严格按照以下示例格式输出
        # 输出格式：推荐切割顺序：0,2,5,6,8
        # """

        prompt = f"""
        You are an expert in mathematical optimization and mixed-integer programming. Your task is to select the most effective subset of cutting planes based on their features.

        Cut selection significantly impacts solver performance, including the following key metrics:
        - Solving Time (to be minimized)
        - Primal-Dual Integral (to be minimized)
        - Number of Nodes (to be minimized)
        - Primal-Dual Gap (to be minimized)

        Your objective is to select a subset of cuts such that these performance metrics are jointly optimized. You need to reason based on the provided cut features.

        You must select no more than {n} cuts.

        Cut Feature Description:
        """
        for i, desc in enumerate(FEATURE_DESC):
            prompt += f"{i}. {desc}\n"

        prompt += "\nCut Feature Vectors:\n"
        for idx, feat in enumerate(features):
            values = ', '.join([f"{v:.4f}" for v in feat])
            prompt += f"Cut{idx}: [{values}]\n"

        prompt += f"""

        Selection Strategy:
        - Prefer cuts that tend to minimize the four metrics listed above.
        - Consider trade-offs among different features and prioritize cuts that balance violation, sparsity, and objective alignment.
        - You must select at most {n} cuts in total.

        Please only output the indices of the selected cuts in a single line, using the following format:
        Recommended cut order: 0,2,5,6,8
        Do not provide any explanation or extra output.
        """

        return prompt

    def _parse_answer(self, answer, max_idx):
        """增强型解析方法"""
        try:
            numbers = []
            for num_str in re.findall(r'\d+', answer):  # 仅提取目标部分的数字
                try:
                    num = int(num_str)
                    numbers.append(num)
                except ValueError:
                    continue
            seen = set()
            valid = []
            for n in numbers:
                if 0 <= n < max_idx and n not in seen:
                    valid.append(n)
                    seen.add(n)

            max_select = max(1, int(max_idx * 0.5))
            # print("所选择切割集为：", valid[:max_select])
            return valid[:max_select]
        except Exception as e:
            logger.warning(f"解析失败: {str(e)}, 使用降级策略")
            return self._fallback_indices(max_idx)

    def _apply_cached_selection(self, cuts, maxnselectedcuts):
        """应用缓存的LLM选择结果"""
        try:
            num_cuts = len(cuts)
            selected = self._parse_answer(self.llm_response_cache, num_cuts)
            valid = self._validate_selection(selected, num_cuts, maxnselectedcuts)
            sorted_cuts, remaining = self._sort_cuts(cuts, valid)
            return {
                'cuts': sorted_cuts + remaining,
                'nselectedcuts': len(valid),
                'result': SCIP_RESULT.SUCCESS
            }
        except Exception as e:
            logger.error(f"缓存应用失败: {str(e)}")
            return self._fallback_selection(cuts, maxnselectedcuts)

    def _sort_cuts(self, cuts, indices):
        """根据选择索引排序切割"""
        selected = [cuts[i] for i in indices]
        remaining = [c for i, c in enumerate(cuts) if i not in indices]
        return selected, remaining

    def _validate_selection(self, indices, max_idx, max_allowed):
        """校验选择结果有效性"""
        indices = list(set(indices))  # 去重
        valid = [i for i in indices if 0 <= i < max_idx]
        return valid[:min(len(valid), max_allowed)]

    def _fallback_selection(self, cuts, maxnselectedcuts):
        """降级选择策略"""
        num = min(len(cuts), maxnselectedcuts)
        return {
            'cuts': cuts[:num],
            'nselectedcuts': num,
            'result': SCIP_RESULT.SUCCESS
        }

    def _fallback_indices(self, max_idx):
        """生成降级选择的索引"""
        from random import sample
        k = int(max_idx * self.sel_cuts_percent)
        return sample(range(max_idx), k=k) if max_idx > 0 else []

    def _base_return(self, cuts, n):
        """基础返回结构"""
        return {
            'cuts': cuts[:n],
            'nselectedcuts': n,
            'result': SCIP_RESULT.SUCCESS
        }

    def free_problem(self):
        """重置缓存状态"""
        self.has_called_llm = False
        self.llm_response_cache = None
        # clear_global_llm_times()  # 重置LLM调用用时记录
        self.scip_model.freeProb()


class llmClient():

    def __init__(self,
                 api_key,
                 model_name="Qwen/Qwen3-30B-A3B",
                 api_url="https://api.siliconflow.cn/v1/chat/completions",
                 stream=False,
                 max_tokens=4096,
                 enable_thinking=False,
                 thinking_budget=4096,
                 min_p=0.05,
                 temperature=0.5,
                 top_p=0.7,
                 top_k=50,
                 frequency_penalty=0.8,
                 n=1
                 ):

        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url
        self.stream = stream
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.min_p = min_p
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.frequency_penalty = frequency_penalty
        self.n = n

    def getResponse(self, prompt):

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "stream": self.stream,
            "max_tokens": self.max_tokens,
            "enable_thinking": self.enable_thinking,
            "thinking_budget": self.thinking_budget,
            "min_p": self.min_p,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "n": self.n,
            "stop": [],
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()  # 自动触发HTTP错误异常
            return response.text
        except Exception as e:
            print(f"API请求失败: {str(e)}")
            return None

    def get_content(self, response_text):
        if not response_text or not isinstance(response_text, str):
            print(f"无效响应类型: {type(response_text)}")
            return None

        try:
            response_json = json.loads(response_text)

            # 层级校验
            if 'choices' not in response_json:
                print(f"响应缺少choices字段，完整响应：{response_text}")
                return None

            if not isinstance(response_json['choices'], list) or len(response_json['choices']) == 0:
                print(f"choices格式异常，完整响应：{response_text}")
                return None

            message = response_json['choices'][0].get('message', {})
            return message.get('content')

        except json.JSONDecodeError:
            print(f"响应非JSON格式，原始内容：{response_text}")
            return None
        except Exception as e:
            print(f"解析异常: {str(e)}")
            return None

    def safe_get_content(self, prompt):
        raw_response = self.getResponse(prompt)
        if raw_response is None:
            return "请求失败，请检查网络和API配置"
        return self.get_content(raw_response) or "内容解析失败"


import requests
import json

import os
from openai import OpenAI


# class QwenClient:
#     def __init__(self,
#                  api_key="sk-b300fa60ac03467aa2383c4459a89726",
#                  model_name="qwen2.5-math-72b-instruct",
#                  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#                  system_prompt="You are a helpful assistant.",
#                  temperature=0.7,
#                  top_p=0.8,
#                  max_tokens=2048,
#                  enable_thinking=False
#                  ):
#         self.api_key = api_key
#         self.model_name = model_name
#         self.base_url = base_url
#         self.system_prompt = system_prompt
#         self.temperature = temperature
#         self.top_p = top_p
#         self.max_tokens = max_tokens
#         self.enable_thinking = enable_thinking

#         self.client = OpenAI(
#             api_key=self.api_key,
#             base_url=self.base_url
#         )

#     def getResponse(self, prompt):
#         try:
#             completion = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[
#                     {"role": "system", "content": self.system_prompt},
#                     {"role": "user", "content": prompt},
#                 ],
#                 temperature=self.temperature,
#                 top_p=self.top_p,
#                 max_tokens=self.max_tokens,
#                 extra_body={"enable_thinking": self.enable_thinking}
#             )
#             return completion.choices[0].message.content
#         except Exception as e:
#             print(f"[ERROR] 请求失败：{e}")
#             return None

#     def safe_get_content(self, prompt):
#         content = self.getResponse(prompt)
#         return content or "生成失败，请检查API Key或模型设置"


# class QwenClient:
#     #使用QWEN官方提供的链接api
#     def __init__(self,
#                  api_key='sk-b300fa60ac03467aa2383c4459a89726',
#                  model_name="qwen-max",
#                  api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#                  temperature=0.7,
#                  top_p=0.8,
#                  max_tokens=2048):

#         self.api_key = api_key
#         self.model_name = model_name
#         self.api_url = api_url
#         self.temperature = temperature
#         self.top_p = top_p
#         self.max_tokens = max_tokens

#     def getResponse(self, prompt):
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }

#         payload = {
#             "model": self.model_name,
#             "input": {
#                 "messages": [
#                     {
#                         "role": "user",
#                         "content": prompt
#                     }
#                 ]
#             },
#             "parameters": {
#                 "temperature": self.temperature,
#                 "top_p": self.top_p,
#                 "max_tokens": self.max_tokens
#             }
#         }

#         try:
#             response = requests.post(self.api_url, json=payload, headers=headers)
#             response.raise_for_status()
#             return response.json()
#         except Exception as e:
#             print(f"API 请求失败: {str(e)}")
#             return None

#     def get_content(self, response_json):
#         if not response_json or "output" not in response_json:
#             print("无效响应")
#             return None

#         try:
#             return response_json["output"]["text"]
#         except KeyError:
#             print("响应格式异常")
#             return None

#     def safe_get_content(self, prompt):
#         raw_response = self.getResponse(prompt)
#         if raw_response is None:
#             return "请求失败，请检查网络或API配置"
#         return self.get_content(raw_response) or "内容解析失败"


class CutSelRandom(Cutsel):
    def __init__(self, sel_cuts_percent=0.05):
        self.sel_cuts_percent = sel_cuts_percent

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        """
        Randomly selects a percentage of cuts.
        """
        num_total = len(cuts)
        if num_total == 0:
            return {'cuts': [], 'nselectedcuts': 0, 'result': SCIP_RESULT.SUCCESS}

        # 随机打乱索引
        indices = list(range(num_total))
        random.shuffle(indices)

        # 选择前若干比例
        n_select = max(1, int(self.sel_cuts_percent * num_total))
        n_select = min(n_select, maxnselectedcuts)

        # selected_cuts = [cuts[i] for i in indices[:n_select]]
        random_cuts = [cuts[i] for i in indices]

        assert len(random_cuts) == len(cuts)

        return {
            'cuts': random_cuts,
            'nselectedcuts': n_select,
            'result': SCIP_RESULT.SUCCESS
        }


class CutSelectNormalizedViolation(Cutsel):
    def __init__(self, scip_model, sel_cuts_percent=0.05):
        super().__init__()
        self.sel_cuts_percent = sel_cuts_percent
        self.scip_model = scip_model

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):

        num_cuts = len(cuts)

        # 如果切割数量较少，直接返回所有切割
        if num_cuts <= 1 or maxnselectedcuts <= 0:
            return {
                'cuts': cuts[:num_cuts],
                'nselectedcuts': num_cuts,
                'result': SCIP_RESULT.SUCCESS
            }

        # 获取当前LP解
        lp_solution = self.scip_model.getBestSol()
        if lp_solution is None:
            return {
                'cuts': cuts[:num_cuts],
                'nselectedcuts': num_cuts,
                'result': SCIP_RESULT.SUCCESS
            }

        # 计算每个切割的归一化违背值
        scored_cuts = []
        for cut in cuts:
            normalized_violation = compute_normalized_violation_scores(cut)
            scored_cuts.append((cut, normalized_violation))

        # 按归一化违背值降序排序
        scored_cuts.sort(key=lambda x: x[1], reverse=True)

        # 选择前 sel_cuts_percent 比例的切割
        num_select = min(maxnselectedcuts, int(len(scored_cuts) * self.sel_cuts_percent))
        selected_cuts = [cut for cut, _ in scored_cuts]
        return {
            'cuts': selected_cuts,
            'nselectedcuts': num_select,
            'result': SCIP_RESULT.SUCCESS
        }


class CutSelectEfficacy(Cutsel):
    def __init__(self, scip_model, sel_cuts_percent=0.05):
        super().__init__()
        self.sel_cuts_percent = sel_cuts_percent
        self.scip_model = scip_model

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):

        num_cuts = len(cuts)

        # 如果切割数量较少，直接返回所有切割
        if num_cuts <= 1 or maxnselectedcuts <= 0:
            return {
                'cuts': cuts[:num_cuts],
                'nselectedcuts': num_cuts,
                'result': SCIP_RESULT.SUCCESS
            }

        # 计算每个切割的Efficacy
        scored_cuts = []
        for cut in cuts:
            eff = self.scip_model.getCutEfficacy(cut)
            scored_cuts.append((cut, eff))

        # 按Efficacy值降序排序
        scored_cuts.sort(key=lambda x: x[1], reverse=True)

        # 选择前 sel_cuts_percent 比例的切割
        num_select = min(maxnselectedcuts, int(len(scored_cuts) * self.sel_cuts_percent))
        selected_cuts = [cut for cut, _ in scored_cuts]
        return {
            'cuts': selected_cuts,
            'nselectedcuts': num_select,
            'result': SCIP_RESULT.SUCCESS
        }

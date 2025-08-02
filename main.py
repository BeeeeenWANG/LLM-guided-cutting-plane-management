import json
import requests
from pyscipopt import Model
from tools import CutSelAgent, CutSelRandom, CutSelectNormalizedViolation, CutSelectEfficacy
import numpy as np
import pandas as pd
import time
import sys
import os
import pdb
from tqdm import tqdm
import argparse

# 添加上一级目录到 sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#国内API接口
API_KEY = ""
MODEL_NAME = ""
API_URL = ""

#国外API接口
# API_KEY_ABO = ""
API_KEY_ABO = ""
MODEL_NAME_ABO = ""
API_URL_ABO = ""

time_list = []
gap_list = []
dualbound_list = []
Primalbound_list = []
primalDualIntegral_list = []
nodeNum_list = []
lpTierNum_list = []
separator_cache = {}  # cache for separator configuration
pd_gap = []


def get_recommended_separators(problem_type, model):
    system_prompt = """
    You are an expert in mathematical optimization, specializing in the solution of integer and mixed-integer programming problems.
    Based on the type of problem provided by the user, recommend the most effective separators in the SCIP solver.
    Return only a JSON-formatted list of separators without any additional explanation or text."""


    """
    你是一个数学优化专家，精通整数规划和混合整数规划问题的求解。
    请根据用户提供的问题类型，推荐SCIP求解器中最有效的切割选择器(separators)。
    只返回JSON格式的切割选择器列表，不要包含任何其他解释或文本。
    """

    # user_prompt = f"""
    # Problem type: {problem_type}

    # Available separator types: aggregation, cgmip, clique, closecuts, cmir, convexproj, disjunctive, eccuts, flowcover, gauge, gomory, impliedbounds, intobj, knapsackcover, mcf, oddcycle, rapidlearning, zerohalf.

    # Please recommend separators from the available types based on the following guidelines:
    # 1. Knapsack problem: Prefer strongcg, flowcover, clique, knapsackcover, impliedbounds, etc.
    # 2. Maximum Independent Set (MIS): Prefer gomory, clique, zerohalf, etc.
    # 3. Set Cover problem: Prefer strongcg, flowcover, cmir, impliedbounds, etc.
    # 4. Car lot scheduling (carlot): Prefer impliedbounds, zerohalf, disjunctive, cmir, aggregation
    # 5. General MILP: Recommend gomory, cmir
    # 6. Other problem types: Please analyze and recommend accordingly.

    # For the {problem_type} problem, return the 3 most effective separators.
    # Format example: ["separator1", "separator2", "separator3"]
    # """

    # """
    # 问题类型: {problem_type}

    # 可选择切割选择器种类：aggregation,cgmip,clique,closecuts,cmir,convexproj,disjunctive,eccuts,flowcover,gauge,gomory,impliedbounds,intobj,knapsackcover,mcf,oddcycle,rapidlearning,zerohalf.

    # 请根据以下指南从可选择切割选择器种类中推荐切割选择器:
    # 1. 背包问题(knapsack): 优先推荐 strongcg, flowcover, clique, knapsackcover,impliedbounds等
    # 2. 最大独立集(MIS): 优先推荐 gomory, clique, zerohalf等
    # 3. 集合覆盖(setcover): 优先推荐 strongcg, flowcover,cmir,impliedbounds等
    # 4. 巡逻调度(carlot): 优先推荐impliedbounds, zerohalf, disjunctive, cmir, aggregation
    # 5. 一般混合整数规划: 推荐 gomory, cmir
    # 6. 其他类型问题：请你通过分析给出推荐

    # 对于{problem_type}问题，请返回最有效的3个切割选择器。
    # 格式示例: ["separator1", "separator2", "separator3"]
    # """
    is_root = model.getNNodes() == 0
    node_position = "root node" if is_root else "branch node"
    user_prompt = f"""
    You are an expert in mathematical optimization. Based on the following information, recommend the most effective subset of separators (≤ 3) for solving the given MILP problem.

    Problem type: {problem_type}
    Node position: {node_position}

    Available separator list: aggregation, cgmip, clique, closecuts, cmir, convexproj, disjunctive, eccuts, flowcover, gauge, gomory, impliedbounds, intobj, knapsackcover, mcf, oddcycle, rapidlearning, zerohalf.

    Your goal is to select separators that:
    - are well-suited to the problem structure
    - generate strong cutting planes
    - reduce LP solving time and node count
    - perform well at the current node position ({node_position})

    Only output the list of selected separators in JSON format. 
    Example output: ["clique", "gomory", "cmir", "oddcycle"]
    """

    # 调用OpenAI API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY_ABO}"
    }

    payload = {
        "model": MODEL_NAME_ABO,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "max_tokens": 4096,
        "enable_thinking": True,
        "thinking_budget": 4096,
        "min_p": 0.05,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.8,
        "n": 1,
        "stop": [],
    }

    try:
        response = requests.post(API_URL_ABO, json=payload, headers=headers, timeout=1000)
        response.raise_for_status()

        # 解析响应
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        separator_data = json.loads(content)

        # 提取切割选择器列表
        if "separators" in separator_data:
            return separator_data["separators"]
        elif "recommendations" in separator_data:
            return separator_data["recommendations"]
        else:
            return separator_data if separator_data else []

    except Exception as e:
        print(f"API调用失败: {str(e)}")
        # 返回默认推荐
        return get_default_separators(problem_type)


def get_default_separators(problem_type):
    """默认切割选择器配置"""
    problem_type = problem_type.lower()
    return ["aggregation", "cgmip", "clique", "closecuts", "cmir", "convexproj",
            "disjunctive", "eccuts", "flowcover", "gauge", "gomory", "impliedbounds",
            "intobj", "knapsackcover", "mcf", "oddcycle", "rapidlearning", "zerohalf", "strongcg"]
    # if "knap" in problem_type:
    #     return ["impliedbounds", "knapsackcover", "gomory"]
    # elif "mis" in problem_type or "independent" in problem_type:
    #     return ["clique", "zerohalf"]
    # elif "cover" in problem_type or "setcover" in problem_type:
    #     return ["impliedbounds", "clique", "flowcover"]
    # elif "flow" in problem_type:
    #     return ["flowcover", "mcf", "gomory"]
    # else:  # 通用MIP配置
    #     return ["gomory", "cmir", "clique"]


def configure_scip_with_separators(model, separators):
    """
    在SCIP模型中配置切割选择器

    参数:
        model: SCIP模型实例
        separators (list): 切割选择器名称列表
    """
    # 获取所有可用切割选择器
    all_separators = ["aggregation", "cgmip", "clique", "closecuts", "cmir", "convexproj",
                      "disjunctive", "eccuts", "flowcover", "gauge", "gomory", "impliedbounds",
                      "intobj", "knapsackcover", "mcf", "oddcycle", "rapidlearning", "zerohalf", "strongcg"]

    dis_separators = [sepa for sepa in all_separators if sepa not in separators]

    # 启用推荐的切割选择器
    enabled_count = 0
    for sepa_name in separators:
        # 查找匹配的切割选择器
        found = False
        if sepa_name in all_separators:
            model.setParam(f'separating/{sepa_name}/freq', 1)
            enabled_count += 1
            # print(f"已启用: {sepa_name}")
            found = True
        if not found:
            print(f"警告: 切割选择器 '{sepa_name}' 不可用")

    for sepa_name in dis_separators:
        model.setParam(f'separating/{sepa_name}/freq', -1)

    # 如果没有启用任何切割选择器，使用默认配置
    if enabled_count == 0:
        print("未启用任何切割选择器，使用默认配置")
        for sepa_name in ["gomory", "cmir"]:
            model.setParam(f'separating/{sepa_name}/freq', 1)

    # print(f"已启用 {enabled_count} 个切割选择器")


def solve_lp_with_ai_optimization(lp_file, problem_type):
    """
    使用AI优化的切割策略求解LP问题

    参数:
        lp_file (str): LP文件路径
        problem_type (str): 问题类型描述
    """
    # 创建模型
    model = Model("AI-Optimized Solver")

    try:
        # 读取问题
        model.readProblem(lp_file)
        # print(f"\n已加载问题: {lp_file}")
        # print(f"问题类型: {problem_type}")

        # 获取推荐的切割选择器
        # print("\n获取AI推荐的切割选择器...")
        # 检查缓存
        if problem_type in separator_cache:
            # if False:
            recommended_separators = separator_cache[problem_type]
            # print("使用缓存的切割选择器配置")
        else:
            # recommended_separators = None #如果是nosep，则采用这条注释下一条
            recommended_separators = get_recommended_separators(problem_type, model)
            if not recommended_separators:
                recommended_separators = get_default_separators(problem_type)
                print("使用默认切割选择器配置")
            # 存储到缓存
            # separator_cache[problem_type] = recommended_separators

        # 配置SCIP
        configure_scip_with_separators(model, recommended_separators)

        cut_selector = CutSelAgent(
            scip_model=model,
            model_name=MODEL_NAME_ABO,
            api_url=API_URL_ABO,
            api_key=API_KEY_ABO,
            sel_cuts_percent=0.1
        )
        model.includeCutsel(cut_selector, "LLMCutSelector",
                            "Cut selector using large language model",
                            priority=100000)
        model.setIntParam("display/verblevel", 0)
        model.setRealParam('limits/time', 300)  #求解时间
        # model.setParam("separating/maxroundsroot", 5)
        # model.setParam("separating/maxstallroundsroot", 5)
        # model.setParam("separating/maxcutsroot", 50)

        # model.setParam("separating/maxrounds", 5)
        # model.setParam("separating/maxstallrounds", 0)
        # model.setParam("separating/maxcuts", 50)
        # 求解问题
        # print("\n开始求解...")
        model.optimize()

        # 输出结果
        status = model.getStatus()
        primal_bound = model.getPrimalbound()
        dual_bound = model.getDualbound()

        # 计算 Primal-Dual Gap（相对误差）
        # 1.0 + abs(primal_bound) 是为了避免除 0
        primal_dual_gap = abs(primal_bound - dual_bound) / (1.0 + abs(primal_bound))

        pd_gap.append(primal_dual_gap)
        primalDualIntegral_list.append(model.getPrimalDualIntegral())
        time_list.append(model.getSolvingTime())
        nodeNum_list.append(model.getNNodes())
        gap_list.append(model.getGap())
        lpTierNum_list.append(model.getNLPIterations())
        dualbound_list.append(model.getDualbound())
        Primalbound_list.append(model.getPrimalbound())
        # round_num = model.getNCutRounds()
        # print("\n求解统计:")
        # print(f"PDI: {model.getPrimalDualIntegral()}")
        # print(f"求解时间: {model.getSolvingTime():.2f} 秒")
        # print(f"节点数: {model.getNNodes()}")
        # print(f"LP迭代次数: {model.getNLPIterations()}")

    except Exception as e:
        print(f"求解过程中出错: {str(e)}")
    finally:
        model.freeProb()  # 释放内存


def solve_lp_with_default_scip(lp_file):
    """
    使用AI优化的切割策略求解LP问题

    参数:
        lp_file (str): LP文件路径
        problem_type (str): 问题类型描述
    """
    # 创建模型
    model = Model("Default-Optimized Solver")

    try:
        # 读取问题
        model.readProblem(lp_file)
        # print(f"\n已加载问题: {lp_file}")
        # print(f"问题类型: {problem_type}")

        recommended_separators = get_default_separators(problem_type)

        # 配置SCIP
        configure_scip_with_separators(model, recommended_separators)

        model.setIntParam("display/verblevel", 0)
        model.setRealParam('limits/time', 300)  #求解时间
        model.setParam("separating/maxroundsroot", 5)
        model.setParam("separating/maxstallroundsroot", 5)
        model.setParam("separating/maxcutsroot", 50)

        model.setParam("separating/maxrounds", 5)
        model.setParam("separating/maxstallrounds", 0)
        model.setParam("separating/maxcuts", 50)
        # 求解问题
        model.optimize()

        # 输出结果
        status = model.getStatus()
        primal_bound = model.getPrimalbound()
        dual_bound = model.getDualbound()

        # 计算 Primal-Dual Gap（相对误差）
        # 1.0 + abs(primal_bound) 是为了避免除 0
        primal_dual_gap = abs(primal_bound - dual_bound) / (1.0 + abs(primal_bound))


        pd_gap.append(primal_dual_gap)
        primalDualIntegral_list.append(model.getPrimalDualIntegral())
        time_list.append(model.getSolvingTime())
        nodeNum_list.append(model.getNNodes())
        gap_list.append(model.getGap())
        lpTierNum_list.append(model.getNLPIterations())
        dualbound_list.append(model.getDualbound())
        Primalbound_list.append(model.getPrimalbound())
        # print("\n求解统计:")
        # print(f"PDI: {model.getPrimalDualIntegral()}")
        # print(f"求解时间: {model.getSolvingTime():.2f} 秒")
        # print(f"节点数: {model.getNNodes()}")

    except Exception as e:
        print(f"求解过程中出错: {str(e)}")
    finally:
        model.freeProb()  # 释放内存


def solve_lp_with_nocut_optimization(lp_file, problem_type):
    """
    使用AI配置separators求解LP问题

    参数:
        lp_file (str): LP文件路径
        problem_type (str): 问题类型描述
    """
    # 创建模型
    model = Model("Nocut-Optimized Solver")

    try:
        # 读取问题
        model.readProblem(lp_file)
        # print(f"\n已加载问题: {lp_file}")
        # print(f"问题类型: {problem_type}")

        # 获取推荐的切割选择器
        # print("\n获取AI推荐的切割选择器...")
        # 检查缓存
        if problem_type in separator_cache:
            # if False:
            recommended_separators = separator_cache[problem_type]
            # print("使用缓存的切割选择器配置")
        else:
            # recommended_separators = None
            recommended_separators = get_recommended_separators(problem_type, model)
            if not recommended_separators:
                recommended_separators = get_default_separators(problem_type)
                print("使用默认切割选择器配置")
            # 存储到缓存
            separator_cache[problem_type] = recommended_separators

        # 配置SCIP
        configure_scip_with_separators(model, recommended_separators)

        model.setIntParam("display/verblevel", 0)
        model.setRealParam('limits/time', 300)  #求解时间
        model.setParam("separating/maxroundsroot", 5)
        model.setParam("separating/maxstallroundsroot", 5)
        model.setParam("separating/maxcutsroot", 50)

        model.setParam("separating/maxrounds", 5)
        model.setParam("separating/maxstallrounds", 0)
        model.setParam("separating/maxcuts", 50)
        # 求解问题
        model.optimize()

        # 输出结果
        status = model.getStatus()
        primal_bound = model.getPrimalbound()
        dual_bound = model.getDualbound()

        # 计算 Primal-Dual Gap（相对误差）
        # 1.0 + abs(primal_bound) 是为了避免除 0
        primal_dual_gap = abs(primal_bound - dual_bound) / (1.0 + abs(primal_bound))

        pd_gap.append(primal_dual_gap)
        primalDualIntegral_list.append(model.getPrimalDualIntegral())
        time_list.append(model.getSolvingTime())
        nodeNum_list.append(model.getNNodes())
        gap_list.append(model.getGap())
        lpTierNum_list.append(model.getNLPIterations())
        dualbound_list.append(model.getDualbound())
        Primalbound_list.append(model.getPrimalbound())

    except Exception as e:
        print(f"求解过程中出错: {str(e)}")
    finally:
        model.freeProb()  # 释放内存


def solve_lp_with_random_scip(lp_file):
    """
    使用随机选择的切割策略求解LP问题
    参数:
        lp_file (str): LP文件路径
        problem_type (str): 问题类型描述
    """
    # 创建模型
    model = Model("Random-Optimized Solver")

    try:
        # 读取问题
        model.readProblem(lp_file)
        # print(f"\n已加载问题: {lp_file}")
        # print(f"问题类型: {problem_type}")

        recommended_separators = get_default_separators(problem_type)

        # 配置SCIP
        configure_scip_with_separators(model, recommended_separators)

        cut_selector = CutSelRandom(sel_cuts_percent=0.1)
        model.includeCutsel(cut_selector, "CutSelRandom", "Random cut selector", priority=100000)

        # cut_selector = CutSelAgent(
        #     scip_model=model,
        #     api_key="",
        #     sel_cuts_percent=0.1
        # )
        # model.includeCutsel(cut_selector, "LLMCutSelector",
        #                         "Cut selector using large language model",
        #                         priority=100000)

        model.setIntParam("display/verblevel", 0)
        model.setRealParam('limits/time', 300)  #求解时间
        model.setParam("separating/maxroundsroot", 5)
        model.setParam("separating/maxstallroundsroot", 5)
        model.setParam("separating/maxcutsroot", 50)

        model.setParam("separating/maxrounds", 5)
        model.setParam("separating/maxstallrounds", 0)
        model.setParam("separating/maxcuts", 50)
        # 求解问题
        model.optimize()

        # 输出结果
        status = model.getStatus()
        primal_bound = model.getPrimalbound()
        dual_bound = model.getDualbound()

        # 计算 Primal-Dual Gap（相对误差）
        # 1.0 + abs(primal_bound) 是为了避免除 0
        primal_dual_gap = abs(primal_bound - dual_bound) / (1.0 + abs(primal_bound))

        pd_gap.append(primal_dual_gap)
        primalDualIntegral_list.append(model.getPrimalDualIntegral())
        time_list.append(model.getSolvingTime())
        nodeNum_list.append(model.getNNodes())
        gap_list.append(model.getGap())
        lpTierNum_list.append(model.getNLPIterations())
        dualbound_list.append(model.getDualbound())
        Primalbound_list.append(model.getPrimalbound())
        # print("\n求解统计:")
        # print(f"PDI: {model.getPrimalDualIntegral()}")
        # print(f"求解时间: {model.getSolvingTime():.2f} 秒")
        # print(f"节点数: {model.getNNodes()}")

    except Exception as e:
        print(f"求解过程中出错: {str(e)}")
    finally:
        model.freeProb()  # 释放内存



def solve_lp_with_nv_scip(lp_file):
    """
    使用选择最高Normalized Violation的切割策略求解LP问题
    参数:
        lp_file (str): LP文件路径
    """
    # 创建模型
    model=Model("Normalized_Violation-Optimized Solver")

    try:
        # 读取问题
        model.readProblem(lp_file)
        # print(f"\n已加载问题: {lp_file}")
        # print(f"问题类型: {problem_type}")

        recommended_separators = get_default_separators(problem_type)

        # 配置SCIP
        configure_scip_with_separators(model, recommended_separators)

        cut_selector = CutSelectNormalizedViolation(model, sel_cuts_percent=0.1)
        model.includeCutsel(cut_selector, "CutSelNormalizedViolation", "Normalized Violation cut selector", priority=100000)

        model.setIntParam("display/verblevel", 0)
        model.setRealParam('limits/time', 300)  #求解时间
        model.setParam("separating/maxroundsroot", 5)
        model.setParam("separating/maxstallroundsroot", 5)
        model.setParam("separating/maxcutsroot", 50)

        model.setParam("separating/maxrounds", 5)
        model.setParam("separating/maxstallrounds", 0)
        model.setParam("separating/maxcuts", 50)

        # 求解问题
        model.optimize()

        # 输出结果
        status = model.getStatus()
        primal_bound = model.getPrimalbound()
        dual_bound = model.getDualbound()

        # 计算 Primal-Dual Gap（相对误差）
        # 1.0 + abs(primal_bound) 是为了避免除 0
        primal_dual_gap = abs(primal_bound - dual_bound) / (1.0 + abs(primal_bound))

        pd_gap.append(primal_dual_gap)
        primalDualIntegral_list.append(model.getPrimalDualIntegral())
        time_list.append(model.getSolvingTime())
        nodeNum_list.append(model.getNNodes())
        gap_list.append(model.getGap())
        lpTierNum_list.append(model.getNLPIterations())
        dualbound_list.append(model.getDualbound())
        Primalbound_list.append(model.getPrimalbound())
        # print("\n求解统计:")
        # print(f"PDI: {model.getPrimalDualIntegral()}")
        # print(f"求解时间: {model.getSolvingTime():.2f} 秒")
        # print(f"节点数: {model.getNNodes()}")

    except Exception as e:
        print(f"求解过程中出错: {str(e)}")
    finally:
        model.freeProb()  # 释放内存


def solve_lp_with_eff_scip(lp_file):
    """
    使用选择最高Normalized Violation的切割策略求解LP问题
    参数:
        lp_file (str): LP文件路径
    """
    # 创建模型
    model=Model("Efficacy-Optimized Solver")

    try:
        # 读取问题
        model.readProblem(lp_file)
        # print(f"\n已加载问题: {lp_file}")
        # print(f"问题类型: {problem_type}")

        recommended_separators = get_default_separators(problem_type)

        # 配置SCIP
        configure_scip_with_separators(model, recommended_separators)

        cut_selector = CutSelectEfficacy(model, sel_cuts_percent=0.1)
        model.includeCutsel(cut_selector, "CutSelEfficacy", "Efficacy cut selector", priority=100000)

        model.setIntParam("display/verblevel", 0)
        model.setRealParam('limits/time', 300)  #求解时间
        model.setParam("separating/maxroundsroot", 5)
        model.setParam("separating/maxstallroundsroot", 5)
        model.setParam("separating/maxcutsroot", 50)

        model.setParam("separating/maxrounds", 5)
        model.setParam("separating/maxstallrounds", 0)
        model.setParam("separating/maxcuts", 50)

        # 求解问题
        model.optimize()

        # 输出结果
        status = model.getStatus()
        primal_bound = model.getPrimalbound()
        dual_bound = model.getDualbound()

        # 计算 Primal-Dual Gap（相对误差）
        # 1.0 + abs(primal_bound) 是为了避免除 0
        primal_dual_gap = abs(primal_bound - dual_bound) / (1.0 + abs(primal_bound))

        pd_gap.append(primal_dual_gap)
        primalDualIntegral_list.append(model.getPrimalDualIntegral())
        time_list.append(model.getSolvingTime())
        nodeNum_list.append(model.getNNodes())
        gap_list.append(model.getGap())
        lpTierNum_list.append(model.getNLPIterations())
        dualbound_list.append(model.getDualbound())
        Primalbound_list.append(model.getPrimalbound())
        # print("\n求解统计:")
        # print(f"PDI: {model.getPrimalDualIntegral()}")
        # print(f"求解时间: {model.getSolvingTime():.2f} 秒")
        # print(f"节点数: {model.getNNodes()}")

    except Exception as e:
        print(f"求解过程中出错: {str(e)}")
    finally:
        model.freeProb()  # 释放内存



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['knapsack', 'mis', 'setcover', 'mik', 'carlot'],
                        help='Dataset type')
    parser.add_argument('--mode', type=str, required=True, choices=['default', 'ai', 'nocut', 'random', 'nv', 'eff'],
                        help='solve method')
    args = parser.parse_args()

    if args.dataset == 'knapsack':
        folder_path = "/home/wangzhe/L2O-HEM-Torch-master/HEM_ICLR23_Dataset/easy/knapsack/test_60_12"
        problem_type = 'knapsack'
    elif args.dataset == 'mis':
        folder_path = "/home/wangzhe/L2O-HEM-Torch-master/HEM_ICLR23_Dataset/easy/mis/transfer_500_4"
        problem_type = 'mis'
    elif args.dataset == 'setcover':
        folder_path = "/home/wangzhe/L2O-HEM-Torch-master/HEM_ICLR23_Dataset/easy/setcover/transfer_500r_1000c_0.05d"
        problem_type = 'setcover'
    elif args.dataset == 'carlot':
        folder_path = "/home/wangzhe/L2O-HEM-Torch-master/HEM_ICLR23_Dataset/medium/corlat/test"
        problem_type = 'carlot'
    elif args.dataset == 'mik':
        folder_path = "/home/wangzhe/L2O-HEM-Torch-master/HEM_ICLR23_Dataset/medium/mik/test"
        problem_type = 'mik'
    else:
        raise ValueError('Unknown dataset type')

    file_names = os.listdir(folder_path)
    for i in tqdm(file_names, desc="Solving LP Instances"):
        lp_file = os.path.join(folder_path, str(i))
        if args.mode == 'ai':
            solve_lp_with_ai_optimization(lp_file, problem_type)
        elif args.mode == 'default':
            solve_lp_with_default_scip(lp_file)
        elif args.mode == 'nocut':
            solve_lp_with_nocut_optimization(lp_file, problem_type)
        elif args.mode == 'random':
            solve_lp_with_random_scip(lp_file)
        elif args.mode == 'nv':
            solve_lp_with_nv_scip(lp_file)
        elif args.mode == 'eff':
            solve_lp_with_eff_scip(lp_file)
        else:
            raise ValueError('Unknown solving mode')
        time.sleep(3)  # to prevent 429 Client Error: Too Many Requests for url

    print("Avg. PDI:", np.mean(primalDualIntegral_list))
    print("Avg. time:", np.mean(time_list))
    print("Avg. Node num:", np.mean(nodeNum_list))
    print("Avg. Dual bound:", np.mean(dualbound_list))
    print("Avg. Primal bound:", np.mean(Primalbound_list))
    print("Avg. Primal dual gap:", np.mean(pd_gap))

    # df = pd.DataFrame({
    #     'PDI': primalDualIntegral_list,
    #     'Time': time_list
    # })

    # df.to_csv('llm_comb_result.csv',index=False,encoding='utf-8')

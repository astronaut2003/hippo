"""
LongMemEval 评估模块（优化版）

参考 zep-longmem-eval 的评估设计，简化提示词，移除对抗性问题支持。
适配新的 LongMemEval 数据集结构。

主要改进：
1. 简化 LLM grader 提示词，更宽松的评分标准
2. 移除对抗性问题的评估逻辑
3. 统一评分接口，简化 API
4. 保持原有的评估指标（EM, F1, ROUGE, BLEU, METEOR, Semantic Similarity, BERT F1）
"""

from datetime import datetime
import regex
import json
import string
import unicodedata
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter
import logging

# 评估相关库
from bert_score import score
from nltk.stem import PorterStemmer
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from llm_client import LLMClient

# 尝试下载NLTK资源
# try:
#     nltk.download("wordnet", quiet=True)
#     nltk.download("punkt", quiet=True)
# except Exception as e:
#     logging.warning(f"Failed to download NLTK resources: {e}")

# 初始化词干提取器
ps = PorterStemmer()

# ================================
# ZEP-style Grading Prompts（不同问题类型使用不同提示词）
# ================================

GRADING_PROMPTS = {
    "temporal-reasoning": """
    I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

    <QUESTION>
    B: {question}
    </QUESTION>
    <CORRECT ANSWER>
    {gold_answer}
    </CORRECT ANSWER>
    <RESPONSE>
    A: {response}
    </RESPONSE>
    """,
    "knowledge-update": """
    I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

    <QUESTION>
    B: {question}
    </QUESTION>
    <CORRECT ANSWER>
    {gold_answer}
    </CORRECT ANSWER>
    <RESPONSE>
    A: {response}
    </RESPONSE>
    """,
    "single-session-preference": """
    I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

    <QUESTION>
    B: {question}
    </QUESTION>
    <RUBRIC>
    {gold_answer}
    </RUBRIC>
    <RESPONSE>
    A: {response}
    </RESPONSE>
    """,
    "default": """
    I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

    <QUESTION>
    B: {question}
    </QUESTION>
    <CORRECT ANSWER>
    {gold_answer}
    </CORRECT ANSWER>
    <RESPONSE>
    A: {response}
    </RESPONSE>
    """,
}

# ================================
# 模型管理器 - 避免重复初始化
# ================================

class ModelManager:
    """模型管理器 - 统一管理各种评估模型，避免重复加载"""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.ModelManager")
    
    def get_sentence_model(self, model_name: str = "all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
        """获取句子嵌入模型"""
        cache_key = f"sentence_transformer:{model_name}"
        
        if cache_key in self._models:
            return self._models[cache_key]
        
        try:
            self.logger.info(f"正在加载句子嵌入模型: {model_name}")
            model = SentenceTransformer(model_name)
            self._models[cache_key] = model
            self.logger.info(f"✅ 句子嵌入模型加载成功: {model_name}")
            return model
        except Exception as e:
            self.logger.error(f"加载句子嵌入模型失败 {model_name}: {e}")
            self._models[cache_key] = None
            return None
    
    def get_bert_score_model(self) -> bool:
        """检查BERTScore模型是否可用"""
        cache_key = "bert_score_available"
        
        if cache_key in self._models:
            return self._models[cache_key]
        
        try:
            from bert_score import score as bert_score
            _, _, f1 = bert_score(["test"], ["test"], lang="en", verbose=False)
            self._models[cache_key] = True
            self.logger.info("✅ BERTScore模型可用")
            return True
        except Exception as e:
            self.logger.warning(f"BERTScore模型不可用: {e}")
            self._models[cache_key] = False
            return False
    
    def clear_cache(self):
        """清空模型缓存"""
        for cache_key, model in self._models.items():
            try:
                if hasattr(model, 'cpu'):
                    model.cpu()
                if hasattr(model, 'cleanup'):
                    model.cleanup()
            except Exception as e:
                self.logger.warning(f"清理模型 {cache_key} 失败: {e}")
        
        self._models.clear()
        
        # 清理GPU缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self.logger.info("评估模型缓存已清空")

# 全局模型管理器实例
_model_manager: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """获取全局模型管理器（单例）"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

def cleanup_evaluation_models():
    """清理评估模型缓存"""
    global _model_manager
    if _model_manager:
        _model_manager.clear_cache()
        _model_manager = None

# ================================
# LLM 评估器（简化版，参考 zep-longmem-eval）
# ================================

def llm_grader(llm_client: LLMClient, 
               question: str, 
               gold_answer: str, 
               response: str,
               question_type: str = "default",
               context: str = "") -> bool:
    """
    使用 LLM 判断答案是否正确（ZEP-style）
    
    完全对齐 ZEP 的评估逻辑：
    - 根据问题类型使用不同的 grading prompt
    - 严格的 yes/no 判断标准
    - 支持 temporal-reasoning, knowledge-update, single-session-preference 等类型
    
    Args:
        llm_client: LLM客户端
        question: 问题文本
        gold_answer: 标准答案
        response: 生成的答案
        question_type: 问题类型（temporal-reasoning, knowledge-update, single-session-preference, default）
        context: 上下文（保留参数以兼容现有代码，但ZEP不使用）
        
    Returns:
        是否正确（True/False）
    """
    
    # 🔥 ZEP-style: 根据问题类型选择 prompt
    prompt_template = GRADING_PROMPTS.get(question_type, GRADING_PROMPTS["default"])
    prompt = prompt_template.format(
        question=question,
        gold_answer=gold_answer,
        response=response
    )
    
    # System prompt（与ZEP保持一致）
    system_prompt = "You are an expert grader that determines if answers to questions match a gold standard answer"

    try:
        # 🔥 尝试使用 JSON 格式（模拟 ZEP 的结构化输出）
        full_prompt = f"""{system_prompt}

        {prompt}

        Return ONLY a JSON object with "is_correct" key containing "yes" or "no".
        Example: {{"is_correct": "yes"}} or {{"is_correct": "no"}}
        """
        
        llm_response = llm_client.generate_answer(
            prompt=full_prompt,
            temperature=0.0,
            max_tokens=50,
            json_format=True
        )
        
        # 解析 JSON 响应
        try:
            if '{' in llm_response and '}' in llm_response:
                start = llm_response.find('{')
                end = llm_response.rfind('}') + 1
                json_str = llm_response[start:end]
                result = json.loads(json_str)
                
                # ZEP 使用 "is_correct" 字段
                is_correct = result.get("is_correct", "").strip().lower()
                return is_correct == "yes"
            else:
                # 回退到文本解析
                llm_response_lower = llm_response.lower().strip()
                
                # ZEP-style: 严格匹配 "yes" 或 "no"
                if llm_response_lower == "yes":
                    return True
                elif llm_response_lower == "no":
                    return False
                
                # 如果响应中包含 yes/no，提取第一个出现的
                if "yes" in llm_response_lower and "no" not in llm_response_lower:
                    return True
                elif "no" in llm_response_lower and "yes" not in llm_response_lower:
                    return False
                else:
                    # 默认返回 False（严格评分）
                    logging.warning(f"Ambiguous LLM response: {llm_response}")
                    return False
                    
        except json.JSONDecodeError as e:
            logging.warning(f"JSON解析失败: {llm_response}, 错误: {e}")
            
            # 回退：严格的 yes/no 匹配
            llm_response_lower = llm_response.lower().strip()
            
            if llm_response_lower == "yes":
                return True
            elif llm_response_lower == "no":
                return False
            
            # 检查是否包含 yes 或 no
            contains_yes = "yes" in llm_response_lower
            contains_no = "no" in llm_response_lower
            
            if contains_yes and not contains_no:
                return True
            elif contains_no and not contains_yes:
                return False
            else:
                # ZEP-style: 默认严格评分，返回 False
                logging.warning(f"Cannot parse LLM response: {llm_response}")
                return False
                
    except Exception as e:
        logging.error(f"LLM grader 失败: {e}")
        return False

def calculate_llm_judgment(llm_client: LLMClient, 
                        question: str, 
                        gold_answer: str, 
                        response: str,
                        question_type: str = "default",
                        num_runs: int = 1,
                        context: str = "") -> Dict[str, Any]:
    """
    计算LLM判断分数（支持多次运行，ZEP-style）
    
    Args:
        llm_client: LLM客户端
        question: 问题
        gold_answer: 标准答案
        response: 生成答案
        question_type: 问题类型（ZEP-style）
        num_runs: 运行次数（用于一致性检查）
        context: 上下文（保留以兼容，但ZEP不使用）
        
    Returns:
        LLM判断结果字典
    """
    judgments = []
    
    for i in range(num_runs):
        try:
            result = llm_grader(llm_client, question, gold_answer, response, question_type, context)
            judgments.append(result)
        except Exception as e:
            logging.warning(f"LLM判断第 {i+1} 次失败: {e}")
            continue
    
    if not judgments:
        return {
            "judgments": [],
            "accuracy": 0.0,
            "num_runs": num_runs,
            "consistency": False,
            "question_type": question_type,
            "error": "所有判断都失败了"
        }
    
    accuracy = sum(judgments) / len(judgments)
    consistency = len(set(judgments)) == 1
    
    return {
        "judgments": judgments,
        "accuracy": accuracy,
        "num_runs": num_runs,
        "consistency": consistency,
        "confidence": "high" if consistency else "low",
        "question_type": question_type,
        "context_provided": bool(context and context.strip())
    }

# ================================
# 综合评估函数
# ================================

def calculate_comprehensive_scores(gold_answer: str, 
                                 response: str, 
                                 question: str = "", 
                                 context: str = "",
                                 question_type: str = "default",  # 🔥 新增参数
                                 llm_client: Optional[LLMClient] = None,
                                 metrics: Optional[List[str]] = None,
                                 sentence_model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """
    计算全面的评估分数（ZEP-compatible）
    
    Args:
        gold_answer: 标准答案
        response: 生成的答案
        question: 问题文本
        context: 上下文
        question_type: 问题类型（ZEP-style）
        llm_client: LLM客户端（可选）
        metrics: 要计算的指标列表
        sentence_model_name: 句子嵌入模型名称
        
    Returns:
        包含各种评估指标的字典
    """
    
    # 默认指标
    if llm_client is not None and metrics is None:
        metrics = ["exact_match", "f1", "rouge", "bleu", "meteor", "semantic_similarity", "bert_f1", "llm_judge"]
    if metrics is None:
        metrics = ["exact_match", "f1", "rouge", "bleu", "meteor", "semantic_similarity", "bert_f1"]
    
    # 数据预处理
    gold_answer = str(gold_answer).strip() if gold_answer else ""
    response = str(response).strip() if response else ""
    
    results = {
        "input_info": {
            "gold_length": len(gold_answer.split()),
            "response_length": len(response.split()),
            "context_length": len(context.split()) if context else 0,
            "question_type": question_type  # 🔥 添加问题类型信息
        },
        "scores": {}
    }
    
    # 基础指标（保持不变）
    if "exact_match" in metrics:
        try:
            results["scores"]["exact_match"] = float(exact_match_score(gold_answer, response))
        except Exception as e:
            logging.warning(f"精确匹配计算失败: {e}")
            results["scores"]["exact_match"] = 0.0
    
    if "f1" in metrics:
        try:
            results["scores"]["token_f1"] = calculate_f1_score(gold_answer, response)
        except Exception as e:
            logging.warning(f"F1计算失败: {e}")
            results["scores"]["token_f1"] = 0.0
    
    if "rouge" in metrics:
        try:
            rouge_scores = calculate_rouge_score(gold_answer, response)
            results["scores"].update(rouge_scores)
        except Exception as e:
            logging.warning(f"ROUGE计算失败: {e}")
            results["scores"].update({"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0})
    
    if "bleu" in metrics:
        try:
            bleu_scores = calculate_bleu_score(gold_answer, response)
            results["scores"].update(bleu_scores)
        except Exception as e:
            logging.warning(f"BLEU计算失败: {e}")
            results["scores"].update({"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0})
    
    if "meteor" in metrics:
        try:
            results["scores"]["meteor"] = calculate_meteor_score(gold_answer, response)
        except Exception as e:
            logging.warning(f"METEOR计算失败: {e}")
            results["scores"]["meteor"] = 0.0
    
    if "semantic_similarity" in metrics:
        try:
            results["scores"]["semantic_similarity"] = calculate_semantic_similarity(
                gold_answer, response, sentence_model_name
            )
        except Exception as e:
            logging.warning(f"语义相似度计算失败: {e}")
            results["scores"]["semantic_similarity"] = 0.0
    
    if "bert_f1" in metrics:
        try:
            results["scores"]["bert_f1"] = calculate_bert_f1_score(gold_answer, response)
        except Exception as e:
            logging.warning(f"BERT F1计算失败: {e}")
            results["scores"]["bert_f1"] = 0.0
    
    # 🔥 LLM评估（ZEP-style，使用 question_type）
    if llm_client and question and "llm_judge" in metrics:
        try:
            llm_result = calculate_llm_judgment(
                llm_client, question, gold_answer, response, 
                question_type=question_type,  # 🔥 传递问题类型
                num_runs=1, 
                context=context
            )
            results["scores"]["llm_accuracy"] = llm_result["accuracy"]
            results["llm_details"] = llm_result
        except Exception as e:
            logging.warning(f"LLM评估失败: {e}")
            results["scores"]["llm_accuracy"] = 0.0
            results["llm_details"] = {"error": str(e)}
    
    # 计算综合分数（保持不变）
    try:
        lexical_scores = []
        semantic_scores = []
        
        for key in ["exact_match", "token_f1", "rouge1_f", "rougeL_f", "bleu4", "meteor"]:
            if key in results["scores"]:
                lexical_scores.append(results["scores"][key])
        
        for key in ["semantic_similarity", "bert_f1"]:
            if key in results["scores"]:
                semantic_scores.append(results["scores"][key])
        
        if lexical_scores:
            results["scores"]["avg_lexical"] = sum(lexical_scores) / len(lexical_scores)
        if semantic_scores:
            results["scores"]["avg_semantic"] = sum(semantic_scores) / len(semantic_scores)
        
        all_scores = lexical_scores + semantic_scores
        if all_scores:
            results["scores"]["overall_average"] = sum(all_scores) / len(all_scores)
            
    except Exception as e:
        logging.warning(f"综合分数计算失败: {e}")
    
    results = convert_numpy_types(results)
    results["evaluation_success"] = True
    
    return results

# ================================
# 批量评估
# ================================

def batch_evaluate(questions: List[str],
                  gold_answers: List[str], 
                  predicted_answers: List[str],
                  contexts: Optional[List[str]] = None,
                  llm_client: Optional[LLMClient] = None,
                  metrics: Optional[List[str]] = None,
                  include_individual: bool = False,
                  sentence_model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """
    批量评估多个问答对
    
    Args:
        questions: 问题列表
        gold_answers: 标准答案列表
        predicted_answers: 预测答案列表
        contexts: 上下文列表（可选）
        llm_client: LLM客户端（可选）
        metrics: 要计算的指标列表
        include_individual: 是否包含单个样本的详细结果
        sentence_model_name: 句子嵌入模型名称
        
    Returns:
        包含批量评估结果的字典
    """
    if not (len(questions) == len(gold_answers) == len(predicted_answers)):
        raise ValueError("输入列表长度不一致")
    
    if contexts is None:
        contexts = [""] * len(questions)
    elif len(contexts) != len(questions):
        raise ValueError("上下文列表长度与问题列表不一致")
    
    results = {
        "summary": {
            "total_samples": len(questions),
            "evaluation_metrics": metrics or ["exact_match", "f1", "rouge", "bleu", "meteor", "semantic_similarity", "bert_f1"],
            "timestamp": datetime.now().isoformat(),
            "sentence_model": sentence_model_name
        },
        "aggregate_scores": {},
        "individual_results": [] if include_individual else None
    }
    
    # 预加载模型
    manager = get_model_manager()
    if "semantic_similarity" in (metrics or []):
        manager.get_sentence_model(sentence_model_name)
    if "bert_f1" in (metrics or []):
        manager.get_bert_score_model()
    
    # 收集所有评估结果
    all_scores = []
    failed_count = 0
    
    for i, (question, gold_answer, predicted_answer, context) in enumerate(
        zip(questions, gold_answers, predicted_answers, contexts)
    ):
        try:
            eval_result = calculate_comprehensive_scores(
                gold_answer=gold_answer,
                response=predicted_answer,
                question=question,
                context=context,
                llm_client=llm_client,
                metrics=metrics,
                sentence_model_name=sentence_model_name
            )
            
            all_scores.append(eval_result["scores"])
            
            if include_individual:
                results["individual_results"].append({
                    "index": i,
                    "question": question,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "evaluation": eval_result
                })
                
        except Exception as e:
            logging.error(f"评估第{i+1}个样本失败: {e}")
            failed_count += 1
            
            if include_individual:
                results["individual_results"].append({
                    "index": i,
                    "question": question,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "evaluation": {"error": str(e)}
                })
        
        # 进度输出
        if (i + 1) % 100 == 0:
            logging.info(f"批量评估进度: {i + 1}/{len(questions)} ({(i + 1)/len(questions)*100:.1f}%)")
    
    # 计算聚合统计
    if all_scores:
        metric_values = {}
        for score_dict in all_scores:
            for metric_name, value in score_dict.items():
                if isinstance(value, (int, float)):
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    metric_values[metric_name].append(value)
        
        for metric_name, values in metric_values.items():
            if values:
                results["aggregate_scores"][metric_name] = {
                    "mean": sum(values) / len(values),
                    "std": np.std(values).item() if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "median": np.median(values).item(),
                    "count": len(values)
                }
    
    results["summary"]["failed_evaluations"] = failed_count
    results["summary"]["success_rate"] = (len(questions) - failed_count) / len(questions) if questions else 0.0
    
    return results

# ================================
# 基础评估函数（保持不变）
# ================================

def calculate_semantic_similarity(gold_answer: str, 
                                response: str, 
                                model_name: str = "all-MiniLM-L6-v2") -> float:
    """计算语义相似度"""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    if not gold_answer.strip() or not response.strip():
        return 0.0
    
    try:
        sentence_model = get_model_manager().get_sentence_model(model_name)
        if sentence_model is None:
            return 0.0
            
        gold_embedding = sentence_model.encode([gold_answer], show_progress_bar=False)[0]
        response_embedding = sentence_model.encode([response], show_progress_bar=False)[0]
        similarity = 1 - cosine(gold_embedding, response_embedding)
        
        return max(0.0, min(1.0, similarity))
        
    except Exception as e:
        logging.error(f"Failed to calculate semantic similarity: {e}")
        return 0.0

def calculate_bert_f1_score(gold_answer: str, response: str) -> float:
    """计算BERT F1分数"""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    if not gold_answer.strip() or not response.strip():
        return 0.0
    
    try:
        manager = get_model_manager()
        if not manager.get_bert_score_model():
            return 0.0
        
        _, _, f1 = score([response], [gold_answer], lang="en", rescale_with_baseline=True, verbose=False)
        return f1.item() if f1 is not None else 0.0
    except Exception as e:
        logging.error(f"Failed to calculate BERT F1 score: {e}")
        return 0.0

class SimpleTokenizer(object):
    """简单的分词器类"""
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def normalize_answer(s):
    """答案标准化"""
    if s is None:
        s = ""
    elif not isinstance(s, str):
        s = str(s)
    
    s = s.replace(',', "")
    
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(gold_answer: str, response: str) -> bool:
    """精确匹配得分"""
    response = str(response) if response is not None else ""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    
    response = normalize_answer(response)
    gold_answer = normalize_answer(gold_answer)
    return set(response.split()) == set(gold_answer.split())

def calculate_f1_score(gold_answer: str, response: str) -> float:
    """F1得分"""
    response = str(response) if response is not None else ""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    
    response_tokens = [ps.stem(w) for w in normalize_answer(response).split()]
    gold_answer_tokens = [ps.stem(w) for w in normalize_answer(gold_answer).split()]
    
    common = Counter(response_tokens) & Counter(gold_answer_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(response_tokens)
    recall = 1.0 * num_same / len(gold_answer_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def calculate_rouge_score(gold_answer: str, response: str) -> Dict[str, float]:
    """计算ROUGE分数"""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    metrics = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = scorer.score(gold_answer, response)
        metrics["rouge1_f"] = rouge_scores["rouge1"].fmeasure
        metrics["rouge2_f"] = rouge_scores["rouge2"].fmeasure
        metrics["rougeL_f"] = rouge_scores["rougeL"].fmeasure
    except Exception as e:
        logging.error(f"Failed to calculate ROUGE scores: {e}")
    
    return metrics

def calculate_bleu_score(gold_answer: str, response: str) -> Dict[str, float]:
    """计算BLEU分数"""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    metrics = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    try:
        gold_tokens = nltk.word_tokenize(gold_answer.lower())
        response_tokens = nltk.word_tokenize(response.lower())
        
        smoothing = SmoothingFunction().method1
        weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]

        for i, weight in enumerate(weights, 1):
            metrics[f"bleu{i}"] = sentence_bleu(
                [gold_tokens], response_tokens, weights=weight, smoothing_function=smoothing
            )
    except Exception as e:
        logging.error(f"Failed to calculate BLEU scores: {e}")

    return metrics

def calculate_meteor_score(gold_answer: str, response: str) -> float:
    """计算METEOR分数"""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    try:
        gold_tokens = nltk.word_tokenize(gold_answer.lower())
        response_tokens = nltk.word_tokenize(response.lower())
        return meteor_score([gold_tokens], response_tokens)
    except Exception as e:
        logging.error(f"Failed to calculate METEOR score: {e}")
        return 0.0

def convert_numpy_types(obj):
    """转换numpy类型为Python原生类型"""
    if isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

# ================================
# 报告生成
# ================================

def generate_evaluation_report(eval_results: Dict[str, Any], 
                             output_format: str = "text",
                             save_path: Optional[str] = None) -> str:
    """生成评估报告"""
    if output_format == "json":
        report = json.dumps(eval_results, indent=2, ensure_ascii=False)
    elif output_format == "markdown":
        report = _generate_markdown_report(eval_results)
    else:
        report = _generate_text_report(eval_results)
    
    if save_path:
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logging.info(f"评估报告已保存到: {save_path}")
        except Exception as e:
            logging.error(f"保存报告失败: {e}")
    
    return report

def _generate_text_report(eval_results: Dict[str, Any]) -> str:
    """生成文本格式报告"""
    lines = []
    lines.append("="*60)
    lines.append("LongMemEval 评估报告")
    lines.append("="*60)
    
    if "summary" in eval_results:
        summary = eval_results["summary"]
        lines.append(f"总样本数: {summary.get('total_samples', 'unknown')}")
        lines.append(f"成功率: {summary.get('success_rate', 0):.2%}")
        lines.append(f"失败数: {summary.get('failed_evaluations', 0)}")
        lines.append("")
    
    if "aggregate_scores" in eval_results:
        lines.append("聚合评估结果:")
        lines.append("-" * 40)
        
        for metric_name, stats in eval_results["aggregate_scores"].items():
            lines.append(f"{metric_name:20} | 均值: {stats['mean']:.4f} | 标准差: {stats['std']:.4f}")
        lines.append("")
    
    return "\n".join(lines)

def _generate_markdown_report(eval_results: Dict[str, Any]) -> str:
    """生成Markdown格式报告"""
    lines = []
    lines.append("# LongMemEval 评估报告")
    lines.append("")
    
    if "summary" in eval_results:
        summary = eval_results["summary"]
        lines.append("## 基本信息")
        lines.append(f"- **总样本数**: {summary.get('total_samples', 'unknown')}")
        lines.append(f"- **成功率**: {summary.get('success_rate', 0):.2%}")
        lines.append("")
    
    if "aggregate_scores" in eval_results:
        lines.append("## 聚合评估结果")
        lines.append("")
        lines.append("| 指标 | 均值 | 标准差 | 最小值 | 最大值 | 中位数 |")
        lines.append("|------|------|--------|--------|--------|--------|")
        
        for metric_name, stats in eval_results["aggregate_scores"].items():
            lines.append(f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['median']:.4f} |")
        lines.append("")
    
    return "\n".join(lines)

# ================================
# 测试代码
# ================================

if __name__ == "__main__":
    # 简单测试
    print("🧪 测试评估模块...")
    
    # 创建LLM客户端
    llm_client = LLMClient("deepseek-chat")
    
    # 测试用例
    question = "How long is my daily commute to work?"
    gold_answer = "45 minutes each way"
    predicted_answer = "Your daily commute takes approximately 45 minutes in each direction."
    
    # 测试LLM评估
    result = llm_grader(llm_client, question, gold_answer, predicted_answer)
    print(f"LLM评估结果: {result}")
    
    # 测试综合评估
    comprehensive_result = calculate_comprehensive_scores(
        gold_answer=gold_answer,
        response=predicted_answer,
        question=question,
        llm_client=llm_client
    )
    print(f"\n综合评估结果:")
    print(json.dumps(comprehensive_result, indent=2, ensure_ascii=False))
    
    # 清理缓存
    cleanup_evaluation_models()
    print("\n✅ 测试完成")
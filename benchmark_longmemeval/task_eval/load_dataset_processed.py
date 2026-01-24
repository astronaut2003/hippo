"""
LongMemEval 数据集加载模块（Processed 版本）

基于 mem0 接口设计的数据集加载函数，支持将 LongMemEval 数据集的对话历史
加载到 mem0 记忆系统中，并进行问答评估。

【与 unprocessed 版本的区别】：
- 默认 infer=True，启用 mem0 的记忆推理功能（提取和更新记忆）
- mem0 会对对话历史进行处理，提取关键信息形成结构化记忆
- 适用于需要记忆压缩和推理的场景
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from mem0 import Memory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔥 调试：明确获取 API Key 并验证
# cstcloud_api_key = os.getenv("CSTCLOUD_API_KEY")
# if not cstcloud_api_key:
#     raise ValueError("❌ CSTCLOUD_API_KEY 环境变量未设置！请先设置：export CSTCLOUD_API_KEY='your-api-key'")
# else:
#     logger.info(f"✅ CSTCLOUD_API_KEY 已加载，长度: {len(cstcloud_api_key)}")

config = {
    # openai接口，可用
    # "llm": {
    #     "provider": "openai",
    #     "config": {
    #         "model": "gpt-5-nano", 
    #         "temperature": 0.1, 
    #         "api_key": os.getenv("CLOSEAI_API_KEY"),
    #         "openai_base_url": "https://api.openai-proxy.org/v1",
    #     },
    # },
    # deepseek接口，可用
    "llm": {
        "provider": "deepseek",
        "config":{
            "model":"deepseek-chat",
            "temperature":0.1,
            "api_key":os.getenv("DEEPSEEK_API_KEY"),
        }
    },
    # "llm": {
    #     "provider": "openai",
    #     "config": {
    #         "model": "deepseek-v3:671b",
    #         "temperature": 0.1,
    #         "api_key": cstcloud_api_key,  # 🔥 使用明确的变量
    #         "openai_base_url": "https://uni-api.cstcloud.cn/v1"
    #     }
    # },
    # "llm": {
    #     "provider": "deepseek",
    #     "config":{
    #         "model":"deepseek-v3:671b",
    #         "temperature":0.1,
    #         "api_key":os.getenv("CSTCLOUD_API_KEY"),
    #         "deepseek_base_url":"https://uni-api.cstcloud.cn/v1"
    #     }
    # },
    "reranker": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-reranker-v2-m3",
            "device": "cuda",
            "batch_size": 32
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "all-MiniLM-L6-v2"
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "embedding_model_dims": 384
        }
    }
}


def load_dataset(dataset_path: str, sample_indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    加载 LongMemEval 数据集文件（纯数据加载，不涉及记忆系统）
    
    数据集结构说明：
    - longmemeval_s_cleaned.json: 包含500个样本的列表
    - 每个样本包含：
        - question_id: 问题唯一标识
        - question_type: 问题类型（single-session-user, temporal-reasoning等）
        - question: 问题文本
        - answer: 标准答案
        - haystack_sessions: 对话历史列表，每个元素是一个session（对话列表）
        - haystack_session_ids: 对话session的ID列表
        - haystack_dates: 对话的时间戳列表
        - answer_session_ids: 包含答案的session ID列表
    
    Args:
        dataset_path: 数据集 JSON 文件路径，支持以下格式：
            1. longmemeval_s_cleaned.json (直接包含样本列表)
            2. extracted_samples_index_X.json (包含 "samples" 字段的字典)
        sample_indices: 要加载的样本索引列表，None 表示加载所有样本
            
    Returns:
        数据集样本列表，每个样本是一个字典
        
    Raises:
        FileNotFoundError: 数据集文件不存在
        ValueError: 数据格式不支持
        
    Examples:
        >>> # 加载完整数据集
        >>> samples = load_dataset("longmemeval_s_cleaned.json")
        >>> print(f"加载了 {len(samples)} 个样本")
        
        >>> # 加载指定样本
        >>> samples = load_dataset("longmemeval_s_cleaned.json", sample_indices=[0, 1, 2])
        >>> print(f"加载了 {len(samples)} 个样本")
    """
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    logger.info(f"正在加载数据集: {dataset_path}")
    
    # 读取 JSON 文件
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理不同的数据格式
    samples = []
    
    if isinstance(data, list):
        # 格式1: 直接的列表格式 (longmemeval_s_cleaned.json, longmemeval_m_cleaned.json)
        # [{"question_id": "xxx", "question": "...", ...}, ...]
        samples = data
        logger.info(f"检测到直接列表格式，包含 {len(samples)} 个样本")
        
    elif isinstance(data, dict):
        if 'samples' in data:
            # 格式2: 包含 "samples" 字段的字典 (extracted_samples_index_X.json)
            # {"metadata": {...}, "samples": [...]}
            samples = data['samples']
            logger.info(f"检测到带 metadata 的格式，包含 {len(samples)} 个样本")
            
            # 可选：记录 metadata 信息
            if 'metadata' in data:
                metadata = data['metadata']
                logger.debug(f"数据集 metadata: {metadata}")
        else:
            raise ValueError(
                f"不支持的数据格式。期望的格式:\n"
                f"1. 直接列表: [{{'question_id': ..., ...}}, ...]\n"
                f"2. 带 samples 字段: {{'samples': [...]}}\n"
                f"实际收到的顶层字段: {list(data.keys())}"
            )
    else:
        raise ValueError(f"不支持的数据类型: {type(data)}，期望 list 或 dict")
    
    # 验证样本格式
    if len(samples) > 0:
        required_fields = ['question_id', 'question', 'answer', 'haystack_sessions']
        sample_keys = set(samples[0].keys())
        missing_fields = [f for f in required_fields if f not in sample_keys]
        
        if missing_fields:
            logger.warning(f"样本缺少以下字段: {missing_fields}")
    
    # 如果指定了样本索引，只返回指定的样本
    if sample_indices is not None:
        logger.info(f"筛选样本索引: {sample_indices}")
        
        # 过滤有效索引
        valid_indices = [i for i in sample_indices if 0 <= i < len(samples)]
        invalid_indices = [i for i in sample_indices if i < 0 or i >= len(samples)]
        
        if invalid_indices:
            logger.warning(f"以下索引超出范围，将被忽略: {invalid_indices} (总样本数: {len(samples)})")
        
        samples = [samples[i] for i in valid_indices]
        logger.info(f"筛选后保留 {len(samples)} 个样本")
    
    logger.info(f"数据集加载完成，共 {len(samples)} 个样本")
    
    return samples


class LongMemEvalLoader:
    """
    LongMemEval 数据集加载器（Processed 版本，包含记忆系统集成）
    
    【与 unprocessed 版本的区别】：
    - 默认 infer=True，启用 mem0 的记忆推理功能
    - mem0 会对对话历史进行处理，提取关键信息形成结构化记忆
    """
    
    def __init__(self, memory: Optional[Memory] = None):
        """
        初始化加载器
        
        Args:
            memory: mem0 Memory 实例，如果为 None 则使用默认配置创建新实例
        """
        self.memory = memory if memory is not None else Memory.from_config(config)
    
    def load_dataset(self, dataset_path: str, sample_indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        加载 LongMemEval 数据集文件（委托给全局 load_dataset 函数）
        
        Args:
            dataset_path: 数据集 JSON 文件路径
            sample_indices: 要加载的样本索引列表，None 表示加载所有样本
            
        Returns:
            数据集样本列表
        """
        return load_dataset(dataset_path, sample_indices)
    
    def add_conversations_to_memory(
        self, 
        sample: Dict[str, Any],
        sample_idx: int,
        user_id_base: Optional[str] = None,
        infer: bool = True,  # 🔥 默认 True，启用记忆推理
        clean_before_add: bool = True
    ) -> Dict[str, Any]:
        """
        将样本的对话历史添加到 mem0 记忆中（启用推理）
        
        【核心隔离机制】参考 Mem0 对 LOCOMO10 的处理方式：
        - 为每个样本创建唯一的 user_id: f"{user_id_base}_{sample_idx}"
        - 例如: sample_0, sample_1, sample_2, ..., sample_499
        - 这样确保500个样本之间的记忆完全隔离，互不干扰
        
        【Processed 模式特点】：
        - infer=True 时，mem0 会对对话进行推理，提取关键信息
        - 生成的记忆是经过处理和压缩的结构化记忆
        - 适用于需要记忆提取和更新的场景
        
        Args:
            sample: 数据集样本（必须包含 haystack_sessions 字段）
            sample_idx: 样本索引（0-499），用于生成唯一的 user_id
            user_id_base: user_id 的基础名称，默认使用 "sample"
            infer: 是否启用 mem0 的推理功能（默认 True）
            clean_before_add: 是否在添加前清空该 user_id 的所有记忆（确保干净状态）
            
        Returns:
            包含添加结果的字典，包括：
                - question_id: 问题ID
                - sample_idx: 样本索引
                - user_id: 唯一的用户ID
                - total_sessions: 总session数
                - added_sessions: 成功添加的session数
                - failed_sessions: 失败的session数
                - memory_results: 每个session的添加结果列表
                - infer_mode: 是否启用了推理模式
        """
        question_id = sample.get('question_id', 'unknown')
        
        # 🔥 核心：生成唯一的 user_id（类似 Mem0 的 f"{speaker_a}_{idx}" 机制）
        if user_id_base is None:
            user_id_base = "sample"
        user_id = f"{user_id_base}_{sample_idx}"
        
        # 🔥 先清空该样本的所有记忆（确保干净状态，避免跨样本污染）
        if clean_before_add:
            try:
                self.memory.delete_all(user_id=user_id)
                logger.info(f"[样本 {sample_idx}] 已清空 user_id={user_id} 的所有记忆")
            except Exception as e:
                logger.warning(f"[样本 {sample_idx}] 清空记忆失败: {e}")
        
        # 获取所有对话 sessions
        haystack_sessions = sample.get('haystack_sessions', [])
        
        results = {
            'question_id': question_id,
            'sample_idx': sample_idx,
            'user_id': user_id,
            'total_sessions': len(haystack_sessions),
            'added_sessions': 0,
            'failed_sessions': 0,
            'memory_results': [],
            'infer_mode': infer  # 🔥 记录推理模式
        }
        
        logger.info(f"[样本 {sample_idx}] 使用 infer={infer} 模式添加 {len(haystack_sessions)} 个对话")
        
        # 逐个添加对话 session
        for session_idx, session in enumerate(haystack_sessions):
            # 跳过空 session
            if not session or not isinstance(session, list):
                logger.debug(f"[样本 {sample_idx}] 跳过空 session {session_idx}")
                results['failed_sessions'] += 1
                continue
            
            try:
                # 将对话历史添加到 mem0
                # session 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
                #  infer=True 时，mem0 会对对话进行推理，提取关键信息
                result = self.memory.add(
                    messages=session,
                    user_id=user_id,  #  使用唯一的 user_id
                    infer=infer  #  默认启用推理
                )
                
                results['memory_results'].append({
                    'session_idx': session_idx,
                    'session_length': len(session),
                    'result': result
                })
                results['added_sessions'] += 1
                
                logger.debug(f"[样本 {sample_idx}] Session {session_idx} 添加成功 ({len(session)} 条消息, infer={infer})")
                
            except Exception as e:
                logger.error(f"[样本 {sample_idx}] 添加 session {session_idx} 失败: {e}")
                results['failed_sessions'] += 1
        
        logger.info(
            f"[样本 {sample_idx}] user_id={user_id}: "
            f"成功添加 {results['added_sessions']}/{results['total_sessions']} 个对话 (infer={infer})"
        )
        
        return results
    
    def query_memory(
        self, 
        question: str,
        sample_idx: int,
        user_id_base: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        根据问题查询相关记忆
        
        Args:
            question: 问题文本
            sample_idx: 样本索引，用于生成对应的 user_id
            user_id_base: user_id 的基础名称，默认使用 "sample"
            top_k: 返回的记忆数量
            
        Returns:
            相关记忆列表
        """
        # 生成与 add_conversations_to_memory 相同的 user_id
        if user_id_base is None:
            user_id_base = "sample"
        user_id = f"{user_id_base}_{sample_idx}"
        
        try:
            memories = self.memory.search(
                query=question,
                user_id=user_id,
                limit=top_k
            )
            
            # 🔥 修复：处理 mem0 的返回格式
            # mem0 返回格式: {'results': [{'id': ..., 'memory': ..., ...}, ...]}
            if isinstance(memories, dict) and 'results' in memories:
                memories_list = memories['results']
                logger.info(f"[样本 {sample_idx}] 查询到 {len(memories_list)} 条相关记忆")
                return memories_list
            elif isinstance(memories, list):
                # 兼容旧版本可能直接返回列表的情况
                logger.info(f"[样本 {sample_idx}] 查询到 {len(memories)} 条相关记忆")
                return memories
            else:
                logger.warning(f"[样本 {sample_idx}] 未知的记忆返回格式: {type(memories)}")
                return []
            
        except Exception as e:
            logger.error(f"[样本 {sample_idx}] 查询记忆失败: {e}")
            return []
    
    def load_sample(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
        user_id_base: Optional[str] = None,
        infer: bool = True,  # 🔥 默认 True
        clean_before_add: bool = True
    ) -> Dict[str, Any]:
        """
        仅仅加载单个样本的对话历史到记忆中（不执行查询，启用推理）
        
        【核心隔离机制】：
        - 每个样本使用唯一的 user_id: f"{user_id_base}_{sample_idx}"
        - 处理前先清空该 user_id 的所有记忆
        - 确保样本之间完全隔离，互不干扰
        
        【Processed 模式特点】：
        - 默认 infer=True，启用 mem0 的记忆推理功能
        
        Args:
            sample: 数据集样本
            sample_idx: 样本索引（0-499）
            user_id_base: user_id 的基础名称，默认使用 "sample"
            infer: 是否启用推理（默认 True）
            clean_before_add: 是否在添加前清空记忆
            
        Returns:
            加载结果字典，包括：
                - question_id: 问题ID
                - sample_idx: 样本索引
                - user_id: 唯一的用户ID
                - question: 问题文本
                - question_type: 问题类型
                - gold_answer: 标准答案
                - question_date: 问题日期
                - add_result: 添加对话历史的结果
        """
        question_id = sample.get('question_id', 'unknown')
        
        # 生成唯一的 user_id
        if user_id_base is None:
            user_id_base = "sample"
        user_id = f"{user_id_base}_{sample_idx}"
        
        logger.info(f"[样本 {sample_idx}] 开始加载对话历史 question_id={question_id}, user_id={user_id}, infer={infer}")
        
        # 添加对话历史到记忆（内部会先清空）
        add_result = self.add_conversations_to_memory(
            sample=sample,
            sample_idx=sample_idx,
            user_id_base=user_id_base,
            infer=infer,
            clean_before_add=clean_before_add
        )
        
        # 返回加载结果
        return {
            'question_id': question_id,
            'sample_idx': sample_idx,
            'user_id': user_id,
            'question': sample.get('question', ''),
            'question_type': sample.get('question_type', 'unknown'),
            'gold_answer': sample.get('answer', ''),
            'question_date': sample.get('question_date', ''),
            'add_result': add_result
        }
    
    def search_sample(
        self,
        question: str,
        sample_idx: int,
        user_id_base: Optional[str] = None,
        query_top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        根据问题查询相关记忆（简化接口）
        
        Args:
            question: 问题文本
            sample_idx: 样本索引，用于生成对应的 user_id
            user_id_base: user_id 的基础名称，默认使用 "sample"
            query_top_k: 返回的记忆数量
            
        Returns:
            相关记忆列表
        """
        memories = self.query_memory(
            question=question,
            sample_idx=sample_idx,
            user_id_base=user_id_base,
            top_k=query_top_k
        )
        return memories
    
    def process_sample(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
        user_id_base: Optional[str] = None,
        infer: bool = True,  # 🔥 默认 True
        query_top_k: int = 5,
        clean_before_add: bool = True
    ) -> Dict[str, Any]:
        """
        处理单个样本：清空记忆 -> 加载对话历史（启用推理） -> 查询问题 -> 返回结果
        
        【核心隔离机制】：
        - 每个样本使用唯一的 user_id: f"{user_id_base}_{sample_idx}"
        - 处理前先清空该 user_id 的所有记忆
        - 确保样本之间完全隔离，互不干扰
        
        【Processed 模式特点】：
        - 默认 infer=True，启用 mem0 的记忆推理功能
        - mem0 会对对话历史进行处理，提取关键信息形成结构化记忆
        
        Args:
            sample: 数据集样本
            sample_idx: 样本索引（0-499）
            user_id_base: user_id 的基础名称，默认使用 "sample"
            infer: 是否启用推理（默认 True）
            query_top_k: 查询返回的记忆数量
            clean_before_add: 是否在添加前清空记忆
            
        Returns:
            处理结果字典，包括：
                - question_id: 问题ID
                - sample_idx: 样本索引
                - user_id: 唯一的用户ID
                - question: 问题文本
                - question_type: 问题类型
                - gold_answer: 标准答案
                - question_date: 问题日期
                - add_result: 添加对话历史的结果
                - retrieved_memories: 检索到的记忆列表
                - num_memories: 检索到的记忆数量
                - infer_mode: 是否启用了推理模式
        """
        question_id = sample.get('question_id', 'unknown')
        
        # 生成唯一的 user_id
        if user_id_base is None:
            user_id_base = "sample"
        user_id = f"{user_id_base}_{sample_idx}"
        
        logger.info(f"[样本 {sample_idx}] 开始处理 question_id={question_id}, user_id={user_id}, infer={infer}")
        
        # 1. 添加对话历史到记忆（内部会先清空）
        add_result = self.add_conversations_to_memory(
            sample=sample,
            sample_idx=sample_idx,
            user_id_base=user_id_base,
            infer=infer,
            clean_before_add=clean_before_add
        )
        
        # 2. 查询问题
        question = sample.get('question', '')
        logger.info(f"[样本 {sample_idx}] 查询问题: {question}")
        memories = self.query_memory(
            question=question,
            sample_idx=sample_idx,
            user_id_base=user_id_base,
            top_k=query_top_k
        )
        
        # 3. 返回完整结果
        return {
            'question_id': question_id,
            'sample_idx': sample_idx,
            'user_id': user_id,
            'question': question,
            'question_type': sample.get('question_type', 'unknown'),
            'gold_answer': sample.get('answer', ''),
            'question_date': sample.get('question_date', ''),
            'add_result': add_result,
            'retrieved_memories': memories,
            'num_memories': len(memories),
            'infer_mode': infer  # 🔥 记录推理模式
        }
    
    def reset_memory(self, sample_idx: Optional[int] = None, user_id_base: Optional[str] = None):
        """
        清空记忆
        
        Args:
            sample_idx: 如果提供，只清空该样本对应的记忆；否则清空所有记忆
            user_id_base: user_id 的基础名称，默认使用 "sample"
        """
        if sample_idx is not None:
            # 清空特定样本的记忆
            if user_id_base is None:
                user_id_base = "sample"
            user_id = f"{user_id_base}_{sample_idx}"
            self.memory.delete_all(user_id=user_id)
            logger.info(f"已清空样本 {sample_idx} (user_id={user_id}) 的所有记忆")
        else:
            # 清空所有记忆
            self.memory.reset()
            logger.info("已清空所有记忆")


def load_longmemeval_s(
    dataset_path: str = "benchmark_longmemeval/dataset/LongMemEval/longmemeval_s_cleaned.json",
    memory: Optional[Memory] = None,
    sample_indices: Optional[List[int]] = None,
    user_id_base: Optional[str] = None,
    infer: bool = True,  # 🔥 默认 True
    clean_before_add: bool = True
) -> Tuple[LongMemEvalLoader, List[Dict[str, Any]]]:
    """
    加载 LongMemEval-S 数据集的便捷函数（Processed 版本，支持样本隔离）
    
    【核心隔离机制】：
    - 每个样本使用唯一的 user_id: f"{user_id_base}_{idx}"
    - 确保500个样本之间的记忆完全隔离
    
    【Processed 模式特点】：
    - 默认 infer=True，启用 mem0 的记忆推理功能
    
    Args:
        dataset_path: 数据集路径
        memory: mem0 Memory 实例
        sample_indices: 要处理的样本索引列表，None 表示处理所有样本
        user_id_base: user_id 的基础名称，默认使用 "sample"
        infer: 是否启用推理（默认 True）
        clean_before_add: 是否在添加前清空每个样本的记忆
        
    Returns:
        (loader, results) 元组
    """
    loader = LongMemEvalLoader(memory=memory)
    
    # 加载数据集
    samples = loader.load_dataset(dataset_path, sample_indices=sample_indices)
    logger.info(f"加载了 {len(samples)} 个样本 (infer={infer})")
    
    # 处理每个样本
    results = []
    for idx, sample in enumerate(samples):
        # 🔥 关键：使用样本在原始数据集中的索引（如果有 sample_index 字段）
        # 或者使用当前循环的索引
        original_idx = sample.get('sample_index', idx)
        
        logger.info(f"处理样本 {idx + 1}/{len(samples)} (original_idx={original_idx}, infer={infer})")
        
        result = loader.process_sample(
            sample=sample,
            sample_idx=original_idx,  # 🔥 使用原始索引确保唯一性
            user_id_base=user_id_base,
            infer=infer,
            clean_before_add=clean_before_add
        )
        results.append(result)
    
    return loader, results


# 使用示例
if __name__ == "__main__":
    # 示例 1: 加载单个样本（从提取的样本文件）
    print("="*60)
    print("LongMemEval Processed 模式示例")
    print("（默认 infer=True，启用记忆推理）")
    print("="*60)
    
    loader = LongMemEvalLoader()
    
    # 加载提取的样本
    sample_path = "benchmark_longmemeval/dataset/LongMemEval/extracted_samples_index_1.json"
    samples = loader.load_dataset(sample_path)
    
    # 只处理第一个样本（longmemeval_s）
    s_sample = samples[0]  # longmemeval_s_cleaned 的样本
    sample_idx = s_sample.get('sample_index', 1)  # 获取原始索引
    
    print(f"\n{'='*60}")
    print(f"处理样本索引: {sample_idx}")
    print(f"问题ID: {s_sample['question_id']}")
    print(f"问题类型: {s_sample['question_type']}")
    print(f"问题: {s_sample['question']}")
    print(f"标准答案: {s_sample['answer']}")
    print(f"对话 session 数: {len(s_sample['haystack_sessions'])}")
    print(f"{'='*60}\n")
    
    # 🔥 使用 Processed 模式：infer=True（默认值）
    result = loader.process_sample(
        sample=s_sample,
        sample_idx=sample_idx,
        user_id_base="longmemeval_s_processed",
        # infer=True,  # 默认就是 True
        query_top_k=5
    )
    
    # 打印结果
    print(f"\n添加结果:")
    print(f"  - 样本索引: {result['sample_idx']}")
    print(f"  - User ID: {result['user_id']}")
    print(f"  - 推理模式: {result.get('infer_mode', True)}")
    print(f"  - 总会话数: {result['add_result']['total_sessions']}")
    print(f"  - 成功添加: {result['add_result']['added_sessions']}")
    print(f"  - 失败: {result['add_result']['failed_sessions']}")
    
    print(f"\n检索到的记忆 ({result['num_memories']} 条):")
    for i, memory in enumerate(result['retrieved_memories'][:5], 1):
        mem_text = memory.get('memory', 'N/A')
        if len(mem_text) > 100:
            mem_text = mem_text[:100] + "..."
        print(f"  {i}. {mem_text}")
        print(f"     Score: {memory.get('score', 'N/A'):.4f}, Rerank: {memory.get('rerank_score', 'N/A'):.4f}")
    
    # 🔥 清理特定样本的记忆
    loader.reset_memory(sample_idx=sample_idx, user_id_base="longmemeval_s_processed")
    
    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)
    
    # 示例 2: 批量处理多个样本（展示隔离效果）
    print("\n\n" + "="*60)
    print("示例 2: 批量处理（Processed 模式，展示样本隔离）")
    print("="*60)
    
    # 加载前 2 个样本
    loader2, results = load_longmemeval_s(
        dataset_path=sample_path,
        sample_indices=[0, 1],
        user_id_base="demo_processed",
        infer=True  # 🔥 默认就是 True
    )
    
    print(f"\n批量处理完成，共处理 {len(results)} 个样本:")
    for r in results:
        print(f"  - 样本 {r['sample_idx']}: user_id={r['user_id']}, "
              f"添加了 {r['add_result']['added_sessions']} 个会话, "
              f"检索到 {r['num_memories']} 条记忆, "
              f"infer={r.get('infer_mode', True)}")
    
    # 清理
    for r in results:
        loader2.reset_memory(sample_idx=r['sample_idx'], user_id_base="demo_processed")
    
    print("\n" + "="*60)
    print("批量处理示例完成！")
    print("="*60)
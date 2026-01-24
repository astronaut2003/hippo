"""
LongMemEval Benchmark 测试脚本（Processed 版本 - 并行）

使用 mem0 记忆系统和 LLM 进行问答评估
逐条保存结果到 benchmark_results_processed/ 目录

【并行设计要点】：
1. 每个样本使用独立的 user_id (user_id_base_{sample_idx}_{worker_id})，确保记忆隔离
2. 使用进程池进行并行处理，避免 GIL 限制
3. 每个 worker 独立初始化 mem0 和 LLM 客户端，避免共享状态
4. 使用信号量控制 LLM API 并发数，避免限流
5. 文件写入无冲突（每个 QA 独立目录）

【与串行版本的区别】：
- 支持多进程并行处理
- 增加 worker_id 确保跨进程记忆隔离
- 增加并发控制参数
"""

import json
import os
import sys
import time
import argparse
import multiprocessing as mp
from multiprocessing import Pool, Manager, Lock
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
import traceback
import signal
import atexit

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Worker 初始化函数（每个进程独立初始化）
# ============================================================

# 全局变量用于存储 worker 的资源（每个进程独立）
_worker_resources: Dict[str, Any] = {}


def init_worker(
    worker_id: int,
    dataset_path: str,
    gen_llm_model: str,
    eval_llm_model: str,
    user_id_base: str,
    infer: bool,
    output_dir: str
):
    """
    初始化 worker 进程的资源
    
    【方案2：独立路径模式】
    每个 worker 使用独立的 Qdrant 存储路径，避免文件锁冲突
    路径格式: /tmp/qdrant_worker_{process_id}
    
    每个 worker 独立初始化：
    - LongMemEvalLoader (包含 mem0 Memory 实例，使用独立存储)
    - 生成 LLM 客户端
    - 评估 LLM 客户端
    
    Args:
        worker_id: Worker 进程 ID
        dataset_path: 数据集路径
        gen_llm_model: 生成 LLM 模型名称
        eval_llm_model: 评估 LLM 模型名称
        user_id_base: user_id 基础名称
        infer: 是否启用 mem0 推理
        output_dir: 输出目录
    """
    global _worker_resources
    
    import os
    import shutil
    from mem0 import Memory
    from task_eval.llm_client import LLMClient
    from task_eval.load_dataset_processed import LongMemEvalLoader
    
    process_id = os.getpid()
    
    # 🔥 核心：为每个 worker 创建独立的 Qdrant 存储路径
    worker_qdrant_path = f"/tmp/qdrant_worker_{process_id}"
    
    # 清理旧的存储目录（如果存在）
    if os.path.exists(worker_qdrant_path):
        try:
            shutil.rmtree(worker_qdrant_path)
            logger.info(f"[Worker-{worker_id}] 清理旧存储目录: {worker_qdrant_path}")
        except Exception as e:
            logger.warning(f"[Worker-{worker_id}] 清理旧目录失败: {e}")
    
    logger.info(f"[Worker-{worker_id}] PID={process_id} 初始化中...")
    logger.info(f"[Worker-{worker_id}] 使用独立 Qdrant 存储: {worker_qdrant_path}")
    
    try:
        # 🔥 为每个 worker 创建独立的 mem0 配置
        worker_config = {
            "llm": {
                "provider": "deepseek",
                "config": {
                    "model": "deepseek-chat",
                    "temperature": 0.1,
                    "api_key": os.getenv("DEEPSEEK_API_KEY"),
                }
            },
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
                    "path": worker_qdrant_path,  # 🔥 每个 worker 独立路径
                    "embedding_model_dims": 384
                }
            }
        }
        
        # 使用独立配置创建 Memory 实例
        memory = Memory.from_config(worker_config)
        
        # 创建 Loader（传入已创建的 memory 实例）
        loader = LongMemEvalLoader(memory=memory)
        
        # 初始化生成 LLM 客户端
        gen_llm_client = LLMClient(model_name=gen_llm_model)
        
        # 初始化评估 LLM 客户端
        if eval_llm_model == gen_llm_model:
            eval_llm_client = gen_llm_client
        else:
            eval_llm_client = LLMClient(model_name=eval_llm_model)
        
        # 保存资源到全局变量
        _worker_resources = {
            'worker_id': worker_id,
            'process_id': process_id,
            'loader': loader,
            'gen_llm_client': gen_llm_client,
            'eval_llm_client': eval_llm_client,
            'gen_llm_model': gen_llm_model,
            'eval_llm_model': eval_llm_model,
            'user_id_base': user_id_base,
            'infer': infer,
            'output_dir': Path(output_dir),
            'qdrant_path': worker_qdrant_path  # 🔥 记录路径，用于清理
        }
        
        logger.info(f"[Worker-{worker_id}] 初始化完成 (infer={infer}, qdrant={worker_qdrant_path})")
        
    except Exception as e:
        logger.error(f"[Worker-{worker_id}] 初始化失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def cleanup_worker():
    """清理 worker 资源，包括独立的 Qdrant 存储"""
    global _worker_resources
    import shutil
    
    if _worker_resources:
        worker_id = _worker_resources.get('worker_id', 'unknown')
        qdrant_path = _worker_resources.get('qdrant_path')
        
        logger.info(f"[Worker-{worker_id}] 清理资源...")
        
        # 🔥 清理独立的 Qdrant 存储目录
        if qdrant_path and os.path.exists(qdrant_path):
            try:
                shutil.rmtree(qdrant_path)
                logger.info(f"[Worker-{worker_id}] 已清理 Qdrant 存储: {qdrant_path}")
            except Exception as e:
                logger.warning(f"[Worker-{worker_id}] 清理 Qdrant 存储失败: {e}")
        
        _worker_resources.clear()


def cleanup_worker():
    """清理 worker 资源"""
    global _worker_resources
    if _worker_resources:
        worker_id = _worker_resources.get('worker_id', 'unknown')
        logger.info(f"[Worker-{worker_id}] 清理资源...")
        _worker_resources.clear()


# ============================================================
# 单个样本处理函数（在 worker 进程中执行）
# ============================================================

def process_single_sample_worker(args: Tuple) -> Dict[str, Any]:
    """
    处理单个样本（worker 进程中执行）
    
    【并行隔离策略】：
    - user_id 格式: {user_id_base}_{sample_idx}_w{worker_id}
    - 确保不同 worker 处理同一样本时也不会冲突
    - 处理完成后立即清理记忆
    
    Args:
        args: (sample, sample_idx, query_top_k) 元组
        
    Returns:
        处理结果字典
    """
    global _worker_resources
    
    # 延迟导入
    from task_eval.evaluation import calculate_comprehensive_scores
    
    sample, sample_idx, query_top_k = args
    
    # 获取 worker 资源
    worker_id = _worker_resources.get('worker_id', 0)
    loader = _worker_resources['loader']
    gen_llm_client = _worker_resources['gen_llm_client']
    eval_llm_client = _worker_resources['eval_llm_client']
    gen_llm_model = _worker_resources['gen_llm_model']
    eval_llm_model = _worker_resources['eval_llm_model']
    user_id_base = _worker_resources['user_id_base']
    infer = _worker_resources['infer']
    output_dir = _worker_resources['output_dir']
    
    # 🔥 核心：生成带 worker_id 的唯一 user_id，确保并行隔离
    # 格式: benchmark_processed_0_w1 (样本0, worker 1)
    parallel_user_id_base = f"{user_id_base}_{sample_idx}_w{worker_id}"
    
    qa_start_time = time.time()
    
    question_id = sample.get('question_id', 'unknown')
    question = sample.get('question', '')
    question_type = sample.get('question_type', 'unknown')
    gold_answer = sample.get('answer', '')
    question_date = sample.get('question_date', '')
    
    logger.info(f"[Worker-{worker_id}][QA_{sample_idx}] 开始处理: {question_id}")
    
    result = {
        'sample_idx': sample_idx,
        'question_id': question_id,
        'question': question,
        'question_type': question_type,
        'gold_answer': gold_answer,
        'question_date': question_date,
        'user_id': parallel_user_id_base,
        'worker_id': worker_id,
        'infer_mode': infer,
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    }
    
    # 初始化保存数据结构
    score_data = {
        'sample_idx': sample_idx,
        'question_id': question_id,
        'question': question,
        'question_type': question_type,
        'gold_answer': gold_answer,
        'question_date': question_date,
        'gen_llm_model': gen_llm_model,
        'eval_llm_model': eval_llm_model,
        'worker_id': worker_id,
        'infer_mode': infer,
        'timestamp': datetime.now().isoformat()
    }
    
    retrieval_data = {
        'sample_idx': sample_idx,
        'question_id': question_id,
        'question': question,
        'query_top_k': query_top_k,
        'worker_id': worker_id,
        'infer_mode': infer,
        'timestamp': datetime.now().isoformat()
    }
    
    timing_info = {
        'load_time': 0.0,
        'retrieval_time': 0.0,
        'generation_time': 0.0,
        'evaluation_time': 0.0,
        'cleanup_time': 0.0,
        'total_time': 0.0
    }
    
    try:
        # 1. 加载对话历史到记忆系统
        load_start = time.time()
        
        load_result = loader.load_sample(
            sample=sample,
            sample_idx=sample_idx,
            user_id_base=parallel_user_id_base,  # 🔥 使用带 worker_id 的 user_id
            infer=infer,
            clean_before_add=True  # 🔥 始终先清空，确保干净状态
        )
        
        timing_info['load_time'] = round(time.time() - load_start, 4)
        
        load_info = {
            'total_sessions': load_result['add_result']['total_sessions'],
            'added_sessions': load_result['add_result']['added_sessions'],
            'failed_sessions': load_result['add_result']['failed_sessions'],
            'infer_mode': load_result['add_result'].get('infer_mode', infer)
        }
        
        result['load_result'] = load_info
        retrieval_data['load_result'] = load_info
        
        # 2. 检索相关记忆
        retrieval_start = time.time()
        
        memories = loader.search_sample(
            question=question,
            sample_idx=sample_idx,
            user_id_base=parallel_user_id_base,
            query_top_k=query_top_k
        )
        
        timing_info['retrieval_time'] = round(time.time() - retrieval_start, 4)
        
        result['retrieved_memories_count'] = len(memories)
        result['retrieved_memories'] = memories
        retrieval_data['retrieved_memories_count'] = len(memories)
        retrieval_data['retrieved_memories'] = memories
        
        # 3. 生成答案
        generation_start = time.time()
        
        prompt = _create_qa_prompt(question, memories, question_type)
        gen_prompt_tokens = gen_llm_client.count_tokens(prompt)
        gen_context_info = gen_llm_client.get_context_info()
        
        predicted_answer = gen_llm_client.generate_answer(
            prompt=prompt,
            temperature=0.1,
            max_tokens=512
        )
        
        timing_info['generation_time'] = round(time.time() - generation_start, 4)
        
        gen_answer_tokens = gen_llm_client.count_tokens(predicted_answer)
        gen_token_usage = {
            'prompt_tokens': gen_prompt_tokens,
            'answer_tokens': gen_answer_tokens,
            'total_tokens': gen_prompt_tokens + gen_answer_tokens,
            'context_length': gen_context_info.get('context_length', 0),
            'prompt_ratio': round(gen_prompt_tokens / gen_context_info.get('context_length', 1) * 100, 2),
        }
        
        result['predicted_answer'] = predicted_answer
        result['gen_token_usage'] = gen_token_usage
        score_data['predicted_answer'] = predicted_answer
        score_data['gen_token_usage'] = gen_token_usage
        
        # 4. 评估答案
        evaluation_start = time.time()
        
        try:
            eval_scores = calculate_comprehensive_scores(
                gold_answer=gold_answer,
                response=predicted_answer,
                question=question,
                question_type=question_type,
                llm_client=eval_llm_client,
                metrics=['exact_match', 'f1', 'rouge', 'semantic_similarity', 'llm_judge']
            )
            
            result['evaluation'] = eval_scores
            score_data['evaluation'] = eval_scores
            score_data['scores'] = eval_scores.get('scores', {})
            
        except Exception as eval_error:
            logger.warning(f"[Worker-{worker_id}][QA_{sample_idx}] 评估失败: {eval_error}")
            result['evaluation'] = {'error': str(eval_error)}
            score_data['evaluation'] = {'error': str(eval_error)}
            score_data['scores'] = {}
        
        timing_info['evaluation_time'] = round(time.time() - evaluation_start, 4)
        
        # 5. 🔥 清理记忆（关键！避免记忆污染其他样本）
        cleanup_start = time.time()
        loader.reset_memory(sample_idx=sample_idx, user_id_base=parallel_user_id_base)
        timing_info['cleanup_time'] = round(time.time() - cleanup_start, 4)
        
        result['status'] = 'success'
        score_data['status'] = 'success'
        retrieval_data['status'] = 'success'
        
    except Exception as e:
        logger.error(f"[Worker-{worker_id}][QA_{sample_idx}] 处理失败: {e}")
        logger.error(traceback.format_exc())
        
        result['status'] = 'failed'
        result['error'] = str(e)
        score_data['status'] = 'failed'
        score_data['error'] = str(e)
        retrieval_data['status'] = 'failed'
        retrieval_data['error'] = str(e)
        
        # 🔥 失败时也要尝试清理记忆
        try:
            loader.reset_memory(sample_idx=sample_idx, user_id_base=parallel_user_id_base)
        except:
            pass
    
    # 计算总时间
    timing_info['total_time'] = round(time.time() - qa_start_time, 4)
    result['timing'] = timing_info
    score_data['timing'] = timing_info
    retrieval_data['timing'] = {
        'retrieval_time': timing_info['retrieval_time'],
        'load_time': timing_info['load_time']
    }
    
    # 保存结果
    _save_sample_results(output_dir, sample_idx, score_data, retrieval_data)
    
    logger.info(
        f"[Worker-{worker_id}][QA_{sample_idx}] 完成: "
        f"status={result['status']}, time={timing_info['total_time']:.2f}s"
    )
    
    return result


def _create_qa_prompt(question: str, memories: List[Dict], question_type: str) -> str:
    """创建 QA prompt"""
    if not memories:
        memories_text = "No relevant memories found."
    else:
        formatted_parts = []
        for i, mem in enumerate(memories, 1):
            memory_text = mem.get('memory', '')
            score = mem.get('score', 0)
            rerank_score = mem.get('rerank_score', 0)
            formatted_parts.append(
                f"Memory {i} (relevance: {score:.3f}, rerank: {rerank_score:.3f}):\n{memory_text}"
            )
        memories_text = "\n\n".join(formatted_parts)
    
    prompt = f"""You are a helpful assistant with access to the user's conversation history and memories.

Based on the following retrieved memories, please answer the user's question accurately and concisely.

Question Type: {question_type}

Retrieved Memories:
{memories_text}

User Question: {question}

Instructions:
- Answer based ONLY on the information provided in the memories above
- If the memories don't contain enough information, say "I don't have enough information to answer this question"
- Be concise and direct
- For temporal questions, pay attention to dates and chronological order
- For preference questions, focus on user's stated preferences
- DO NOT make up information

Answer:"""
    
    return prompt


def _save_sample_results(output_dir: Path, qa_index: int, score_data: Dict, retrieval_data: Dict):
    """保存样本结果"""
    qa_dir = output_dir / f"QA_{qa_index}"
    qa_dir.mkdir(parents=True, exist_ok=True)
    
    score_file = qa_dir / "score.json"
    with open(score_file, 'w', encoding='utf-8') as f:
        json.dump(score_data, f, indent=2, ensure_ascii=False)
    
    retrieval_file = qa_dir / "retrieval.json"
    with open(retrieval_file, 'w', encoding='utf-8') as f:
        json.dump(retrieval_data, f, indent=2, ensure_ascii=False)


# ============================================================
# 并行 Benchmark 主类
# ============================================================

class LongMemEvalBenchmarkParallel:
    """
    LongMemEval 基准测试类（Processed 版本 - 并行）
    
    【并行策略】：
    1. 使用 ProcessPoolExecutor 进行多进程并行
    2. 每个 worker 独立初始化 mem0 和 LLM 客户端
    3. 使用带 worker_id 的 user_id 确保记忆隔离
    4. 支持控制最大并发数
    """
    
    def __init__(
        self,
        dataset_path: str,
        gen_llm_model: str = "gpt-4o-mini-closeai",
        eval_llm_model: str = "gpt-4o-mini-closeai",
        user_id_base: str = "benchmark_processed_parallel",
        infer: bool = True,
        output_dir: str = "benchmark_results_processed",
        num_workers: int = 4,
        max_concurrent_llm: int = 8
    ):
        """
        初始化并行 Benchmark
        
        Args:
            dataset_path: 数据集路径
            gen_llm_model: 生成答案的 LLM 模型名称
            eval_llm_model: 评估答案的 LLM 模型名称
            user_id_base: user_id 基础名称
            infer: 是否启用 mem0 的推理功能
            output_dir: 输出目录
            num_workers: 并行 worker 数量
            max_concurrent_llm: 最大 LLM API 并发数（预留）
        """
        self.dataset_path = dataset_path
        self.gen_llm_model = gen_llm_model
        self.eval_llm_model = eval_llm_model
        self.user_id_base = user_id_base
        self.infer = infer
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.max_concurrent_llm = max_concurrent_llm
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("初始化 LongMemEval Benchmark（Processed 并行版本）")
        logger.info("="*80)
        logger.info(f"数据集: {dataset_path}")
        logger.info(f"生成 LLM: {gen_llm_model}")
        logger.info(f"评估 LLM: {eval_llm_model}")
        logger.info(f"🔥 Infer 模式: {infer}")
        logger.info(f"并行 Workers: {num_workers}")
        logger.info(f"输出目录: {output_dir}")
        logger.info("="*80)
    
    def check_existing_results(self, start_index: int, end_index: int) -> List[int]:
        """检查已存在的结果"""
        completed = []
        for idx in range(start_index, end_index + 1):
            qa_dir = self.output_dir / f"QA_{idx}"
            score_file = qa_dir / "score.json"
            retrieval_file = qa_dir / "retrieval.json"
            
            if score_file.exists() and retrieval_file.exists():
                completed.append(idx)
        
        return completed
    
    def run_benchmark(
        self,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        query_top_k: int = 5,
        skip_existing: bool = True,
        save_summary: bool = True
    ) -> Dict[str, Any]:
        """
        运行并行基准测试
        
        Args:
            start_index: 开始索引
            end_index: 结束索引
            query_top_k: 检索 top-k
            skip_existing: 是否跳过已存在结果
            save_summary: 是否保存汇总
            
        Returns:
            测试结果汇总
        """
        # 延迟导入
        from task_eval.load_dataset_processed import load_dataset
        
        logger.info("\n加载数据集...")
        all_samples = load_dataset(self.dataset_path)
        total_samples_in_dataset = len(all_samples)
        logger.info(f"数据集共 {total_samples_in_dataset} 个样本")
        
        # 确定索引范围
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = total_samples_in_dataset - 1
        
        start_index = max(0, start_index)
        end_index = min(total_samples_in_dataset - 1, end_index)
        
        # 检查已完成的结果
        completed = []
        if skip_existing:
            completed = self.check_existing_results(start_index, end_index)
            if completed:
                logger.info(f"发现 {len(completed)} 个已完成的样本，将跳过")
        
        # 准备待处理的任务
        tasks = []
        for idx in range(start_index, end_index + 1):
            if idx in completed:
                continue
            sample = all_samples[idx]
            original_idx = sample.get('sample_index', idx)
            tasks.append((sample, original_idx, query_top_k))
        
        logger.info(f"将并行处理 {len(tasks)} 个样本（{self.num_workers} workers）")
        
        if not tasks:
            logger.info("所有样本已完成，无需处理")
            return {'status': 'all_completed', 'completed_count': len(completed)}
        
        # 🔥 使用进程池并行处理
        start_time = datetime.now()
        all_results = []
        
        # 创建进程池
        # 注意：使用 spawn 方式启动进程，确保每个进程独立初始化
        ctx = mp.get_context('spawn')
        
        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=ctx,
            initializer=init_worker,
            initargs=(
                0,  # worker_id 会在任务分配时动态设置
                self.dataset_path,
                self.gen_llm_model,
                self.eval_llm_model,
                self.user_id_base,
                self.infer,
                str(self.output_dir)
            )
        ) as executor:
            # 提交所有任务
            future_to_idx = {}
            for i, task in enumerate(tasks):
                # 更新 worker_id（通过任务索引模运算分配）
                future = executor.submit(process_single_sample_worker, task)
                future_to_idx[future] = task[1]  # sample_idx
            
            # 使用 tqdm 显示进度
            with tqdm(total=len(tasks), desc="并行处理") as pbar:
                for future in as_completed(future_to_idx):
                    sample_idx = future_to_idx[future]
                    try:
                        result = future.result(timeout=600)  # 10分钟超时
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"[QA_{sample_idx}] 任务执行失败: {e}")
                        all_results.append({
                            'sample_idx': sample_idx,
                            'status': 'failed',
                            'error': str(e)
                        })
                    pbar.update(1)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # 统计结果
        successful = [r for r in all_results if r.get('status') == 'success']
        failed = [r for r in all_results if r.get('status') == 'failed']
        
        avg_metrics = self._calculate_average_metrics(successful)
        
        summary = {
            'benchmark_info': {
                'dataset_path': self.dataset_path,
                'gen_llm_model': self.gen_llm_model,
                'eval_llm_model': self.eval_llm_model,
                'user_id_base': self.user_id_base,
                'infer': self.infer,
                'mode': 'processed_parallel',
                'num_workers': self.num_workers,
                'query_top_k': query_top_k,
                'start_index': start_index,
                'end_index': end_index,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_time_seconds': total_time
            },
            'statistics': {
                'total_in_range': end_index - start_index + 1,
                'skipped': len(completed),
                'processed': len(tasks),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(tasks) if tasks else 0,
                'avg_time_per_sample': total_time / len(tasks) if tasks else 0,
                'parallelism_speedup': f"{self.num_workers}x (ideal)"
            },
            'average_metrics': avg_metrics,
            'failed_indices': [r.get('sample_idx') for r in failed]
        }
        
        self._print_summary(summary)
        
        if save_summary:
            self._save_summary(summary)
        
        return summary
    
    def _calculate_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算平均指标"""
        if not results:
            return {}
        
        metrics = {}
        metric_names = ['exact_match', 'f1', 'semantic_similarity', 'rouge_1', 'rouge_l']
        
        for metric_name in metric_names:
            values = []
            for r in results:
                if 'evaluation' in r and 'scores' in r['evaluation']:
                    scores = r['evaluation']['scores']
                    if metric_name.startswith('rouge_'):
                        rouge_key = metric_name.replace('rouge_', 'rouge-')
                        if 'rouge' in scores and rouge_key in scores.get('rouge', {}):
                            value = scores['rouge'][rouge_key]
                            if value is not None:
                                values.append(value)
                    else:
                        value = scores.get(metric_name)
                        if value is not None:
                            values.append(value)
            
            if values:
                metrics[f'avg_{metric_name}'] = sum(values) / len(values)
        
        memory_counts = [r.get('retrieved_memories_count', 0) for r in results]
        if memory_counts:
            metrics['avg_retrieved_memories'] = sum(memory_counts) / len(memory_counts)
        
        # 计算平均耗时
        times = [r.get('timing', {}).get('total_time', 0) for r in results]
        if times:
            metrics['avg_total_time'] = sum(times) / len(times)
        
        return metrics
    
    def _print_summary(self, summary: Dict[str, Any]):
        """打印测试摘要"""
        stats = summary['statistics']
        metrics = summary['average_metrics']
        info = summary['benchmark_info']
        
        print("\n" + "="*80)
        print("📊 测试摘要（Processed 并行模式）")
        print("="*80)
        print(f"处理范围: QA_{info['start_index']} ~ QA_{info['end_index']}")
        print(f"🔥 Infer 模式: {info['infer']}")
        print(f"🚀 并行 Workers: {info['num_workers']}")
        print(f"总样本数: {stats['processed']} (跳过 {stats['skipped']})")
        print(f"成功: {stats['successful']} | 失败: {stats['failed']}")
        print(f"成功率: {stats['success_rate']:.2%}")
        print(f"总耗时: {stats['avg_time_per_sample']*stats['processed']:.2f}秒")
        print(f"平均耗时: {stats['avg_time_per_sample']:.2f}秒/样本")
        
        if summary['failed_indices']:
            print(f"\n❌ 失败的样本: {summary['failed_indices'][:20]}{'...' if len(summary['failed_indices']) > 20 else ''}")
        
        print("\n📈 平均指标:")
        for metric_name, value in metrics.items():
            if metric_name.startswith('avg_'):
                display_name = metric_name.replace('avg_', '').replace('_', ' ').title()
                if 'memories' in metric_name or 'time' in metric_name:
                    print(f"  {display_name}: {value:.2f}")
                else:
                    print(f"  {display_name}: {value:.4f}")
        
        print(f"\n📁 结果保存在: {self.output_dir}")
        print("="*80)
    
    def _save_summary(self, summary: Dict[str, Any]):
        """保存汇总结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_model_name = self.gen_llm_model.replace(':', '_').replace('/', '_')
        eval_model_name = self.eval_llm_model.replace(':', '_').replace('/', '_')
        
        start_idx = summary['benchmark_info']['start_index']
        end_idx = summary['benchmark_info']['end_index']
        num_workers = summary['benchmark_info']['num_workers']
        
        summary_file = self.output_dir / f"summary_processed_parallel_{num_workers}w_gen_{gen_model_name}_eval_{eval_model_name}_QA{start_idx}-{end_idx}_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 汇总结果已保存到: {summary_file}")


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='LongMemEval Benchmark 测试（Processed 并行版本）'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='benchmark_longmemeval/dataset/LongMemEval/longmemeval_s_cleaned.json',
        help='数据集路径'
    )
    parser.add_argument(
        '--gen-model',
        type=str,
        default='gpt-4o-mini-closeai',
        help='生成答案的 LLM 模型名称'
    )
    parser.add_argument(
        '--eval-model',
        type=str,
        default='gpt-4o-mini-closeai',
        help='评估答案的 LLM 模型名称'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=None,
        help='开始的 QA 索引（包含）'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='结束的 QA 索引（包含）'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='检索返回的记忆数量'
    )
    parser.add_argument(
        '--no-infer',
        action='store_true',
        help='禁用 mem0 的推理功能（默认启用）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_longmemeval/benchmark_results_processed',
        help='输出目录'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='并行 worker 数量（默认 5）'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='不跳过已存在的结果（覆盖模式）'
    )
    
    args = parser.parse_args()
    
    infer = not args.no_infer
    
    benchmark = LongMemEvalBenchmarkParallel(
        dataset_path=args.dataset,
        gen_llm_model=args.gen_model,
        eval_llm_model=args.eval_model,
        user_id_base='benchmark_processed_parallel',
        infer=infer,
        output_dir=args.output_dir,
        num_workers=args.workers
    )
    
    results = benchmark.run_benchmark(
        start_index=args.start,
        end_index=args.end,
        query_top_k=args.top_k,
        skip_existing=not args.no_skip,
        save_summary=True
    )
    
    return results


if __name__ == "__main__":
    # 设置 multiprocessing 启动方式
    mp.set_start_method('spawn', force=True)
    main()
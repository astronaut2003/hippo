"""
LongMemEval Benchmark 测试脚本（Processed 版本）

使用 mem0 记忆系统和 LLM 进行问答评估
逐条保存结果到 benchmark_results/QA_X/ 目录

【与 unprocessed 版本的区别】：
- 默认 infer=True，启用 mem0 的记忆推理功能（提取和更新记忆）
- mem0 会对对话历史进行处理，提取关键信息形成结构化记忆
- 适用于需要记忆压缩和推理的场景
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 🔥 使用 processed 版本的加载器
from task_eval.load_dataset_processed import LongMemEvalLoader, load_dataset
from task_eval.llm_client import LLMClient
from task_eval.evaluation import calculate_comprehensive_scores

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LongMemEvalBenchmark:
    """
    LongMemEval 基准测试类（Processed 版本）
    
    【与 unprocessed 版本的区别】：
    - 默认 infer=True，启用 mem0 的记忆推理功能
    - mem0 会对对话历史进行处理，提取关键信息形成结构化记忆
    """
    
    def __init__(
        self,
        dataset_path: str,
        gen_llm_model: str = "gpt-4o-mini-closeai",
        eval_llm_model: str = "gpt-4o-mini-closeai",
        user_id_base: str = "benchmark_processed",
        infer: bool = True,  # 🔥 默认 True，启用记忆推理
        output_dir: str = "benchmark_results_processed"
    ):
        """
        初始化 Benchmark
        
        Args:
            dataset_path: 数据集路径
            gen_llm_model: 生成答案的 LLM 模型名称
            eval_llm_model: 评估答案的 LLM 模型名称
            user_id_base: user_id 基础名称
            infer: 是否启用 mem0 的推理功能（默认 True）
            output_dir: 输出目录
        """
        self.dataset_path = dataset_path
        self.gen_llm_model = gen_llm_model
        self.eval_llm_model = eval_llm_model
        self.user_id_base = user_id_base
        self.infer = infer
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化加载器（使用 processed 版本）
        logger.info("初始化 LongMemEval 加载器（Processed 版本）...")
        self.loader = LongMemEvalLoader()
        
        # 初始化生成 LLM 客户端
        logger.info(f"初始化生成 LLM 客户端: {gen_llm_model}")
        self.gen_llm_client = LLMClient(model_name=gen_llm_model)
        
        # 初始化评估 LLM 客户端（如果与生成 LLM 相同则复用）
        if eval_llm_model == gen_llm_model:
            logger.info(f"评估 LLM 与生成 LLM 相同，复用客户端")
            self.eval_llm_client = self.gen_llm_client
        else:
            logger.info(f"初始化评估 LLM 客户端: {eval_llm_model}")
            self.eval_llm_client = LLMClient(model_name=eval_llm_model)
        
        logger.info(f"Benchmark 初始化完成 (infer={self.infer})")
    
    def format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """
        将检索到的记忆格式化为 LLM prompt
        
        Args:
            memories: 记忆列表
            
        Returns:
            格式化后的记忆文本
        """
        if not memories:
            return "No relevant memories found."
        
        formatted_parts = []
        for i, mem in enumerate(memories, 1):
            memory_text = mem.get('memory', '')
            score = mem.get('score', 0)
            rerank_score = mem.get('rerank_score', 0)
            
            formatted_parts.append(
                f"Memory {i} (relevance: {score:.3f}, rerank: {rerank_score:.3f}):\n{memory_text}"
            )
        
        return "\n\n".join(formatted_parts)
    
    def create_qa_prompt(
        self,
        question: str,
        memories: List[Dict[str, Any]],
        question_type: str = "unknown"
    ) -> str:
        """
        创建问答 prompt
        
        Args:
            question: 问题
            memories: 检索到的记忆
            question_type: 问题类型
            
        Returns:
            完整的 prompt
        """
        memories_text = self.format_memories_for_prompt(memories)
        
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
    
    def save_sample_results(
        self,
        qa_index: int,
        score_data: Dict[str, Any],
        retrieval_data: Dict[str, Any]
    ):
        """
        保存单个样本的结果到独立目录
        
        目录结构:
        benchmark_results_processed/
            QA_0/
                score.json      # 评分结果
                retrieval.json  # 检索结果
            QA_1/
                score.json
                retrieval.json
            ...
        
        Args:
            qa_index: QA 样本索引
            score_data: 评分数据
            retrieval_data: 检索数据
        """
        # 创建 QA 目录
        qa_dir = self.output_dir / f"QA_{qa_index}"
        qa_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存评分结果
        score_file = qa_dir / "score.json"
        with open(score_file, 'w', encoding='utf-8') as f:
            json.dump(score_data, f, indent=2, ensure_ascii=False)
        
        # 保存检索结果
        retrieval_file = qa_dir / "retrieval.json"
        with open(retrieval_file, 'w', encoding='utf-8') as f:
            json.dump(retrieval_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"[QA_{qa_index}] 结果已保存到 {qa_dir}")
    
    def process_single_sample(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
        query_top_k: int = 5,
        save_immediately: bool = True
    ) -> Dict[str, Any]:
        """
        处理单个样本
        
        Args:
            sample: 样本数据
            sample_idx: 样本索引
            query_top_k: 检索返回的记忆数量
            save_immediately: 是否立即保存结果
            
        Returns:
            处理结果
        """
        # 记录整体开始时间
        qa_start_time = time.time()
        
        question_id = sample.get('question_id', 'unknown')
        question = sample.get('question', '')
        question_type = sample.get('question_type', 'unknown')
        gold_answer = sample.get('answer', '')
        question_date = sample.get('question_date', '')
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[QA_{sample_idx}] {question_id} (Processed Mode, infer={self.infer})")
        logger.info(f"问题类型: {question_type}")
        logger.info(f"问题: {question}")
        logger.info(f"标准答案: {gold_answer}")
        logger.info(f"{'='*80}")
        
        result = {
            'sample_idx': sample_idx,
            'question_id': question_id,
            'question': question,
            'question_type': question_type,
            'gold_answer': gold_answer,
            'question_date': question_date,
            'user_id': f"{self.user_id_base}_{sample_idx}",
            'infer_mode': self.infer,  # 🔥 记录推理模式
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
            'gen_llm_model': self.gen_llm_model,
            'eval_llm_model': self.eval_llm_model,
            'infer_mode': self.infer,  # 🔥 记录推理模式
            'timestamp': datetime.now().isoformat()
        }
        
        retrieval_data = {
            'sample_idx': sample_idx,
            'question_id': question_id,
            'question': question,
            'query_top_k': query_top_k,
            'infer_mode': self.infer,  # 🔥 记录推理模式
            'timestamp': datetime.now().isoformat()
        }
        
        # 初始化时间统计
        timing_info = {
            'load_time': 0.0,
            'retrieval_time': 0.0,
            'generation_time': 0.0,
            'evaluation_time': 0.0,
            'cleanup_time': 0.0,
            'total_time': 0.0
        }
        
        try:
            # 1. 加载对话历史到记忆系统（🔥 使用 infer=True 进行预处理）
            logger.info(f"[QA_{sample_idx}] 加载对话历史并进行记忆预处理 (infer={self.infer})...")
            load_start = time.time()
            
            load_result = self.loader.load_sample(
                sample=sample,
                sample_idx=sample_idx,
                user_id_base=self.user_id_base,
                infer=self.infer,  # 🔥 默认 True，进行记忆预处理
                clean_before_add=True
            )
            
            load_end = time.time()
            timing_info['load_time'] = round(load_end - load_start, 4)
            
            load_info = {
                'total_sessions': load_result['add_result']['total_sessions'],
                'added_sessions': load_result['add_result']['added_sessions'],
                'failed_sessions': load_result['add_result']['failed_sessions'],
                'infer_mode': load_result['add_result'].get('infer_mode', self.infer)
            }
            
            result['load_result'] = load_info
            retrieval_data['load_result'] = load_info
            
            logger.info(
                f"[QA_{sample_idx}] 加载完成: "
                f"{load_info['added_sessions']}/{load_info['total_sessions']} 个会话 "
                f"(infer={self.infer}, 耗时: {timing_info['load_time']:.2f}s)"
            )
            
            # 2. 检索相关记忆
            logger.info(f"[QA_{sample_idx}] 检索相关记忆...")
            retrieval_start = time.time()
            
            memories = self.loader.search_sample(
                question=question,
                sample_idx=sample_idx,
                user_id_base=self.user_id_base,
                query_top_k=query_top_k
            )
            
            retrieval_end = time.time()
            timing_info['retrieval_time'] = round(retrieval_end - retrieval_start, 4)
            
            result['retrieved_memories_count'] = len(memories)
            result['retrieved_memories'] = memories
            
            # 保存检索结果
            retrieval_data['retrieved_memories_count'] = len(memories)
            retrieval_data['retrieved_memories'] = memories
            
            logger.info(
                f"[QA_{sample_idx}] 检索到 {len(memories)} 条记忆 "
                f"(耗时: {timing_info['retrieval_time']:.2f}s)"
            )
            
            # 3. 使用生成 LLM 生成答案
            logger.info(f"[QA_{sample_idx}] 使用生成 LLM ({self.gen_llm_model}) 生成答案...")
            generation_start = time.time()
            
            prompt = self.create_qa_prompt(question, memories, question_type)
            
            # 计算生成 prompt 的 token 数量
            gen_prompt_tokens = self.gen_llm_client.count_tokens(prompt)
            gen_context_info = self.gen_llm_client.get_context_info()
            
            predicted_answer = self.gen_llm_client.generate_answer(
                prompt=prompt,
                temperature=0.1,
                max_tokens=512
            )
            
            generation_end = time.time()
            timing_info['generation_time'] = round(generation_end - generation_start, 4)
            
            # 计算生成答案的 token 数量
            gen_answer_tokens = self.gen_llm_client.count_tokens(predicted_answer)
            
            # 生成阶段的 token 使用信息
            gen_token_usage = {
                'prompt_tokens': gen_prompt_tokens,
                'answer_tokens': gen_answer_tokens,
                'total_tokens': gen_prompt_tokens + gen_answer_tokens,
                'context_length': gen_context_info.get('context_length', 0),
                'max_context_tokens': gen_context_info.get('max_context_tokens', 0),
                'prompt_ratio': round(gen_prompt_tokens / gen_context_info.get('context_length', 1) * 100, 2),
                'tokenizer_type': 'tiktoken' if gen_context_info.get('tokenizer_available') else 'estimated',
                'encoding': gen_context_info.get('encoding', 'unknown')
            }
            
            result['predicted_answer'] = predicted_answer
            result['prompt_length'] = len(prompt)
            result['gen_token_usage'] = gen_token_usage
            
            score_data['predicted_answer'] = predicted_answer
            score_data['prompt_length'] = len(prompt)
            score_data['gen_token_usage'] = gen_token_usage
            
            logger.info(f"[QA_{sample_idx}] 预测答案: {predicted_answer}")
            logger.info(
                f"[QA_{sample_idx}] 生成 Token 使用: "
                f"prompt={gen_prompt_tokens}, answer={gen_answer_tokens}, total={gen_prompt_tokens + gen_answer_tokens} "
                f"(耗时: {timing_info['generation_time']:.2f}s)"
            )
            
            # 4. 评估答案质量（使用评估 LLM 进行 LLM Judge）
            logger.info(f"[QA_{sample_idx}] 使用评估 LLM ({self.eval_llm_model}) 评估答案质量...")
            evaluation_start = time.time()
            
            try:
                eval_scores = calculate_comprehensive_scores(
                    gold_answer=gold_answer,
                    response=predicted_answer,
                    question=question,
                    question_type=question_type,
                    llm_client=self.eval_llm_client,  # 使用评估 LLM
                    metrics=['exact_match', 'f1', 'rouge', 'semantic_similarity', 'llm_judge']
                )
                
                result['evaluation'] = eval_scores
                score_data['evaluation'] = eval_scores
                score_data['scores'] = eval_scores.get('scores', {})
                
                f1_score = eval_scores.get('scores', {}).get('token_f1', 0)
                llm_accuracy = eval_scores.get('scores', {}).get('llm_accuracy', 0)
                logger.info(f"[QA_{sample_idx}] F1分数: {f1_score:.3f}, LLM Judge: {llm_accuracy}")
                
            except Exception as eval_error:
                logger.warning(f"[QA_{sample_idx}] 评估失败: {eval_error}")
                result['evaluation'] = {'error': str(eval_error)}
                score_data['evaluation'] = {'error': str(eval_error)}
                score_data['scores'] = {}
            
            evaluation_end = time.time()
            timing_info['evaluation_time'] = round(evaluation_end - evaluation_start, 4)
            logger.info(f"[QA_{sample_idx}] 评估耗时: {timing_info['evaluation_time']:.2f}s")
            
            # 5. 清理记忆
            cleanup_start = time.time()
            self.loader.reset_memory(sample_idx=sample_idx, user_id_base=self.user_id_base)
            cleanup_end = time.time()
            timing_info['cleanup_time'] = round(cleanup_end - cleanup_start, 4)
            
            result['status'] = 'success'
            score_data['status'] = 'success'
            retrieval_data['status'] = 'success'
            
        except Exception as e:
            logger.error(f"[QA_{sample_idx}] 处理失败: {e}", exc_info=True)
            result['status'] = 'failed'
            result['error'] = str(e)
            
            score_data['status'] = 'failed'
            score_data['error'] = str(e)
            
            retrieval_data['status'] = 'failed'
            retrieval_data['error'] = str(e)
        
        # 计算总时间
        qa_end_time = time.time()
        timing_info['total_time'] = round(qa_end_time - qa_start_time, 4)
        
        # 添加时间统计到结果中
        result['timing'] = timing_info
        score_data['timing'] = timing_info
        retrieval_data['timing'] = {
            'retrieval_time': timing_info['retrieval_time'],
            'load_time': timing_info['load_time']
        }
        
        logger.info(
            f"[QA_{sample_idx}] 时间统计: "
            f"加载={timing_info['load_time']:.2f}s, "
            f"检索={timing_info['retrieval_time']:.2f}s, "
            f"生成={timing_info['generation_time']:.2f}s, "
            f"评估={timing_info['evaluation_time']:.2f}s, "
            f"总计={timing_info['total_time']:.2f}s"
        )
        
        # 立即保存结果
        if save_immediately:
            self.save_sample_results(sample_idx, score_data, retrieval_data)
        
        return result
    
    def run_benchmark(
        self,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        query_top_k: int = 5,
        save_summary: bool = True
    ) -> Dict[str, Any]:
        """
        运行基准测试
        
        Args:
            start_index: 开始的 QA 索引（包含），None 表示从头开始
            end_index: 结束的 QA 索引（包含），None 表示到末尾
            query_top_k: 检索返回的记忆数量
            save_summary: 是否保存汇总结果
            
        Returns:
            测试结果
        """
        logger.info("="*80)
        logger.info("开始 LongMemEval Benchmark 测试（Processed 版本）")
        logger.info("="*80)
        logger.info(f"数据集: {self.dataset_path}")
        logger.info(f"生成 LLM 模型: {self.gen_llm_model}")
        logger.info(f"评估 LLM 模型: {self.eval_llm_model}")
        logger.info(f"🔥 Infer 模式: {self.infer} (记忆预处理)")
        logger.info(f"检索 Top-K: {query_top_k}")
        logger.info(f"输出目录: {self.output_dir}")
        
        # 加载数据集（先加载全部，再根据索引筛选）
        logger.info("\n加载数据集...")
        all_samples = load_dataset(self.dataset_path)
        total_samples_in_dataset = len(all_samples)
        logger.info(f"数据集共 {total_samples_in_dataset} 个样本")
        
        # 确定索引范围
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = total_samples_in_dataset - 1
        
        # 验证索引范围
        start_index = max(0, start_index)
        end_index = min(total_samples_in_dataset - 1, end_index)
        
        if start_index > end_index:
            raise ValueError(f"无效的索引范围: start_index={start_index}, end_index={end_index}")
        
        logger.info(f"处理范围: QA_{start_index} 到 QA_{end_index}")
        
        # 筛选样本
        samples_to_process = []
        indices_to_process = []
        
        for idx in range(start_index, end_index + 1):
            sample = all_samples[idx]
            # 使用样本自带的索引（如果有），否则使用列表索引
            original_idx = sample.get('sample_index', idx)
            samples_to_process.append(sample)
            indices_to_process.append(original_idx)
        
        logger.info(f"将处理 {len(samples_to_process)} 个样本")
        
        # 处理每个样本
        all_results = []
        start_time = datetime.now()
        
        for i, (sample, sample_idx) in enumerate(tqdm(
            zip(samples_to_process, indices_to_process),
            total=len(samples_to_process),
            desc="处理样本 (Processed)"
        )):
            logger.info(f"\n处理 {i+1}/{len(samples_to_process)} (QA_{sample_idx})")
            
            result = self.process_single_sample(
                sample=sample,
                sample_idx=sample_idx,
                query_top_k=query_top_k,
                save_immediately=True
            )
            
            all_results.append(result)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # 统计结果
        logger.info("\n" + "="*80)
        logger.info("测试完成，统计结果...")
        logger.info("="*80)
        
        successful = [r for r in all_results if r['status'] == 'success']
        failed = [r for r in all_results if r['status'] == 'failed']
        
        # 计算平均指标
        avg_metrics = self._calculate_average_metrics(successful)
        
        summary = {
            'benchmark_info': {
                'dataset_path': self.dataset_path,
                'gen_llm_model': self.gen_llm_model,
                'eval_llm_model': self.eval_llm_model,
                'user_id_base': self.user_id_base,
                'infer': self.infer,
                'mode': 'processed',  # 🔥 标记为 processed 模式
                'query_top_k': query_top_k,
                'start_index': start_index,
                'end_index': end_index,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_time_seconds': total_time
            },
            'statistics': {
                'total_samples_in_dataset': total_samples_in_dataset,
                'samples_processed': len(samples_to_process),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(samples_to_process) if samples_to_process else 0,
                'avg_time_per_sample': total_time / len(samples_to_process) if samples_to_process else 0
            },
            'average_metrics': avg_metrics,
            'processed_indices': indices_to_process,
            'failed_indices': [r['sample_idx'] for r in failed]
        }
        
        # 打印摘要
        self._print_summary(summary)
        
        # 保存汇总结果
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
                    
                    # 处理 rouge 嵌套结构
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
        
        # 计算平均检索记忆数
        memory_counts = [r.get('retrieved_memories_count', 0) for r in results]
        if memory_counts:
            metrics['avg_retrieved_memories'] = sum(memory_counts) / len(memory_counts)
        
        return metrics
    
    def _print_summary(self, summary: Dict[str, Any]):
        """打印测试摘要"""
        stats = summary['statistics']
        metrics = summary['average_metrics']
        info = summary['benchmark_info']
        
        print("\n" + "="*80)
        print("📊 测试摘要（Processed 模式）")
        print("="*80)
        print(f"处理范围: QA_{info['start_index']} ~ QA_{info['end_index']}")
        print(f"🔥 Infer 模式: {info['infer']} (记忆预处理)")
        print(f"总样本数: {stats['samples_processed']}")
        print(f"成功: {stats['successful']} | 失败: {stats['failed']}")
        print(f"成功率: {stats['success_rate']:.2%}")
        print(f"总耗时: {stats['avg_time_per_sample']*stats['samples_processed']:.2f}秒")
        print(f"平均耗时: {stats['avg_time_per_sample']:.2f}秒/样本")
        
        if summary['failed_indices']:
            print(f"\n❌ 失败的样本: {summary['failed_indices']}")
        
        print("\n📈 平均指标:")
        for metric_name, value in metrics.items():
            if metric_name.startswith('avg_'):
                display_name = metric_name.replace('avg_', '').replace('_', ' ').title()
                if 'memories' in metric_name:
                    print(f"  {display_name}: {value:.1f}")
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
        
        # 保存汇总文件（添加 processed 标记）
        summary_file = self.output_dir / f"summary_processed_gen_{gen_model_name}_eval_{eval_model_name}_QA{start_idx}-{end_idx}_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 汇总结果已保存到: {summary_file}")
    
    def check_existing_results(self, start_index: int, end_index: int) -> List[int]:
        """
        检查已存在的结果
        
        Args:
            start_index: 开始索引
            end_index: 结束索引
            
        Returns:
            已完成的索引列表
        """
        completed = []
        for idx in range(start_index, end_index + 1):
            qa_dir = self.output_dir / f"QA_{idx}"
            score_file = qa_dir / "score.json"
            retrieval_file = qa_dir / "retrieval.json"
            
            if score_file.exists() and retrieval_file.exists():
                completed.append(idx)
        
        return completed
    
    def run_benchmark_resume(
        self,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        query_top_k: int = 5,
        skip_existing: bool = True,
        save_summary: bool = True
    ) -> Dict[str, Any]:
        """
        运行基准测试（支持断点续传）
        
        Args:
            start_index: 开始的 QA 索引（包含），None 表示从头开始
            end_index: 结束的 QA 索引（包含），None 表示到末尾
            query_top_k: 检索返回的记忆数量
            skip_existing: 是否跳过已存在的结果
            save_summary: 是否保存汇总结果
            
        Returns:
            测试结果
        """
        logger.info("="*80)
        logger.info("开始 LongMemEval Benchmark 测试（Processed 版本，支持断点续传）")
        logger.info("="*80)
        logger.info(f"生成 LLM 模型: {self.gen_llm_model}")
        logger.info(f"评估 LLM 模型: {self.eval_llm_model}")
        logger.info(f"🔥 Infer 模式: {self.infer} (记忆预处理)")
        
        # 加载数据集
        all_samples = load_dataset(self.dataset_path)
        total_samples_in_dataset = len(all_samples)
        
        # 确定索引范围
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = total_samples_in_dataset - 1
        
        start_index = max(0, start_index)
        end_index = min(total_samples_in_dataset - 1, end_index)
        
        # 检查已完成的结果
        if skip_existing:
            completed = self.check_existing_results(start_index, end_index)
            if completed:
                logger.info(f"发现 {len(completed)} 个已完成的样本，将跳过")
        else:
            completed = []
        
        # 筛选需要处理的样本
        samples_to_process = []
        indices_to_process = []
        
        for idx in range(start_index, end_index + 1):
            if idx in completed:
                continue
            sample = all_samples[idx]
            original_idx = sample.get('sample_index', idx)
            samples_to_process.append(sample)
            indices_to_process.append(original_idx)
        
        logger.info(f"将处理 {len(samples_to_process)} 个样本（跳过 {len(completed)} 个）")
        
        if not samples_to_process:
            logger.info("所有样本已完成，无需处理")
            return {'status': 'all_completed', 'completed_count': len(completed)}
        
        # 处理样本
        all_results = []
        start_time = datetime.now()
        
        for i, (sample, sample_idx) in enumerate(tqdm(
            zip(samples_to_process, indices_to_process),
            total=len(samples_to_process),
            desc="处理样本 (Processed)"
        )):
            result = self.process_single_sample(
                sample=sample,
                sample_idx=sample_idx,
                query_top_k=query_top_k,
                save_immediately=True
            )
            all_results.append(result)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # 统计和保存
        successful = [r for r in all_results if r['status'] == 'success']
        failed = [r for r in all_results if r['status'] == 'failed']
        avg_metrics = self._calculate_average_metrics(successful)
        
        summary = {
            'benchmark_info': {
                'dataset_path': self.dataset_path,
                'gen_llm_model': self.gen_llm_model,
                'eval_llm_model': self.eval_llm_model,
                'infer': self.infer,
                'mode': 'processed',  # 🔥 标记为 processed 模式
                'start_index': start_index,
                'end_index': end_index,
                'query_top_k': query_top_k,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_time_seconds': total_time
            },
            'statistics': {
                'total_in_range': end_index - start_index + 1,
                'skipped': len(completed),
                'processed': len(samples_to_process),
                'successful': len(successful),
                'failed': len(failed)
            },
            'average_metrics': avg_metrics,
            'failed_indices': [r['sample_idx'] for r in failed]
        }
        
        self._print_summary(summary)
        
        if save_summary:
            self._save_summary(summary)
        
        return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LongMemEval Benchmark 测试（Processed 版本，逐条保存）')
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
        help='开始的 QA 索引（包含），不指定则从0开始'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='结束的 QA 索引（包含），不指定则到末尾'
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
        '--resume',
        action='store_true',
        help='断点续传模式，跳过已完成的样本'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='不跳过已存在的结果（覆盖模式）'
    )
    
    args = parser.parse_args()
    
    # 🔥 默认 infer=True，使用 --no-infer 可以禁用
    infer = not args.no_infer
    
    # 创建 benchmark 实例
    benchmark = LongMemEvalBenchmark(
        dataset_path=args.dataset,
        gen_llm_model=args.gen_model,
        eval_llm_model=args.eval_model,
        user_id_base='benchmark_processed',
        infer=infer,  # 🔥 默认 True
        output_dir=args.output_dir
    )
    
    # 运行测试
    if args.resume:
        results = benchmark.run_benchmark_resume(
            start_index=args.start,
            end_index=args.end,
            query_top_k=args.top_k,
            skip_existing=not args.no_skip,
            save_summary=True
        )
    else:
        results = benchmark.run_benchmark(
            start_index=args.start,
            end_index=args.end,
            query_top_k=args.top_k,
            save_summary=True
        )
    
    return results


if __name__ == "__main__":
    main()
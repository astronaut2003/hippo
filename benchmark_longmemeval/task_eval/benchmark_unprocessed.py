"""
LongMemEval Benchmark 测试脚本

使用 mem0 记忆系统和 LLM 进行问答评估
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from task_eval.load_dataset_unprocessed import LongMemEvalLoader, load_dataset
from task_eval.llm_client import LLMClient
from task_eval.evaluation import calculate_comprehensive_scores

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LongMemEvalBenchmark:
    """LongMemEval 基准测试类"""
    
    def __init__(
        self,
        dataset_path: str,
        llm_model: str = "gpt-4o-mini-closeai",
        user_id_base: str = "benchmark",
        infer: bool = False,
        output_dir: str = "benchmark_results"
    ):
        """
        初始化 Benchmark
        
        Args:
            dataset_path: 数据集路径
            llm_model: LLM 模型名称
            user_id_base: user_id 基础名称
            infer: 是否启用 mem0 的推理功能
            output_dir: 输出目录
        """
        self.dataset_path = dataset_path
        self.llm_model = llm_model
        self.user_id_base = user_id_base
        self.infer = infer
        self.output_dir = output_dir
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化加载器和 LLM 客户端
        logger.info("初始化 LongMemEval 加载器...")
        self.loader = LongMemEvalLoader()
        
        logger.info(f"初始化 LLM 客户端: {llm_model}")
        self.llm_client = LLMClient(model_name=llm_model)
        
        logger.info("Benchmark 初始化完成")
    
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
    
    def process_single_sample(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
        query_top_k: int = 5
    ) -> Dict[str, Any]:
        """
        处理单个样本
        
        Args:
            sample: 样本数据
            sample_idx: 样本索引
            query_top_k: 检索返回的记忆数量
            
        Returns:
            处理结果
        """
        question_id = sample.get('question_id', 'unknown')
        question = sample.get('question', '')
        question_type = sample.get('question_type', 'unknown')
        gold_answer = sample.get('answer', '')
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[样本 {sample_idx}] {question_id}")
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
            'user_id': f"{self.user_id_base}_{sample_idx}",
            'status': 'success'
        }
        
        try:
            # 1. 加载对话历史到记忆系统
            logger.info(f"[样本 {sample_idx}] 加载对话历史...")
            load_result = self.loader.load_sample(
                sample=sample,
                sample_idx=sample_idx,
                user_id_base=self.user_id_base,
                infer=self.infer,
                clean_before_add=True
            )
            
            result['load_result'] = {
                'total_sessions': load_result['add_result']['total_sessions'],
                'added_sessions': load_result['add_result']['added_sessions'],
                'failed_sessions': load_result['add_result']['failed_sessions']
            }
            
            logger.info(
                f"[样本 {sample_idx}] 加载完成: "
                f"{load_result['add_result']['added_sessions']}/{load_result['add_result']['total_sessions']} 个会话"
            )
            
            # 2. 检索相关记忆
            logger.info(f"[样本 {sample_idx}] 检索相关记忆...")
            memories = self.loader.search_sample(
                question=question,
                sample_idx=sample_idx,
                user_id_base=self.user_id_base,
                query_top_k=query_top_k
            )
            
            result['retrieved_memories_count'] = len(memories)
            result['retrieved_memories'] = memories
            
            logger.info(f"[样本 {sample_idx}] 检索到 {len(memories)} 条记忆")
            
            # 3. 使用 LLM 生成答案
            logger.info(f"[样本 {sample_idx}] 使用 LLM 生成答案...")
            prompt = self.create_qa_prompt(question, memories, question_type)
            
            predicted_answer = self.llm_client.generate_answer(
                prompt=prompt,
                temperature=0.1,
                max_tokens=512
            )
            
            result['predicted_answer'] = predicted_answer
            result['prompt_length'] = len(prompt)
            
            logger.info(f"[样本 {sample_idx}] 预测答案: {predicted_answer}")
            
            # 4. 评估答案质量
            logger.info(f"[样本 {sample_idx}] 评估答案质量...")
            eval_scores = calculate_comprehensive_scores(
                gold_answer=gold_answer,
                response=predicted_answer,
                question=question,
                question_type=question_type,
                llm_client=self.llm_client,
                metrics=['exact_match', 'f1', 'rouge', 'semantic_similarity']
            )
            
            result['evaluation'] = eval_scores
            
            logger.info(f"[样本 {sample_idx}] F1分数: {eval_scores['scores'].get('f1', 0):.3f}")
            
            # 5. 清理记忆
            self.loader.reset_memory(sample_idx=sample_idx, user_id_base=self.user_id_base)
            
        except Exception as e:
            logger.error(f"[样本 {sample_idx}] 处理失败: {e}", exc_info=True)
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def run_benchmark(
        self,
        sample_indices: Optional[List[int]] = None,
        query_top_k: int = 5,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        运行基准测试
        
        Args:
            sample_indices: 要测试的样本索引列表，None 表示测试所有样本
            query_top_k: 检索返回的记忆数量
            save_results: 是否保存结果
            
        Returns:
            测试结果
        """
        logger.info("="*80)
        logger.info("开始 LongMemEval Benchmark 测试")
        logger.info("="*80)
        logger.info(f"数据集: {self.dataset_path}")
        logger.info(f"LLM 模型: {self.llm_model}")
        logger.info(f"Infer 模式: {self.infer}")
        logger.info(f"检索 Top-K: {query_top_k}")
        
        # 加载数据集
        logger.info("\n加载数据集...")
        samples = load_dataset(self.dataset_path, sample_indices=sample_indices)
        logger.info(f"加载了 {len(samples)} 个样本")
        
        # 处理每个样本
        all_results = []
        start_time = datetime.now()
        
        for idx, sample in enumerate(tqdm(samples, desc="处理样本")):
            # 使用原始索引（如果存在）
            original_idx = sample.get('sample_index', idx)
            
            result = self.process_single_sample(
                sample=sample,
                sample_idx=original_idx,
                query_top_k=query_top_k
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
                'llm_model': self.llm_model,
                'user_id_base': self.user_id_base,
                'infer': self.infer,
                'query_top_k': query_top_k,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_time_seconds': total_time
            },
            'statistics': {
                'total_samples': len(samples),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(samples) if samples else 0,
                'avg_time_per_sample': total_time / len(samples) if samples else 0
            },
            'average_metrics': avg_metrics,
            'detailed_results': all_results
        }
        
        # 打印摘要
        self._print_summary(summary)
        
        # 保存结果
        if save_results:
            self._save_results(summary)
        
        return summary
    
    def _calculate_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算平均指标"""
        if not results:
            return {}
        
        metrics = {}
        metric_names = ['exact_match', 'f1', 'semantic_similarity']
        
        for metric_name in metric_names:
            values = []
            for r in results:
                if 'evaluation' in r and 'scores' in r['evaluation']:
                    value = r['evaluation']['scores'].get(metric_name)
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
        
        print("\n" + "="*80)
        print("📊 测试摘要")
        print("="*80)
        print(f"总样本数: {stats['total_samples']}")
        print(f"成功: {stats['successful']} | 失败: {stats['failed']}")
        print(f"成功率: {stats['success_rate']:.2%}")
        print(f"总耗时: {stats['avg_time_per_sample']*stats['total_samples']:.2f}秒")
        print(f"平均耗时: {stats['avg_time_per_sample']:.2f}秒/样本")
        
        print("\n📈 平均指标:")
        for metric_name, value in metrics.items():
            if metric_name.startswith('avg_'):
                display_name = metric_name.replace('avg_', '').replace('_', ' ').title()
                if 'memories' in metric_name:
                    print(f"  {display_name}: {value:.1f}")
                else:
                    print(f"  {display_name}: {value:.4f}")
        
        print("="*80)
    
    def _save_results(self, summary: Dict[str, Any]):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.llm_model.replace(':', '_').replace('/', '_')
        
        # 保存完整结果
        output_file = Path(self.output_dir) / f"benchmark_{model_name}_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 完整结果已保存到: {output_file}")
        
        # 保存简化版结果（不包含详细记忆）
        summary_lite = {
            'benchmark_info': summary['benchmark_info'],
            'statistics': summary['statistics'],
            'average_metrics': summary['average_metrics'],
            'sample_results': [
                {
                    'sample_idx': r['sample_idx'],
                    'question_id': r['question_id'],
                    'question': r['question'],
                    'gold_answer': r['gold_answer'],
                    'predicted_answer': r.get('predicted_answer', ''),
                    'status': r['status'],
                    'evaluation_scores': r.get('evaluation', {}).get('scores', {})
                }
                for r in summary['detailed_results']
            ]
        }
        
        output_file_lite = Path(self.output_dir) / f"benchmark_{model_name}_{timestamp}_lite.json"
        with open(output_file_lite, 'w', encoding='utf-8') as f:
            json.dump(summary_lite, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 简化结果已保存到: {output_file_lite}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LongMemEval Benchmark 测试')
    parser.add_argument(
        '--dataset',
        type=str,
        default='benchmark_longmemeval/dataset/LongMemEval/extracted_samples_index_1.json',
        help='数据集路径'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini-closeai',
        help='LLM 模型名称'
    )
    parser.add_argument(
        '--indices',
        type=str,
        default=None,
        help='样本索引范围，例如: "0,1,2" 或 "0-10"'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='检索返回的记忆数量'
    )
    parser.add_argument(
        '--infer',
        action='store_true',
        help='是否启用 mem0 的推理功能'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_results',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    # 解析索引范围
    sample_indices = None
    if args.indices:
        if '-' in args.indices:
            # 范围格式: "0-10"
            start, end = map(int, args.indices.split('-'))
            sample_indices = list(range(start, end + 1))
        else:
            # 逗号分隔格式: "0,1,2"
            sample_indices = [int(x.strip()) for x in args.indices.split(',')]
    
    # 创建 benchmark 实例
    benchmark = LongMemEvalBenchmark(
        dataset_path=args.dataset,
        llm_model=args.model,
        user_id_base='benchmark',
        infer=args.infer,
        output_dir=args.output_dir
    )
    
    # 运行测试
    results = benchmark.run_benchmark(
        sample_indices=sample_indices,
        query_top_k=args.top_k,
        save_results=True
    )
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Benchmark 测试成功完成！")
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
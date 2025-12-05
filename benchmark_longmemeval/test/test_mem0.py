import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from mem0 import Memory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-5-nano", 
            "temperature": 0.1, 
            "api_key": os.getenv("CLOSEAI_API_KEY"),
            "base_url": "https://api.openai-proxy.org/v1",
        },
    },
    "reranker": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-reranker-base",
            "device": "cuda",
            "batch_size": 32
        }
    }
}

class Mem0Test:
    """LongMemEval 数据集加载器"""
    
    def __init__(self, memory: Optional[Memory] = None):
        """
        初始化加载器
        
        Args:
            memory: mem0 Memory 实例，如果为 None 则创建新实例
        """
        self.memory = memory if memory is not None else Memory.from_config(config)
        
    
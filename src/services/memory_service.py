"""
mem0 记忆服务封装
提供记忆的增删改查功能
"""
from mem0 import Memory
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryService:
    """记忆管理服务"""
    
    def __init__(self, config: Dict):
        """
        初始化记忆服务
        
        Args:
            config: mem0 配置字典
        """
        try:
            self.memory = Memory.from_config(config)
            logger.info("✅ mem0 记忆服务初始化成功")
        except Exception as e:
            logger.error(f"❌ mem0 初始化失败: {e}")
            raise
    
    async def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        添加记忆
        
        Args:
            content: 对话内容或事实描述
            user_id: 用户ID
            metadata: 额外元数据
        
        Returns:
            添加结果，包含提取的记忆列表
        """
        try:
            # mem0 支持两种输入格式
            if isinstance(content, list):
                messages = content
            else:
                messages = [{"role": "user", "content": content}]
            
            result = self.memory.add(
                messages=messages,
                user_id=user_id,
                metadata=metadata or {}
            )
            
            memory_count = len(result.get('results', []))
            logger.info(f"✅ 添加记忆成功: user={user_id}, memories={memory_count}")
            return result
        except Exception as e:
            logger.error(f"❌ 添加记忆失败: {e}")
            raise
    
    async def search_memory(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        检索记忆
        
        Args:
            query: 搜索查询
            user_id: 用户ID
            limit: 返回数量
            filters: 过滤条件
        
        Returns:
            记忆列表
        """
        try:
            results = self.memory.search(
                query=query,
                user_id=user_id,
                limit=limit,
                filters=filters or {}
            )
            
            logger.info(
                f"🔍 检索记忆: user={user_id}, "
                f"query='{query[:30]}...', results={len(results)}"
            )
            return results
        except Exception as e:
            logger.error(f"❌ 检索记忆失败: {e}")
            return []
    
    async def get_all_memories(self, user_id: str) -> List[Dict]:
        """
        获取用户所有记忆
        
        Args:
            user_id: 用户ID
        
        Returns:
            记忆列表
        """
        try:
            memories = self.memory.get_all(user_id=user_id)
            logger.info(f"📚 获取所有记忆: user={user_id}, count={len(memories)}")
            return memories
        except Exception as e:
            logger.error(f"❌ 获取记忆失败: {e}")
            return []
    
    async def delete_memory(self, memory_id: str):
        """
        删除记忆
        
        Args:
            memory_id: 记忆ID
        """
        try:
            self.memory.delete(memory_id=memory_id)
            logger.info(f"🗑️ 删除记忆: {memory_id}")
        except Exception as e:
            logger.error(f"❌ 删除记忆失败: {e}")
            raise
    
    async def update_memory(self, memory_id: str, content: str):
        """
        更新记忆
        
        Args:
            memory_id: 记忆ID
            content: 新内容
        """
        try:
            self.memory.update(memory_id=memory_id, data=content)
            logger.info(f"✏️ 更新记忆: {memory_id}")
        except Exception as e:
            logger.error(f"❌ 更新记忆失败: {e}")
            raise


# 全局记忆服务实例（延迟初始化）
_memory_service: Optional[MemoryService] = None


def get_memory_service(config: Dict) -> MemoryService:
    """
    获取记忆服务实例（单例模式）
    
    Args:
        config: mem0 配置
    
    Returns:
        MemoryService 实例
    """
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService(config)
    return _memory_service

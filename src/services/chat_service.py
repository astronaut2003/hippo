"""
对话服务
整合记忆检索和 LLM 生成
"""
from typing import List, Dict, AsyncGenerator, Optional
from src.services.memory_service import MemoryService
from src.services.llm_service import LLMService
from src.utils.prompt_templates import SYSTEM_PROMPT
import logging

logger = logging.getLogger(__name__)


class ChatService:
    """对话管理服务"""
    
    def __init__(self, memory_service: MemoryService, llm_service: LLMService):
        """
        初始化对话服务
        
        Args:
            memory_service: 记忆服务实例
            llm_service: LLM 服务实例
        """
        self.memory_service = memory_service
        self.llm_service = llm_service
        logger.info("✅ 对话服务初始化成功")
    
    async def chat_stream(
        self,
        user_input: str,
        user_id: str,
        conversation_id: str,
        history: Optional[List[Dict]] = None
    ) -> AsyncGenerator[str, None]:
        """
        流式对话生成
        
        Args:
            user_input: 用户输入
            user_id: 用户ID
            conversation_id: 会话ID
            history: 对话历史
        
        Yields:
            生成的文本块
        """
        logger.info(f"💬 开始对话: user={user_id}, conv={conversation_id}")
        
        # 1. 检索相关记忆
        relevant_memories = await self.memory_service.search_memory(
            query=user_input,
            user_id=user_id,
            limit=5
        )
        
        # 2. 构造上下文
        memory_context = self._format_memory_context(relevant_memories)
        history_context = self._format_history_context(history or [])
        
        # 3. 构造 prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        if memory_context:
            messages.append({
                "role": "system",
                "content": f"📚 相关记忆:\n{memory_context}"
            })
        
        if history_context:
            messages.append({
                "role": "system",
                "content": f"💭 最近对话:\n{history_context}"
            })
        
        messages.append({"role": "user", "content": user_input})
        
        # 4. 流式生成回答
        full_response = ""
        async for chunk in self.llm_service.chat_stream(messages):
            full_response += chunk
            yield chunk
        
        # 5. 存储新记忆（异步，不阻塞返回）
        try:
            await self.memory_service.add_memory(
                content=f"User: {user_input}\nAssistant: {full_response}",
                user_id=user_id,
                metadata={
                    "conversation_id": conversation_id,
                    "type": "conversation"
                }
            )
        except Exception as e:
            logger.warning(f"⚠️ 存储记忆失败: {e}")
        
        logger.info(f"✅ 对话完成: user={user_id}")
    
    def _format_memory_context(self, memories: List[Dict]) -> str:
        """
        格式化记忆为上下文
        
        Args:
            memories: 记忆列表
        
        Returns:
            格式化后的记忆文本
        """
        if not memories:
            return ""
        
        context_lines = []
        for i, mem in enumerate(memories, 1):
            # mem0 返回的记忆格式可能是 'memory' 或 'text'
            memory_text = mem.get('memory', mem.get('text', ''))
            if memory_text:
                context_lines.append(f"{i}. {memory_text}")
        
        return "\n".join(context_lines)
    
    def _format_history_context(self, history: List[Dict]) -> str:
        """
        格式化对话历史
        
        Args:
            history: 对话历史列表
        
        Returns:
            格式化后的历史文本
        """
        if not history:
            return ""
        
        context_lines = []
        # 只保留最近5轮对话
        for msg in history[-5:]:
            role = "👤 用户" if msg["role"] == "user" else "🤖 Hippo"
            # 截断过长的内容
            content = msg["content"][:100]
            if len(msg["content"]) > 100:
                content += "..."
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)

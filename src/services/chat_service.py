"""
对话服务
整合记忆检索和 LLM 生成，自动保存和加载对话历史
"""
from typing import List, Dict, AsyncGenerator, Optional
from src.services.memory_service import MemoryService
from src.services.llm_service import LLMService
from src.utils.prompt_templates import SYSTEM_PROMPT
import logging
import asyncpg
import os
from urllib.parse import quote_plus

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
    
    async def _get_db_connection(self):
        """获取数据库连接"""
        db_password = os.getenv('POSTGRES_PASSWORD')
        
        return await asyncpg.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            database=os.getenv('POSTGRES_DB', 'hippo'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=db_password
        )
    
    async def _save_message(self, session_id: str, role: str, content: str):
        """
        保存消息到数据库
        
        Args:
            session_id: 会话ID
            role: 角色 (user/assistant)
            content: 消息内容
        """
        try:
            conn = await self._get_db_connection()
            try:
                await conn.execute(
                    """
                    INSERT INTO chat_messages (session_id, role, content)
                    VALUES ($1, $2, $3)
                    """,
                    session_id,
                    role,
                    content
                )
                # 更新会话的 updated_at 时间
                await conn.execute(
                    """
                    UPDATE sessions
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = $1
                    """,
                    session_id
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"⚠️ 保存消息失败: {e}")
    
    async def _fetch_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """
        从数据库获取对话历史
        
        Args:
            session_id: 会话ID
            limit: 获取的消息数量
        
        Returns:
            List[Dict]: 对话历史列表
        """
        try:
            conn = await self._get_db_connection()
            try:
                rows = await conn.fetch(
                    """
                    SELECT role, content
                    FROM chat_messages
                    WHERE session_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    session_id,
                    limit
                )
                # 反转顺序（最旧的在前）
                history = [{"role": row['role'], "content": row['content']} for row in reversed(rows)]
                return history
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"⚠️ 获取历史失败: {e}")
            return []
    
    async def chat_stream(
        self,
        user_input: str,
        user_id: str,
        session_id: str
    ) -> AsyncGenerator[str, None]:
        """
        流式对话生成（自动保存消息到数据库）
        
        Args:
            user_input: 用户输入
            user_id: 用户ID
            session_id: 会话ID
        
        Yields:
            生成的文本块
        """
        logger.info(f"💬 开始对话: user={user_id}, session={session_id}")
        
        # 1. 保存用户消息到数据库
        await self._save_message(session_id, "user", user_input)
        
        # 2. 从数据库获取对话历史
        history = await self._fetch_history(session_id, limit=10)
        
        # 3. 检索相关记忆
        relevant_memories = await self.memory_service.search_memory(
            query=user_input,
            user_id=user_id,
            limit=5
        )
        
        # 4. 构造上下文
        memory_context = self._format_memory_context(relevant_memories)
        history_context = self._format_history_context(history)
        
        # 5. 构造 prompt
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
        
        # 6. 流式生成回答
        full_response = ""
        async for chunk in self.llm_service.chat_stream(messages):
            full_response += chunk
            yield chunk
        
        # 7. 保存助手消息到数据库
        await self._save_message(session_id, "assistant", full_response)
        
        # 8. 存储新记忆（异步，不阻塞返回）
        try:
            await self.memory_service.add_memory(
                content=f"User: {user_input}\nAssistant: {full_response}",
                user_id=user_id,
                metadata={
                    "session_id": session_id,
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

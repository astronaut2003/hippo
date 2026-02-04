"""
聊天 API 路由
处理对话请求
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


# Pydantic 模型
class Message(BaseModel):
    """消息模型"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str
    user_id: str = "default_user"
    session_id: str  # 改为 session_id，使用 UUID 格式


@router.post("/message")
async def chat_message(request: ChatRequest):
    """
    发送消息并获取流式响应
    
    返回 Server-Sent Events (SSE) 格式的流式数据
    
    Args:
        request: 聊天请求
    
    Returns:
        StreamingResponse: SSE 流式响应
    """
    try:
        # 导入服务（延迟导入避免循环依赖）
        from src.main import get_chat_service_instance
        
        chat_service = get_chat_service_instance()
        
        async def generate():
            """生成器函数，产生 SSE 格式的数据"""
            try:
                # 流式生成回答 (不再传递 history，由 ChatService 自动从 DB 获取)
                async for chunk in chat_service.chat_stream(
                    user_input=request.message,
                    user_id=request.user_id,
                    session_id=request.session_id
                ):
                    # SSE 格式: data: {json}\n\n
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                
                # 发送结束标志
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                logger.error(f"生成响应失败: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # 禁用 nginx 缓冲
            }
        )
    
    except Exception as e:
        logger.error(f"聊天接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/welcome/{user_id}")
async def get_welcome_message(user_id: str):
    """获取个性化欢迎消息"""
    try:
        # 导入服务（延迟导入避免循环依赖）
        from src.main import get_memory_service_instance
        
        memory_service = get_memory_service_instance()
        memories = await memory_service.get_all_memories(user_id=user_id)
        
        if len(memories) == 0:
            welcome_text = """👋 你好！我是 **Hippo**，一个具备长期记忆的智能助手！

🦛 **关于我：**
• 我能记住我们的每次对话
• 学习你的偏好和习惯
• 提供越来越个性化的帮助

🌟 **我能帮你：**
• 日常问题解答和建议
• 工作学习相关讨论
• 记住重要信息和偏好
• 基于历史对话提供更好的服务

💬 **开始对话：**
你可以告诉我你的兴趣、工作、习惯等任何想让我记住的信息。我们聊得越多，我就能更好地为你服务！

有什么想聊的吗？"""
        else:
            # 获取最近的一些记忆作为上下文
            recent_memories = memories[:3] if len(memories) >= 3 else memories
            memory_hints = []
            for mem in recent_memories:
                if isinstance(mem, dict):
                    memory_text = mem.get('memory', mem.get('text', ''))
                    if memory_text:
                        memory_hints.append(f"• {memory_text}")
                        
            hints_text = "\n".join(memory_hints) if memory_hints else "我们之前聊过很多有趣的话题"
            
            welcome_text = f"""欢迎回来！👋

我记得我们之前的对话，目前为你保存了 **{len(memories)}** 条记忆：

{hints_text}

今天想继续聊什么呢？我已经了解了你的一些偏好，可以为你提供更个性化的建议！"""

        return {
            "message": welcome_text,
            "is_new_user": len(memories) == 0,
            "memory_count": len(memories),
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"❌ 获取欢迎消息失败: {e}")
        return {
            "message": "👋 你好！我是 Hippo，很高兴见到你！有什么我可以帮你的吗？",
            "is_new_user": True,
            "memory_count": 0,
            "user_id": user_id
        }


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "chat",
        "message": "Chat service is running"
    }

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
    conversation_id: str
    history: Optional[List[Message]] = []


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
        from src.main import chat_service
        
        async def generate():
            """生成器函数，产生 SSE 格式的数据"""
            try:
                # 将 Pydantic 模型转换为字典
                history_list = [msg.dict() for msg in request.history] if request.history else []
                
                # 流式生成回答
                async for chunk in chat_service.chat_stream(
                    user_input=request.message,
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    history=history_list
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


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "chat",
        "message": "Chat service is running"
    }

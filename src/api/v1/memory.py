"""
记忆管理 API 路由
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


# Pydantic 模型
class MemoryAddRequest(BaseModel):
    """添加记忆请求"""
    content: str
    user_id: str = "default_user"
    metadata: Optional[Dict] = None


class MemorySearchRequest(BaseModel):
    """搜索记忆请求"""
    query: str
    user_id: str = "default_user"
    limit: int = 10


@router.post("/add")
async def add_memory(request: MemoryAddRequest):
    """
    添加记忆
    
    Args:
        request: 添加记忆请求
    
    Returns:
        添加结果
    """
    try:
        from src.main import memory_service
        
        result = await memory_service.add_memory(
            content=request.content,
            user_id=request.user_id,
            metadata=request.metadata
        )
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"添加记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_memory(request: MemorySearchRequest):
    """
    检索记忆
    
    Args:
        request: 搜索记忆请求
    
    Returns:
        记忆列表
    """
    try:
        from src.main import memory_service
        
        results = await memory_service.search_memory(
            query=request.query,
            user_id=request.user_id,
            limit=request.limit
        )
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"检索记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all/{user_id}")
async def get_all_memories(user_id: str):
    """
    获取用户所有记忆
    
    Args:
        user_id: 用户ID
    
    Returns:
        记忆列表
    """
    try:
        from src.main import memory_service
        
        memories = await memory_service.get_all_memories(user_id=user_id)
        return {"success": True, "memories": memories}
    except Exception as e:
        logger.error(f"获取记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str):
    """
    删除记忆
    
    Args:
        memory_id: 记忆ID
    
    Returns:
        删除结果
    """
    try:
        from src.main import memory_service
        
        await memory_service.delete_memory(memory_id=memory_id)
        return {"success": True, "message": "记忆已删除"}
    except Exception as e:
        logger.error(f"删除记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "memory",
        "message": "Memory service is running"
    }

"""
会话管理 API 路由
处理多会话系统的 CRUD 操作
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import logging
import asyncpg
import os
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


# Pydantic 模型
class SessionCreate(BaseModel):
    """创建会话请求"""
    user_id: str
    title: Optional[str] = "New Chat"


class SessionResponse(BaseModel):
    """会话响应"""
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    """消息响应"""
    id: int
    session_id: str
    role: str
    content: str
    created_at: datetime


# 数据库连接辅助函数
async def get_db_connection():
    """获取数据库连接"""
    db_password = os.getenv('POSTGRES_PASSWORD')
    encoded_password = quote_plus(db_password)
    
    return await asyncpg.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'hippo'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=db_password
    )


@router.post("", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    """
    创建新会话
    
    Args:
        request: 会话创建请求
    
    Returns:
        SessionResponse: 创建的会话信息
    """
    try:
        conn = await get_db_connection()
        try:
            # Ensure the user exists in `users` table to satisfy FK constraint
            try:
                await conn.execute(
                    """
                    INSERT INTO users (id, username, created_at)
                    VALUES ($1, $2, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    request.user_id,
                    request.user_id
                )
            except Exception:
                # If users table doesn't exist or insert fails, continue and let session insert handle errors
                logger.debug("无法确保 users 表中存在用户，继续尝试创建会话")

            row = await conn.fetchrow(
                """
                INSERT INTO sessions (user_id, title)
                VALUES ($1, $2)
                RETURNING id, user_id, title, created_at, updated_at
                """,
                request.user_id,
                request.title
            )
            
            return SessionResponse(
                id=str(row['id']),
                user_id=row['user_id'],
                title=row['title'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
        finally:
            await conn.close()
    
    except Exception as e:
        logger.error(f"创建会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[SessionResponse])
async def list_sessions(user_id: str, limit: int = 50):
    """
    获取用户的所有会话列表
    
    Args:
        user_id: 用户ID
        limit: 返回数量限制
    
    Returns:
        List[SessionResponse]: 会话列表
    """
    try:
        conn = await get_db_connection()
        try:
            rows = await conn.fetch(
                """
                SELECT id, user_id, title, created_at, updated_at
                FROM sessions
                WHERE user_id = $1
                ORDER BY updated_at DESC
                LIMIT $2
                """,
                user_id,
                limit
            )
            
            return [
                SessionResponse(
                    id=str(row['id']),
                    user_id=row['user_id'],
                    title=row['title'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                for row in rows
            ]
        finally:
            await conn.close()
    
    except Exception as e:
        logger.error(f"获取会话列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/messages", response_model=List[MessageResponse])
async def get_session_messages(session_id: str, limit: int = 100):
    """
    获取指定会话的消息历史
    
    Args:
        session_id: 会话ID
        limit: 返回数量限制
    
    Returns:
        List[MessageResponse]: 消息列表
    """
    try:
        conn = await get_db_connection()
        try:
            rows = await conn.fetch(
                """
                SELECT id, session_id, role, content, created_at
                FROM chat_messages
                WHERE session_id = $1
                ORDER BY created_at ASC
                LIMIT $2
                """,
                session_id,
                limit
            )
            
            return [
                MessageResponse(
                    id=row['id'],
                    session_id=str(row['session_id']),
                    role=row['role'],
                    content=row['content'],
                    created_at=row['created_at']
                )
                for row in rows
            ]
        finally:
            await conn.close()
    
    except Exception as e:
        logger.error(f"获取消息历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """
    删除会话及其所有消息
    
    Args:
        session_id: 会话ID
    
    Returns:
        删除成功的消息
    """
    try:
        conn = await get_db_connection()
        try:
            result = await conn.execute(
                """
                DELETE FROM sessions
                WHERE id = $1
                """,
                session_id
            )
            
            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="会话不存在")
            
            return {"message": "会话已删除"}
        finally:
            await conn.close()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{session_id}")
async def update_session_title(session_id: str, title: str):
    """
    更新会话标题
    
    Args:
        session_id: 会话ID
        title: 新标题
    
    Returns:
        更新后的会话信息
    """
    try:
        conn = await get_db_connection()
        try:
            row = await conn.fetchrow(
                """
                UPDATE sessions
                SET title = $1, updated_at = CURRENT_TIMESTAMP
                WHERE id = $2
                RETURNING id, user_id, title, created_at, updated_at
                """,
                title,
                session_id
            )
            
            if not row:
                raise HTTPException(status_code=404, detail="会话不存在")
            
            return SessionResponse(
                id=str(row['id']),
                user_id=row['user_id'],
                title=row['title'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
        finally:
            await conn.close()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新会话标题失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

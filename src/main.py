"""
Hippo Agent - FastAPI 应用入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.core.config import settings
from src.core.logger import setup_logging
from src.services.memory_service import get_memory_service
from src.services.llm_service import LLMService
from src.services.chat_service import ChatService
from src.api.v1 import chat, memory

# 设置日志
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# 全局服务实例
memory_service = None
llm_service = None
chat_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    启动时初始化服务，关闭时清理资源
    """
    global memory_service, llm_service, chat_service
    
    logger.info("🚀 正在启动 Hippo Agent...")
    
    try:
        # URL 编码数据库密码（处理特殊字符）
        from urllib.parse import quote_plus
        encoded_password = quote_plus(settings.POSTGRES_PASSWORD)
        db_url = f"postgresql://{settings.POSTGRES_USER}:{encoded_password}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        
        # 初始化 mem0 配置
        mem0_config = {
            "vector_store": {
                "provider": settings.MEM0_VECTOR_STORE,
                "config": {
                    "url": db_url,  # 使用完整 URL 避免特殊字符问题
                    "collection_name": settings.MEM0_COLLECTION_NAME,
                    "embedding_model_dims": settings.EMBEDDING_DIMS
                }
            },
            "llm": {
                "provider": "openai",  # DeepSeek 兼容 OpenAI API
                "config": {
                    "model": settings.LLM_MODEL,  # deepseek-chat
                    "api_key": settings.DEEPSEEK_API_KEY or settings.OPENAI_API_KEY,
                    "base_url": settings.LLM_BASE_URL,  # https://api.deepseek.com/v1
                    "temperature": 0.7
                }
            },
            "embedder": {
                "provider": "huggingface",  # 本地 HuggingFace 模型
                "config": {
                    "model": settings.EMBEDDING_MODEL  # BAAI/bge-large-en-v1.5
                }
            }
        }
        
        # 初始化服务
        memory_service = get_memory_service(mem0_config)
        llm_service = LLMService(
            api_key=settings.DEEPSEEK_API_KEY or settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL,
            base_url=settings.LLM_BASE_URL
        )
        chat_service = ChatService(memory_service, llm_service)
        
        logger.info("✅ 所有服务初始化成功")
        logger.info(f"📚 数据库: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")
        logger.info(f"🤖 LLM: {settings.LLM_MODEL} @ {settings.LLM_BASE_URL}")
        logger.info(f"📝 Embedding: {settings.EMBEDDING_MODEL} (本地 HuggingFace)")
        logger.info(f"🤖 LLM 模型: {settings.LLM_MODEL}")
        
    except Exception as e:
        logger.error(f"❌ 服务初始化失败: {e}")
        raise
    
    yield  # 应用运行
    
    # 清理资源
    logger.info("👋 正在关闭 Hippo Agent...")


# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="具备长期记忆能力的智能问答 Agent",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix="/api/v1")
app.include_router(memory.router, prefix="/api/v1")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Welcome to Hippo Agent API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )

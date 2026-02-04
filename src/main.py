"""
Hippo Agent - FastAPI 应用入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

# ✅ 按照 test_mem0.py 的方式加载环境变量
from dotenv import load_dotenv
from urllib.parse import quote_plus

# 加载环境变量（与 test_mem0.py 相同的方式）
backend_dir = Path(__file__).parent.parent  # src/ -> backend/
env_path = backend_dir / '.env'
load_dotenv(dotenv_path=env_path)

from src.core.logger import setup_logging
from src.services.memory_service import get_memory_service
from src.services.llm_service import LLMService
from src.services.chat_service import ChatService
from src.api.v1 import chat, memory, sessions

# 设置日志
setup_logging(os.getenv('LOG_LEVEL', 'INFO'))
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
        # ✅ 按照 test_mem0.py 的方式读取环境变量
        deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        db_password = os.getenv('POSTGRES_PASSWORD')
        
        # 检查必需的环境变量
        if not deepseek_key:
            raise ValueError("❌ 缺少 DEEPSEEK_API_KEY 环境变量")
        if not db_password:
            raise ValueError("❌ 缺少 POSTGRES_PASSWORD 环境变量")
        
        # URL 编码数据库密码（与 test_mem0.py 相同）
        encoded_password = quote_plus(db_password)
        
        logger.info("正在初始化 mem0 服务...")
        logger.info(f"数据库配置: {os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', 5432)}/{os.getenv('POSTGRES_DB', 'hippo')}")
        logger.info(f"用户: {os.getenv('POSTGRES_USER', 'postgres')}")
        logger.info(f"Embedding 维度: {os.getenv('EMBEDDING_DIMS', 384)}")
        
        # ✅ 完全按照 test_mem0.py 的配置格式
        mem0_config = {
            "vector_store": {
                "provider": "pgvector",
                "config": {
                    "dbname": os.getenv('POSTGRES_DB', 'hippo'),
                    "host": os.getenv('POSTGRES_HOST', 'localhost'),
                    "port": int(os.getenv('POSTGRES_PORT', 5432)),
                    "user": os.getenv('POSTGRES_USER', 'postgres'),
                    "password": encoded_password,  # ✅ 使用编码后的密码
                    "embedding_model_dims": int(os.getenv('EMBEDDING_DIMS', 384)),
                    "collection_name": os.getenv('MEM0_COLLECTION_NAME', 'hippo_memories')
                }
            },
            "llm": {
                "provider": "deepseek",  # ✅ 与 test_mem0.py 相同
                "config": {
                    "model": os.getenv('LLM_MODEL', 'deepseek-chat'),
                    "api_key": deepseek_key,
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
                }
            }
        }
        
        # 初始化服务
        logger.info("正在初始化 mem0 服务...")

        
        try:
            # ✅ 使用已导入的函数并传递配置
            memory_service = get_memory_service(mem0_config)
            logger.info("✅ mem0 服务初始化成功")
        except Exception as mem_error:
            logger.error(f"❌ mem0 初始化失败: {mem_error}")
            logger.error(f"错误类型: {type(mem_error).__name__}")
            import traceback
            logger.error(f"详细堆栈:\n{traceback.format_exc()}")
            raise
        
        llm_service = LLMService(
            api_key=deepseek_key,  # ✅ 直接使用读取的 API Key
            model=os.getenv('LLM_MODEL', 'deepseek-chat'),
            base_url=os.getenv('LLM_BASE_URL', 'https://api.deepseek.com/v1')
        )
        chat_service = ChatService(memory_service, llm_service)
        
        logger.info("✅ 所有服务初始化成功")
        logger.info(f"📚 数据库: {os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', 5432)}/{os.getenv('POSTGRES_DB', 'hippo')}")
        logger.info(f"🤖 LLM: {os.getenv('LLM_MODEL', 'deepseek-chat')} @ {os.getenv('LLM_BASE_URL', 'https://api.deepseek.com/v1')}")
        logger.info(f"📝 Embedding: {os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')} (本地 HuggingFace)")
        
    except Exception as e:
        logger.error(f"❌ 服务初始化失败: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        raise
    
    yield  # 应用运行
    
    # 清理资源
    logger.info("👋 正在关闭 Hippo Agent...")


# 创建 FastAPI 应用
app = FastAPI(
    title=os.getenv('APP_NAME', 'Hippo Agent'),
    version=os.getenv('APP_VERSION', '1.0.0'),
    description="具备长期记忆能力的智能问答 Agent",
    lifespan=lifespan
)

# 配置 CORS
import json
try:
    cors_origins = json.loads(os.getenv('CORS_ORIGINS', '["http://localhost:3000","http://localhost:5173",'
                                                        '"http://127.0.0.1:3000"]'))
except:
    cors_origins = ["http://localhost:3000", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix="/api/v1")
app.include_router(memory.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Welcome to Hippo Agent API",
        "version": os.getenv('APP_VERSION', '1.0.0'),
        "docs": "/docs",
        "health": "/health"
    }


# 服务访问函数
def get_memory_service_instance():
    """获取记忆服务实例"""
    global memory_service
    if memory_service is None:
        raise RuntimeError("Memory service not initialized")
    return memory_service


def get_llm_service_instance():
    """获取LLM服务实例"""
    global llm_service
    if llm_service is None:
        raise RuntimeError("LLM service not initialized")
    return llm_service


def get_chat_service_instance():
    """获取聊天服务实例"""
    global chat_service
    if chat_service is None:
        raise RuntimeError("Chat service not initialized")
    return chat_service


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "app": os.getenv('APP_NAME', 'Hippo Agent'),
        "version": os.getenv('APP_VERSION', '1.0.0')
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv('DEBUG', 'True').lower() == 'true'
    )

"""
核心配置模块
管理应用的所有配置项
"""
from pydantic_settings import BaseSettings
from typing import List
import json


class Settings(BaseSettings):
    """应用配置类"""

    # 应用基本信息
    APP_NAME: str = "Hippo Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # 数据库配置
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "hippo"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "20031109@WJX"

    # OpenAI/硅基流动配置
    OPENAI_API_KEY: str = ""
    SILICONFLOW_API_KEY: str = ""
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"
    LLM_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct"
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMS: int = 1024  # BAAI/bge-large-en-v1.5=1024, text-embedding-3-small=1536

    # mem0 配置
    MEM0_VECTOR_STORE: str = "pgvector"
    MEM0_COLLECTION_NAME: str = "hippo_memories"

    # CORS 配置
    CORS_ORIGINS: str = '["http://localhost:3000","http://localhost:5173"]'

    # 安全配置
    SECRET_KEY: str = "your-secret-key-change-in-production"

    # 日志配置
    LOG_LEVEL: str = "INFO"

    @property
    def DATABASE_URL(self) -> str:
        """异步数据库连接 URL"""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def SYNC_DATABASE_URL(self) -> str:
        """同步数据库连接 URL (用于 mem0)"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def CORS_ORIGINS_LIST(self) -> List[str]:
        """解析 CORS 来源列表"""
        try:
            return json.loads(self.CORS_ORIGINS)
        except:
            return ["http://localhost:3000"]

    class Config:
        env_file = ".env"
        case_sensitive = True


# 全局配置实例
settings = Settings()

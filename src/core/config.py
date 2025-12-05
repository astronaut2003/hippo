"""
核心配置模块
管理应用的所有配置项
"""
from pydantic_settings import BaseSettings
from typing import List
import json
from pathlib import Path


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

    # LLM API 配置
    DEEPSEEK_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    SILICONFLOW_API_KEY: str = ""
    
    # LLM 配置
    LLM_MODEL: str = "deepseek-chat"
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    
    # Embedding 配置
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMS: int = 384  # all-MiniLM-L6-v2 的维度

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
        # 使用绝对路径指向 .env 文件
        # 从 src/core/config.py 向上两级到 backend/
        env_file = str(Path(__file__).parent.parent.parent / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True


# 全局配置实例
settings = Settings()

# 打印加载的配置路径（调试用）
if __name__ == "__main__":
    print(f"✅ .env 文件路径: {Settings.Config.env_file}")
    print(f"✅ 文件存在: {Path(Settings.Config.env_file).exists()}")
    print(f"✅ DEEPSEEK_API_KEY: {'已设置' if settings.DEEPSEEK_API_KEY else '未设置'}")
    print(f"✅ POSTGRES_PASSWORD: {'已设置' if settings.POSTGRES_PASSWORD else '未设置'}")

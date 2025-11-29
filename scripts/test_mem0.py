"""
测试 mem0 连接和基本功能
LLM: DeepSeek API
Embedding: HuggingFace 本地模型
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from mem0 import Memory
from urllib.parse import quote_plus

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 明确指定 .env 路径
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

print(f"📁 .env 文件: {env_path}")
print(f"📁 文件存在: {env_path.exists()}\n")


def test_mem0():
    """测试 mem0 功能"""
    print("=" * 60)
    print("测试 mem0 - DeepSeek LLM + HuggingFace Embedding")
    print("=" * 60 + "\n")

    # 检查必需的配置
    deepseek_key = os.getenv('DEEPSEEK_API_KEY')
    db_password = os.getenv('POSTGRES_PASSWORD')

    # URL 编码密码（处理 @ 等特殊字符）
    encoded_password = quote_plus(db_password)

    # mem0 配置
    config = {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "dbname": os.getenv('POSTGRES_DB', 'hippo'),
                "host": os.getenv('POSTGRES_HOST', 'localhost'),
                "port": int(os.getenv('POSTGRES_PORT', 5432)),
                "user": os.getenv('POSTGRES_USER', 'postgres'),
                "password": encoded_password,
                "collection_name": "test_memories"
            }
        },
        "llm": {
            "provider": "deepseek",  # DeepSeek 兼容 OpenAI API
            "config": {
                "model": "deepseek-chat",  # DeepSeek 的聊天模型
                "api_key": deepseek_key,
            }
        },
        "embedder": {
            "provider": "huggingface",  # 本地 HuggingFace 模型
            "config": {
                "model": "all-MiniLM-L6-v2"  # 英文优化的 BGE 模型
            }
        }
    }

    try:
        print("1️⃣ 初始化 mem0...")
        memory = Memory.from_config(config)
        print("✅ mem0 初始化成功\n")

        test_user = "test_user_001"

        print("2️⃣ 添加测试记忆...")
        result = memory.add(
            "我喜欢吃川菜，特别是麻辣火锅和水煮鱼。我还喜欢看电影，最喜欢科幻片。",
            user_id=test_user
        )
        print(f"✅ 添加成功: {result}\n")

        print("3️⃣ 搜索记忆...")
        results = memory.search(
            query="推荐美食",
            user_id=test_user
        )
        print(f"✅ 搜索到 {len(results)} 条记忆:")
        for i, mem in enumerate(results, 1):
            print(f"  {i}. {mem.get('memory', mem.get('text', 'N/A'))}")
        print()

        print("4️⃣ 获取所有记忆...")
        all_memories = memory.get_all(user_id=test_user)
        print(f"✅ 共有 {len(all_memories)} 条记忆\n")

        print("=" * 60)
        print("🎉 所有测试通过！")
        print("💡 LLM: DeepSeek API (高质量、低成本)")
        print("💡 Embedding: HuggingFace 本地模型 (免费、快速)")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        print("\n详细错误:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_mem0()
    sys.exit(0 if success else 1)

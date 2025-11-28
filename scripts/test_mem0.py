"""
测试 mem0 连接和基本功能
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from mem0 import Memory

# 加载环境变量
load_dotenv()


def test_mem0():
    """测试 mem0 功能"""
    print("=" * 60)
    print("测试 mem0 连接和功能")
    print("=" * 60 + "\n")

    # mem0 配置
    config = {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "dbname": os.getenv('POSTGRES_DB', 'hippo'),
                "host": os.getenv('POSTGRES_HOST', 'localhost'),
                "port": int(os.getenv('POSTGRES_PORT', 5432)),
                "user": os.getenv('POSTGRES_USER', 'postgres'),
                "password": os.getenv('POSTGRES_PASSWORD', '20031109@WJX'),
                "collection_name": "test_memories"
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.getenv('LLM_MODEL', 'Qwen/Qwen3-VL-8B-Instruct'),
                "api_key": os.getenv('SILICONFLOW_API_KEY', os.getenv('OPENAI_API_KEY')),
                "base_url": os.getenv('SILICONFLOW_BASE_URL', None)
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5'),
                "api_key": os.getenv('SILICONFLOW_API_KEY', os.getenv('OPENAI_API_KEY')),
                "base_url": os.getenv('SILICONFLOW_BASE_URL', None)
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
            "我喜欢吃川菜，特别是麻辣火锅和水煮鱼",
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
        print("🎉 所有测试通过！mem0 工作正常")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("\n请检查:")
        print("  1. PostgreSQL 是否运行")
        print("  2. pgvector 扩展是否安装")
        print("  3. OpenAI API Key 是否正确")
        print("  4. 网络连接是否正常")
        return False


if __name__ == "__main__":
    success = test_mem0()
    sys.exit(0 if success else 1)

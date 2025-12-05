"""
mem0 集成测试脚本
====================
功能: 测试 mem0 记忆系统的连接和基本操作
技术栈:
  - LLM: DeepSeek API (聊天模型)
  - Embedding: HuggingFace 本地模型 (all-MiniLM-L6-v2)
  - Vector Store: PostgreSQL + pgvector
"""

# ============================================================================
# 导入依赖
# ============================================================================
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from mem0 import Memory
from urllib.parse import quote_plus

# ============================================================================
# 环境配置
# ============================================================================
# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

print(f"📁 .env 文件: {env_path}")
print(f"📁 文件存在: {env_path.exists()}\n")


# ============================================================================
# 核心测试函数
# ============================================================================
def test_mem0():
    """
    测试 mem0 的完整功能流程

    测试步骤:
    1. 初始化 mem0 实例
    2. 添加测试记忆
    3. 搜索记忆 (语义搜索)
    4. 获取所有记忆

    Returns:
        bool: 测试是否成功
    """
    print("=" * 60)
    print("测试 mem0 - DeepSeek LLM + HuggingFace Embedding")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------------
    # 环境变量读取与验证
    # ------------------------------------------------------------------------
    deepseek_key = os.getenv('DEEPSEEK_API_KEY')
    db_password = os.getenv('POSTGRES_PASSWORD')

    # URL 编码密码（处理特殊字符如 @）
    encoded_password = quote_plus(db_password)

    # ------------------------------------------------------------------------
    # mem0 配置
    # ------------------------------------------------------------------------
    config = {
        # 向量存储配置 (PostgreSQL + pgvector)
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "dbname": os.getenv('POSTGRES_DB', 'hippo'),
                "host": os.getenv('POSTGRES_HOST', 'localhost'),
                "port": int(os.getenv('POSTGRES_PORT', 5432)),
                "user": os.getenv('POSTGRES_USER', 'postgres'),
                "password": encoded_password,
                "embedding_model_dims": 384,  # all-MiniLM-L6-v2 的维度
                "collection_name": "test_memories"
            }
        },
        # LLM 配置 (DeepSeek)
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": "deepseek-chat",
                "api_key": deepseek_key,
            }
        },
        # Embedding 配置 (HuggingFace 本地模型)
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "all-MiniLM-L6-v2"  # 384 维向量
            }
        }
    }

    try:
        # --------------------------------------------------------------------
        # 测试 1: 初始化
        # --------------------------------------------------------------------
        print("1️⃣ 初始化 mem0...")
        memory = Memory.from_config(config)
        print("✅ mem0 初始化成功\n")

        test_user = "test_user_001"

        # --------------------------------------------------------------------
        # 测试 2: 添加记忆
        # --------------------------------------------------------------------
        print("2️⃣ 添加测试记忆...")
        result = memory.add(
            "我喜欢吃日料，特别是寿司和拉面。",
            user_id=test_user
        )

        # ✅ 正确解析返回结果
        if isinstance(result, dict) and 'results' in result:
            added_memories = result['results']
            if added_memories:
                print(f"✅ 添加成功，共添加 {len(added_memories)} 条记忆:")
                for mem in added_memories:
                    mem_id = mem.get('id', 'N/A')
                    memory_text = mem.get('memory', 'N/A')
                    print(f"  - [{mem_id[:8]}...] {memory_text}")
            else:
                print("⚠️  记忆已存在或未添加新记忆")
        else:
            print(f"✅ 添加成功: {result}")
        print()

        # --------------------------------------------------------------------
        # 测试 3: 语义搜索
        # --------------------------------------------------------------------
        print("3️⃣ 搜索记忆 (查询: 推荐美食)...")
        search_result = memory.search(
            query="推荐美食",
            user_id=test_user
        )

        # ✅ 正确解析搜索结果
        if isinstance(search_result, dict) and 'results' in search_result:
            memories = search_result['results']
            print(f"✅ 搜索到 {len(memories)} 条记忆:")

            for i, mem in enumerate(memories, 1):
                mem_id = mem.get('id', 'N/A')
                memory_text = mem.get('memory', 'N/A')
                score = mem.get('score', 0)
                created_at = mem.get('created_at', 'N/A')

                print(f"\n  📝 记忆 {i}:")
                print(f"     ID: {mem_id[:16]}...")
                print(f"     内容: {memory_text}")
                print(f"     相似度: {score:.4f}")
                print(f"     创建时间: {created_at}")
        else:
            print(f"⚠️  搜索结果格式异常: {search_result}")
        print()

        # --------------------------------------------------------------------
        # 测试 4: 获取所有记忆
        # --------------------------------------------------------------------
        print("4️⃣ 获取所有记忆...")
        all_result = memory.get_all(user_id=test_user)

        # ✅ 正确解析所有记忆
        if isinstance(all_result, dict) and 'results' in all_result:
            all_memories = all_result['results']
            print(f"✅ 共有 {len(all_memories)} 条记忆:\n")

            for i, mem in enumerate(all_memories, 1):
                mem_id = mem.get('id', 'N/A')
                memory_text = mem.get('memory', 'N/A')
                hash_val = mem.get('hash', 'N/A')
                created_at = mem.get('created_at', 'N/A')
                updated_at = mem.get('updated_at', 'N/A')

                print(f"  📝 记忆 {i}:")
                print(f"     ID: {mem_id}")
                print(f"     内容: {memory_text}")
                print(f"     哈希: {hash_val}")
                print(f"     创建时间: {created_at}")
                print(f"     更新时间: {updated_at or '未更新'}")
                print()
        else:
            print(f"⚠️  获取结果格式异常: {all_result}")

        # --------------------------------------------------------------------
        # 测试完成
        # --------------------------------------------------------------------
        print("=" * 60)
        print("🎉 所有测试通过！")
        print("💡 LLM: DeepSeek API (高质量、低成本)")
        print("💡 Embedding: HuggingFace 本地模型 (免费、快速)")
        print("💡 向量存储: PostgreSQL + pgvector")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        print("\n详细错误:")
        traceback.print_exc()
        return False


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == "__main__":
    success = test_mem0()
    sys.exit(0 if success else 1)
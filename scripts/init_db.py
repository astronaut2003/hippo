"""
数据库初始化 Python 脚本
运行 SQL 初始化脚本
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psycopg2
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def init_database():
    """初始化数据库"""
    # 数据库连接参数
    db_params = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'database': os.getenv('POSTGRES_DB', 'hippo'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD')
    }
    
    if not db_params['password']:
        print("❌ 错误: 未设置 POSTGRES_PASSWORD 环境变量")
        return False
    
    try:
        print(f"📡 正在连接到数据库: {db_params['host']}:{db_params['port']}/{db_params['database']}")
        
        # 连接数据库
        conn = psycopg2.connect(**db_params)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # 读取 SQL 文件
        sql_file = Path(__file__).parent / 'init_db.sql'
        print(f"📄 读取 SQL 文件: {sql_file}")
        
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        # 执行 SQL
        print("🔧 正在执行初始化脚本...")
        cursor.execute(sql_script)
        
        # 验证表创建
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
              AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        print(f"\n✅ 数据库初始化成功！创建了 {len(tables)} 个表:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # 验证 pgvector 扩展
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        if cursor.fetchone():
            print("\n✅ pgvector 扩展已启用")
        else:
            print("\n⚠️ 警告: pgvector 扩展未找到")
        
        cursor.close()
        conn.close()
        
        print("\n🎉 数据库初始化完成！")
        return True
        
    except psycopg2.Error as e:
        print(f"\n❌ 数据库错误: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ SQL 文件未找到: {sql_file}")
        return False
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Hippo Agent - 数据库初始化")
    print("=" * 60 + "\n")
    
    success = init_database()
    
    if success:
        print("\n" + "=" * 60)
        print("下一步:")
        print("  1. 运行: python scripts/test_mem0.py  # 测试 mem0 连接")
        print("  2. 启动后端: uvicorn src.main:app --reload")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("初始化失败，请检查:")
        print("  1. PostgreSQL 服务是否运行")
        print("  2. .env 文件配置是否正确")
        print("  3. 数据库连接信息是否正确")
        print("=" * 60)
        sys.exit(1)

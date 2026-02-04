-- Hippo Agent 数据库初始化脚本
-- PostgreSQL 15+ with pgvector

-- 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 创建用户表
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(50) PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==============================
-- 多会话系统表结构
-- ==============================

-- 创建会话表 (sessions)
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(50) REFERENCES users(id) ON DELETE CASCADE,
    title TEXT DEFAULT 'New Chat',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建聊天消息表 (chat_messages)
CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==============================
-- 兼容旧版：保留 conversations 和 messages 表
-- ==============================

-- 创建会话表
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(50) UNIQUE NOT NULL,
    user_id VARCHAR(50) REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 创建消息表
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(50) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 创建记忆元数据表（补充 mem0 的向量表）
CREATE TABLE IF NOT EXISTS memory_metadata (
    id SERIAL PRIMARY KEY,
    memory_id VARCHAR(100) UNIQUE NOT NULL,
    user_id VARCHAR(50) REFERENCES users(id) ON DELETE CASCADE,
    conversation_id VARCHAR(50) REFERENCES conversations(conversation_id),
    memory_type VARCHAR(50),
    importance_score FLOAT DEFAULT 0.5,
    access_count INT DEFAULT 0,
    last_accessed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 创建索引以提高查询性能
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created ON chat_messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_metadata(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_metadata ON memory_metadata USING GIN (metadata);

-- 插入测试用户
INSERT INTO users (id, username, email) 
VALUES ('test-user-001', 'testuser', 'test@example.com')
ON CONFLICT (id) DO NOTHING;

-- 插入测试会话
INSERT INTO conversations (conversation_id, user_id, title) 
VALUES ('test-conv-001', 'test-user-001', '测试会话')
ON CONFLICT (conversation_id) DO NOTHING;

-- 验证表创建
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_type = 'BASE TABLE'
ORDER BY table_name;

-- 验证 pgvector 扩展
SELECT * FROM pg_extension WHERE extname = 'vector';

COMMENT ON TABLE users IS '用户表';
COMMENT ON TABLE sessions IS '会话表 (新版多会话)';
COMMENT ON TABLE chat_messages IS '聊天消息表 (新版多会话)';
COMMENT ON TABLE conversations IS '会话表 (旧版兼容)';
COMMENT ON TABLE messages IS '消息表 (旧版兼容)';
COMMENT ON TABLE memory_metadata IS '记忆元数据表';

/**
 * TypeScript 类型定义
 */

// 消息类型
export interface Message {
    id: string
    role: 'user' | 'assistant' | 'system'
    content: string
    timestamp: string
}

// 会话类型
export interface Conversation {
    id: string
    title: string
    created_at: string
    updated_at: string
}

// 记忆类型
export interface Memory {
    id: string
    memory: string
    user_id: string
    created_at: string
    metadata?: Record<string, any>
}

// 用户类型
export interface User {
    id: string
    username: string
    email?: string
}

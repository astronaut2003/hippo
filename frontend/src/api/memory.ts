/**
 * 记忆管理相关 API
 */
import apiClient from './index'

export interface Memory {
    id: string
    memory: string
    user_id: string
    created_at: string
    metadata?: Record<string, any>
}

export interface MemoryAddRequest {
    content: string
    user_id?: string
    metadata?: Record<string, any>
}

export interface MemorySearchRequest {
    query: string
    user_id?: string
    limit?: number
}

/**
 * 记忆相关 API
 */
export const memoryAPI = {
    /**
     * 添加记忆
     */
    async addMemory(request: MemoryAddRequest): Promise<any> {
        return apiClient.post('/api/v1/memory/add', request)
    },

    /**
     * 搜索记忆
     */
    async searchMemory(request: MemorySearchRequest): Promise<any> {
        return apiClient.post('/api/v1/memory/search', request)
    },

    /**
     * 获取所有记忆
     */
    async getAllMemories(userId: string): Promise<any> {
        return apiClient.get(`/api/v1/memory/all/${userId}`)
    },

    /**
     * 删除记忆
     */
    async deleteMemory(memoryId: string): Promise<any> {
        return apiClient.delete(`/api/v1/memory/${memoryId}`)
    }
}

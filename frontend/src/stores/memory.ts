/**
 * 记忆状态管理
 */
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { memoryAPI } from '@/api/memory'
import type { Memory } from '@/types'

export const useMemoryStore = defineStore('memory', () => {
    // 状态
    const memories = ref<Memory[]>([])
    const isLoading = ref(false)

    // 从 localStorage 获取与聊天一致的 user id（如果不存在则生成并保存）
    function getUserId(): string {
        let userId = localStorage.getItem('hippo_user_id')
        if (!userId) {
            userId = `user_${Date.now()}_${Math.random().toString(36).substr(2,9)}`
            localStorage.setItem('hippo_user_id', userId)
        }
        return userId
    }

    const currentUserId = ref(getUserId())

    // 获取所有记忆
    async function fetchMemories() {
        isLoading.value = true
        try {
            const response = await memoryAPI.getAllMemories(currentUserId.value)
            if (response.success) {
                memories.value = response.memories || []
            }
        } catch (error) {
            console.error('Fetch memories error:', error)
        } finally {
            isLoading.value = false
        }
    }

    // 搜索记忆
    async function searchMemories(query: string) {
        isLoading.value = true
        try {
            const response = await memoryAPI.searchMemory({
                query,
                user_id: currentUserId.value,
                limit: 20
            })
            if (response.success) {
                memories.value = response.results || []
            }
        } catch (error) {
            console.error('Search memories error:', error)
        } finally {
            isLoading.value = false
        }
    }

    // 添加记忆
    async function addMemory(content: string, metadata?: Record<string, any>) {
        try {
            const response = await memoryAPI.addMemory({
                content,
                user_id: currentUserId.value,
                metadata
            })
            if (response.success) {
                // 重新获取记忆列表
                await fetchMemories()
            }
            return response
        } catch (error) {
            console.error('Add memory error:', error)
            throw error
        }
    }

    // 删除记忆
    async function deleteMemory(memoryId: string) {
        try {
            const response = await memoryAPI.deleteMemory(memoryId)
            if (response.success) {
                // 从列表中移除
                memories.value = memories.value.filter(m => m.id !== memoryId)
            }
            return response
        } catch (error) {
            console.error('Delete memory error:', error)
            throw error
        }
    }

    // 设置用户
    function setUserId(userId: string) {
        currentUserId.value = userId
    }

    return {
        memories,
        isLoading,
        currentUserId,
        fetchMemories,
        searchMemories,
        addMemory,
        deleteMemory,
        setUserId
    }
})

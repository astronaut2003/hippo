/**
 * 聊天状态管理
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { chatAPI } from '@/api/chat'
import type { Message } from '@/types'

export const useChatStore = defineStore('chat', () => {
    // 状态
    const messages = ref<Message[]>([])
    const currentConversationId = ref<string>('default-conversation')
    const isLoading = ref(false)
    const streamingContent = ref('')

    // 计算属性
    const chatHistory = computed(() => {
        return messages.value.map(msg => ({
            role: msg.role,
            content: msg.content
        }))
    })

    // 发送消息
    async function sendMessage(content: string, userId: string = 'default_user') {
        if (!content.trim()) {
            return
        }

        // 添加用户消息
        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: content.trim(),
            timestamp: new Date().toISOString()
        }
        messages.value.push(userMessage)

        // 创建助手消息占位
        const assistantMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: 'assistant',
            content: '',
            timestamp: new Date().toISOString()
        }
        messages.value.push(assistantMessage)

        isLoading.value = true
        streamingContent.value = ''

        try {
            // 流式接收回答
            await chatAPI.sendMessageStream(
                {
                    message: content,
                    user_id: userId,
                    conversation_id: currentConversationId.value,
                    history: chatHistory.value.slice(0, -2) // 不包括刚添加的两条消息
                },
                (chunk: string) => {
                    streamingContent.value += chunk
                    assistantMessage.content = streamingContent.value
                }
            )
        } catch (error) {
            console.error('Send message error:', error)
            assistantMessage.content = '抱歉，发生了错误，请重试。'
        } finally {
            isLoading.value = false
            streamingContent.value = ''
        }
    }

    // 清空消息
    function clearMessages() {
        messages.value = []
    }

    // 设置当前会话
    function setConversation(conversationId: string) {
        currentConversationId.value = conversationId
    }

    return {
        messages,
        currentConversationId,
        isLoading,
        streamingContent,
        chatHistory,
        sendMessage,
        clearMessages,
        setConversation
    }
})

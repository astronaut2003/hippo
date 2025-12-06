/**
 * 聊天相关 API
 */

export interface Message {
    role: 'user' | 'assistant' | 'system'
    content: string
}

export interface ChatRequest {
    message: string
    user_id?: string
    conversation_id: string
    history?: Message[]
}

export interface ChatResponse {
    content?: string
    done?: boolean
    error?: string
}

/**
 * 发送消息并接收流式响应
 */
export const chatAPI = {
    /**
     * 流式发送消息
     */
    async sendMessageStream(
        request: ChatRequest,
        onChunk: (chunk: string) => void
    ): Promise<void> {
        const response = await fetch(
            `${import.meta.env.VITE_API_URL}/api/v1/chat/message`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(request)
            }
        )

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`)
        }

        const reader = response.body?.getReader()
        if (!reader) {
            throw new Error('Response body is not readable')
        }

        const decoder = new TextDecoder()

        try {
            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                const chunk = decoder.decode(value, { stream: true })
                const lines = chunk.split('\n')

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data: ChatResponse = JSON.parse(line.slice(6))

                            if (data.error) {
                                throw new Error(data.error)
                            }

                            if (data.content) {
                                onChunk(data.content)
                            }

                            if (data.done) {
                                return
                            }
                        } catch (e) {
                            // 忽略 JSON 解析错误（可能是不完整的数据块）
                            console.debug('Parse error:', e)
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock()
        }
    }
}

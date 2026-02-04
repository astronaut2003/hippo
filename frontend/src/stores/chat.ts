// frontend/src/stores/chat.ts
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { sendMessage as apiSendMessage } from '../api/chat'
import { createSession, getSessions, getSessionMessages } from '../api/sessions'

function getUserId(): string {
  let userId = localStorage.getItem('hippo_user_id')
  if (!userId) {
    userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    localStorage.setItem('hippo_user_id', userId)
  }
  return userId
}

export const useChatStore = defineStore('chat', () => {
  // 状态
  const messages = ref<any[]>([])
  const isLoading = ref(false)
  const streamingContent = ref('')
  const sessions = ref<any[]>([]) 
  const currentSessionId = ref<string>('')

  // Action: 加载所有会话
  async function loadSessions() {
    try {
      const res: any = await getSessions()
      // axios拦截器已经返回了res.data，所以直接用res
      sessions.value = Array.isArray(res) ? res : []
    } catch (e) {
      console.error('加载会话失败', e)
      sessions.value = []
    }
  }

  // Action: 新建会话
  async function newSession() {
    try {
      const res: any = await createSession()
      const newSession = res // axios拦截器已经返回了res.data
      sessions.value.unshift(newSession)
      currentSessionId.value = newSession.id
      messages.value = []
    } catch (e) {
      console.error('新建会话失败', e)
      throw e
    }
  }

  // Action: 切换会话
  async function switchSession(sessionId: string) {
    if (currentSessionId.value === sessionId) return
    try {
      isLoading.value = true
      currentSessionId.value = sessionId
      const res: any = await getSessionMessages(sessionId)
      // 转换消息格式
      messages.value = Array.isArray(res) ? res.map((msg: any) => ({
        id: msg.id,
        role: msg.role,
        content: msg.content,
        timestamp: msg.created_at
      })) : []
    } catch (e) {
      console.error('切换会话失败', e)
      messages.value = []
    } finally {
      isLoading.value = false
    }
  }

  // Action: 发送消息
  async function sendMessage(content: string) {
    if (!currentSessionId.value) {
      console.error('没有当前会话')
      return
    }

    isLoading.value = true
    streamingContent.value = ''
    
    const userMsg = {
      id: Date.now().toString(),
      role: 'user',
      content: content,
      timestamp: new Date().toISOString()
    }
    messages.value.push(userMsg)

    try {
      const aiMsg = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString()
      }
      messages.value.push(aiMsg)

      await apiSendMessage({
        message: content,
        session_id: currentSessionId.value,
        user_id: getUserId()
      }, (chunk: string) => {
        streamingContent.value += chunk
        messages.value[messages.value.length - 1].content = streamingContent.value
      })
      
    } catch (error) {
      console.error(error)
      // 移除失败的消息
      messages.value.pop()
      messages.value.push({
        id: Date.now().toString(),
        role: 'system',
        content: '发送失败，请检查网络或后端连接'
      })
    } finally {
      isLoading.value = false
      loadSessions()
    }
  }

  return {
    messages,
    isLoading,
    streamingContent,
    sessions,
    currentSessionId,
    loadSessions,
    newSession,
    switchSession,
    sendMessage
  }
})
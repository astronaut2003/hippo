// frontend/src/api/sessions.ts
import request from './index'

// 获取用户ID
function getUserId(): string {
  let userId = localStorage.getItem('hippo_user_id')
  if (!userId) {
    userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    localStorage.setItem('hippo_user_id', userId)
  }
  return userId
}

// 获取会话列表
export function getSessions() {
  const userId = getUserId()
  return request({
    url: `/api/v1/sessions?user_id=${userId}`,
    method: 'get'
  })
}

// 创建新会话
export function createSession() {
  const userId = getUserId()
  return request({
    url: '/api/v1/sessions',
    method: 'post',
    data: {
      user_id: userId,
      title: 'New Chat'
    }
  })
}

// 获取某会话的消息记录
export function getSessionMessages(sessionId: string) {
  return request({
    url: `/api/v1/sessions/${sessionId}/messages`,
    method: 'get'
  })
}
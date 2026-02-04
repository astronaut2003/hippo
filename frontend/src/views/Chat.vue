<template>
  <div class="app-layout">
    <aside class="sidebar">
      <div class="sidebar-header">
        <div class="brand">
          <span class="logo-icon">🦛</span>
          <h1>Hippo</h1>
        </div>
        <el-button type="primary" class="new-chat-btn" @click="handleNewChat">
          <el-icon><Plus /></el-icon> 新建会话
        </el-button>
      </div>

      <div class="session-list">
        <div 
          v-for="session in chatStore.sessions" 
          :key="session.id"
          class="session-item"
          :class="{ active: chatStore.currentSessionId === session.id }"
          @click="handleSwitchSession(session.id)"
        >
          <div class="session-info">
            <span class="session-title">{{ session.title || '新对话' }}</span>
            <span class="session-time">{{ formatTime(session.updated_at || session.created_at) }}</span>
          </div>
          <el-button 
            v-if="chatStore.currentSessionId === session.id"
            class="delete-btn" 
            text 
            circle 
            size="small"
            @click.stop="handleDeleteSession(session.id)"
          >
            <el-icon><Delete /></el-icon>
          </el-button>
        </div>
      </div>

      <div class="sidebar-footer">
        <el-button text class="footer-btn" @click="goToMemory">
          <el-icon><Notebook /></el-icon> 记忆管理
        </el-button>
      </div>
    </aside>

    <main class="main-content">
      <header class="chat-header">
        <div class="current-session-info">
          <h2>{{ currentSessionTitle }}</h2>
        </div>
      </header>

      <div class="messages-container" ref="messagesContainer">
        <div v-if="chatStore.messages.length === 0" class="empty-state">
          <div class="empty-icon">👋</div>
          <h3>你好！我是 Hippo</h3>
          <p>我可以帮你记住任何事情，开始一个新的话题吧！</p>
        </div>

        <div class="messages" v-else>
          <div
            v-for="msg in chatStore.messages"
            :key="msg.id"
            :class="['message-item', `message-${msg.role}`, { 'message-welcome': msg.isWelcome }]"
          >
            <div class="message-avatar">
              <span v-if="msg.role === 'user'">👤</span>
              <span v-else>🦛</span>
            </div>
            <div class="message-content">
              <div class="message-header">
                <span class="message-role">
                  {{ msg.role === 'user' ? '用户' : 'Hippo' }}
                </span>
                <span class="message-time">
                  {{ formatTime(msg.timestamp) }}
                </span>
              </div>
              <div class="message-text">
                <div v-if="msg.role === 'assistant'" v-html="renderMarkdown(msg.content)"></div>
                <div v-else>{{ msg.content }}</div>
              </div>
            </div>
          </div>

          <div v-if="chatStore.isLoading && !chatStore.streamingContent" class="loading-indicator">
            <el-icon class="is-loading"><Loading /></el-icon>
            <span>Hippo 正在思考...</span>
          </div>
        </div>
      </div>

      <div class="input-area">
        <div class="input-box">
          <el-input
            v-model="inputText"
            type="textarea"
            :rows="3"
            placeholder="输入消息... (Enter 发送，Shift+Enter 换行)"
            @keydown.enter="handleKeyDown"
            :disabled="chatStore.isLoading"
          />
          <div class="input-actions">
            <span class="input-hint">{{ inputText.length }} / 2000</span>
            <el-button
              type="primary"
              @click="handleSend"
              :loading="chatStore.isLoading"
              :disabled="!inputText.trim()"
            >
              <el-icon><Promotion /></el-icon> 发送
            </el-button>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick, watch, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useChatStore } from '../stores/chat'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, Notebook, Loading, Promotion, Plus } from '@element-plus/icons-vue'
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import 'highlight.js/styles/github.css'

const router = useRouter()
const chatStore = useChatStore()
const inputText = ref('')
const messagesContainer = ref<HTMLElement>()

// 计算当前会话标题
const currentSessionTitle = computed(() => {
  const current = chatStore.sessions.find(s => s.id === chatStore.currentSessionId)
  return current ? (current.title || '新对话') : 'Hippo Chat'
})

// Markdown 配置 (保持不变)
const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
  highlight: (str, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      try { return hljs.highlight(str, { language: lang }).value } catch (e) {}
    }
    return ''
  }
})

function renderMarkdown(content: string) {
  return md.render(content)
}

function formatTime(timestamp: string | Date) {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  // 简单的时间格式化
  return date.getHours().toString().padStart(2, '0') + ':' + 
         date.getMinutes().toString().padStart(2, '0')
}

// --- 核心交互逻辑 ---

// 新建会话
async function handleNewChat() {
  try {
    await chatStore.newSession()
    ElMessage.success('已创建新会话')
    inputText.value = '' // 清空输入框
  } catch (error) {
    ElMessage.error('创建会话失败')
  }
}

// 切换会话
async function handleSwitchSession(sessionId: string) {
  if (chatStore.currentSessionId === sessionId) return
  try {
    await chatStore.switchSession(sessionId)
    scrollToBottom()
  } catch (error) {
    console.error('切换失败', error)
  }
}

// 删除会话
async function handleDeleteSession(sessionId: string) {
  try {
    await ElMessageBox.confirm('确定要删除这个会话及其所有聊天记录吗？（记忆不会被删除）', '删除确认', {
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      type: 'warning'
    })
    await chatStore.deleteSession(sessionId) 
    ElMessage.success('会话已删除')
  } catch (e) {
    // 取消删除或删除失败
    if (e !== 'cancel') {
      ElMessage.error('删除会话失败')
    }
  }
}

// 发送消息
async function handleSend() {
  if (!inputText.value.trim() || chatStore.isLoading) return

  const content = inputText.value.trim()
  inputText.value = ''

  try {
    // 如果没有当前会话，先创建一个
    if (!chatStore.currentSessionId) {
      await chatStore.newSession()
    }
    await chatStore.sendMessage(content)
    scrollToBottom()
  } catch (error) {
    ElMessage.error('发送消息失败')
    inputText.value = content // 恢复输入内容
  }
}

function handleKeyDown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    handleSend()
  }
}

function goToMemory() {
  router.push('/memory')
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

// 监听滚动
watch(
  () => [chatStore.messages.length, chatStore.streamingContent],
  () => scrollToBottom()
)

// 初始化
onMounted(async () => {
  // 1. 加载所有会话列表
  // 注意：确保你的 store 里实现了 loadSessions
  if (chatStore.loadSessions) {
    await chatStore.loadSessions()
  }
  
  // 2. 如果列表为空，自动创建一个新会话，避免空白
  if (chatStore.sessions.length === 0 && chatStore.newSession) {
    await chatStore.newSession()
  }
  
  scrollToBottom()
})
</script>

<style scoped lang="scss">
/* 布局容器：改为 Flex Row */
.app-layout {
  display: flex;
  height: 100vh;
  background: #f5f7fa;
  overflow: hidden;
}

/* 左侧侧边栏 */
.sidebar {
  width: 260px;
  background: #202123; /* 深色背景，类似 ChatGPT */
  color: #fff;
  display: flex;
  flex-direction: column;
  border-right: 1px solid #444;

  .sidebar-header {
    padding: 20px;
    
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 20px;
      
      .logo-icon { font-size: 24px; }
      h1 { margin: 0; font-size: 20px; color: #fff; }
    }

    .new-chat-btn {
      width: 100%;
      background: #343541;
      border: 1px solid #565869;
      color: #fff;
      &:hover { background: #40414f; }
    }
  }

  .session-list {
    flex: 1;
    overflow-y: auto;
    padding: 0 10px;

    .session-item {
      padding: 12px;
      margin-bottom: 4px;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: background 0.2s;
      
      &:hover { background: #2A2B32; }
      &.active { background: #343541; }

      .session-info {
        display: flex;
        flex-direction: column;
        overflow: hidden;
        
        .session-title {
          font-size: 14px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .session-time {
          font-size: 12px;
          color: #8e8ea0;
          margin-top: 4px;
        }
      }

      .delete-btn {
        opacity: 0;
        color: #8e8ea0;
        &:hover { color: #ff6b6b; background: rgba(255,255,255,0.1); }
      }

      &:hover .delete-btn { opacity: 1; }
    }
  }

  .sidebar-footer {
    padding: 16px;
    border-top: 1px solid #444;
    .footer-btn {
      color: #fff;
      width: 100%;
      justify-content: flex-start;
      &:hover { color: #409eff; }
    }
  }
}

/* 右侧主区域 */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  position: relative;
  background: white;

  .chat-header {
    height: 60px;
    border-bottom: 1px solid #eee;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #fff;
    h2 { margin: 0; font-size: 16px; font-weight: 600; }
  }

  .messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    
    .empty-state {
      text-align: center;
      margin-top: 100px;
      .empty-icon { font-size: 48px; margin-bottom: 20px; }
      h3 { margin-bottom: 10px; color: #333; }
      p { color: #666; }
    }
  }
  
  /* 消息样式保持原样 (省略部分细节，上面代码已包含) ... */
  .messages { max-width: 800px; margin: 0 auto; }
  
  .message-item {
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
    
    .message-avatar {
      width: 36px; height: 36px;
      border-radius: 4px;
      display: flex; align-items: center; justify-content: center;
      background: #f0f2f5;
    }
    
    .message-content {
      flex: 1; 
      .message-header { display: flex; gap: 8px; margin-bottom: 4px; font-size: 12px; color: #999; }
      .message-text {
        padding: 10px 16px;
        border-radius: 8px;
        background: #f4f6f8;
        line-height: 1.6;
      }
    }
    
    &.message-user {
      flex-direction: row-reverse;
      .message-text { background: #95ec69; color: #000; } 
      /* 或者用蓝色风格 */
      .message-text { background: #409eff; color: white; }
    }
  }

  .input-area {
    padding: 24px;
    border-top: 1px solid #eee;
    background: #fff;
    
    .input-box {
      max-width: 800px;
      margin: 0 auto;
      position: relative;
      
      .input-actions {
        display: flex;
        justify-content: space-between;
        margin-top: 8px;
        color: #999;
        font-size: 12px;
      }
    }
  }
}
</style>
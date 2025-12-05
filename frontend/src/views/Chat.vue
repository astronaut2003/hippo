<template>
  <div class="chat-page">
    <!-- 头部 -->
    <div class="header">
      <div class="header-content">
        <div class="logo">
          <span class="icon">🦛</span>
          <h1>Hippo</h1>
        </div>
        <div class="header-actions">
          <el-button text @click="clearChat">
            <el-icon><Delete /></el-icon>
            清空对话
          </el-button>
          <el-button text @click="goToMemory">
            <el-icon><Notebook /></el-icon>
            记忆管理
          </el-button>
        </div>
      </div>
    </div>

    <!-- 消息列表 -->
    <div class="messages-container" ref="messagesContainer">
      <div class="messages">
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

        <!-- 加载中提示 -->
        <div v-if="chatStore.isLoading && !chatStore.streamingContent" class="loading-indicator">
          <el-icon class="is-loading"><Loading /></el-icon>
          <span>Hippo 正在思考...</span>
        </div>
      </div>
    </div>

    <!-- 输入框 -->
    <div class="input-container">
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
          <span class="input-hint">
            {{ inputText.length }} / 2000
          </span>
          <el-button
            type="primary"
            @click="handleSend"
            :loading="chatStore.isLoading"
            :disabled="!inputText.trim()"
          >
            <el-icon><Promotion /></el-icon>
            发送
          </el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useChatStore } from '../stores/chat'
import { getWelcomeMessage } from '../api/chat'
import { ElMessage } from 'element-plus'
import { Delete, Notebook, Loading, Promotion } from '@element-plus/icons-vue'
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import 'highlight.js/styles/github.css'

const router = useRouter()
const chatStore = useChatStore()
const inputText = ref('')
const messagesContainer = ref<HTMLElement>()

// Markdown 渲染器
const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
  highlight: (str, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return hljs.highlight(str, { language: lang }).value
      } catch (e) {
        console.error(e)
      }
    }
    return ''
  }
})

// 渲染 Markdown
function renderMarkdown(content: string): string {
  return md.render(content)
}

// 格式化时间
function formatTime(timestamp: string): string {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now.getTime() - date.getTime()

  if (diff < 60000) {
    return '刚刚'
  } else if (diff < 3600000) {
    return `${Math.floor(diff / 60000)}分钟前`
  } else if (diff < 86400000) {
    return `${Math.floor(diff / 3600000)}小时前`
  } else {
    return date.toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  }
}

// 发送消息
async function handleSend() {
  if (!inputText.value.trim() || chatStore.isLoading) {
    return
  }

  const content = inputText.value.trim()
  inputText.value = ''

  try {
    await chatStore.sendMessage(content)
    scrollToBottom()
  } catch (error) {
    ElMessage.error('发送消息失败，请重试')
  }
}

// 处理键盘事件
function handleKeyDown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    handleSend()
  }
}

// 清空对话
function clearChat() {
  ElMessage.success('对话已清空')
  chatStore.clearMessages()
}

// 跳转到记忆管理
function goToMemory() {
  router.push('/memory')
}

// 滚动到底部
function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

// 监听消息变化，自动滚动
watch(
  () => chatStore.messages.length,
  () => {
    scrollToBottom()
  }
)

// 监听流式内容，自动滚动
watch(
  () => chatStore.streamingContent,
  () => {
    scrollToBottom()
  }
)

onMounted(async () => {
  // 检查是否已有欢迎消息
  if (chatStore.messages.length === 0) {
    try {
      // 生成或获取用户ID
      let userId = localStorage.getItem('hippo_user_id')
      if (!userId) {
        userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        localStorage.setItem('hippo_user_id', userId)
      }
      
      // 获取个性化欢迎消息
      const welcomeData = await getWelcomeMessage(userId)
      
      // 添加欢迎消息
      const welcomeMessage = {
        id: `welcome_${Date.now()}`,
        content: welcomeData.message,
        role: 'assistant' as const,
        timestamp: new Date(),
        conversationId: `welcome_${userId}`,
        isWelcome: true
      }
      
      chatStore.addMessage(welcomeMessage)
      
      // 设置当前对话ID
      chatStore.setConversation(`conversation_${userId}_${Date.now()}`)
      
      // 日志记录
      if (welcomeData.is_new_user) {
        console.log('🎉 新用户访问，显示欢迎信息')
      } else {
        console.log(`📚 老用户回归，已有 ${welcomeData.memory_count} 条记忆`)
      }
      
    } catch (error) {
      console.error('Failed to load welcome message:', error)
      
      // 降级到静态欢迎消息
      const fallbackMessage = {
        id: `welcome_fallback_${Date.now()}`,
        content: "👋 你好！我是 Hippo，很高兴见到你！有什么我可以帮你的吗？",
        role: 'assistant' as const,
        timestamp: new Date(),
        conversationId: 'welcome_fallback',
        isWelcome: true
      }
      
      chatStore.addMessage(fallbackMessage)
    }
  }
  
  scrollToBottom()
})
</script>

<style scoped lang="scss">
.chat-page {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #f5f7fa;
}

.header {
  background: white;
  border-bottom: 1px solid #e4e7ed;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);

  .header-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 12px;

    .icon {
      font-size: 32px;
    }

    h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 600;
      color: #303133;
    }
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px;

  .messages {
    max-width: 900px;
    margin: 0 auto;
  }
}

.message-item {
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
  animation: fadeIn 0.3s ease-in;

  .message-avatar {
    flex-shrink: 0;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    background: #f0f2f5;
  }

  .message-content {
    flex: 1;
    min-width: 0;
  }

  .message-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;

    .message-role {
      font-weight: 600;
      font-size: 14px;
      color: #303133;
    }

    .message-time {
      font-size: 12px;
      color: #909399;
    }
  }

  .message-text {
    background: white;
    padding: 12px 16px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
    line-height: 1.6;
    color: #303133;
    word-wrap: break-word;

    :deep(pre) {
      background: #f6f8fa;
      padding: 12px;
      border-radius: 4px;
      overflow-x: auto;

      code {
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 14px;
      }
    }

    :deep(code) {
      background: #f6f8fa;
      padding: 2px 6px;
      border-radius: 3px;
      font-family: 'Consolas', 'Monaco', monospace;
      font-size: 14px;
    }

    :deep(p) {
      margin: 8px 0;
      &:first-child {
        margin-top: 0;
      }
      &:last-child {
        margin-bottom: 0;
      }
    }
  }

  // 欢迎消息特殊样式
  &.message-welcome {
    .message-content .message-text {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border-radius: 15px;
      padding: 20px;
      box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
      
      :deep(strong) {
        color: #ffd700;
      }
      
      :deep(h1), :deep(h2), :deep(h3), :deep(h4) {
        color: #fff;
        margin-top: 16px;
        margin-bottom: 8px;
      }
      
      :deep(ul), :deep(ol) {
        margin: 12px 0;
        padding-left: 20px;
      }
      
      :deep(li) {
        margin-bottom: 4px;
      }
      
      :deep(p) {
        margin: 8px 0;
      }
    }
    
    .message-avatar {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      font-size: 20px;
    }
  }

  &.message-user {
    flex-direction: row-reverse;

    .message-text {
      background: #409eff;
      color: white;
    }
  }
}

.loading-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #909399;
  font-size: 14px;
  padding: 12px;

  .el-icon {
    font-size: 18px;
  }
}

.input-container {
  background: white;
  border-top: 1px solid #e4e7ed;
  padding: 16px 20px;

  .input-box {
    max-width: 900px;
    margin: 0 auto;
  }

  .input-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 8px;

    .input-hint {
      font-size: 12px;
      color: #909399;
    }
  }
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>

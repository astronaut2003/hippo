# 前端开发说明

## 📝 注意事项

由于项目规模较大，前端代码框架已搭建完成，但完整的 Vue 组件实现需要您根据以下指南补充。

## 🏗️ 已创建的文件

- ✅ `package.json` - 项目依赖配置
- ✅ `tsconfig.json` - TypeScript 配置  
- ✅ `vite.config.ts` - Vite 构建配置
- ✅ `.env.development` - 环境变量
- ✅ `index.html` - HTML 入口

## 📋 需要创建的核心文件

### 1. 应用入口 (`src/main.ts`)

```typescript
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import router from './router'

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.use(ElementPlus)
app.mount('#app')
```

### 2. 根组件 (`src/App.vue`)

```vue
<template>
  <div id="app">
    <router-view />
  </div>
</template>

<style>
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
</style>
```

### 3. 路由配置 (`src/router/index.ts`)

```typescript
import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'Home',
      component: () => import('@/views/Chat.vue')
    }
  ]
})

export default router
```

### 4. API 配置 (`src/api/index.ts`)

```typescript
import axios from 'axios'

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 30000
})

export default apiClient
```

### 5. 聊天 API (`src/api/chat.ts`)

```typescript
export const chatAPI = {
  async sendMessageStream(
    request: any,
    onChunk: (chunk: string) => void
  ): Promise<void> {
    const response = await fetch(
      `${import.meta.env.VITE_API_URL}/api/v1/chat/message`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      }
    )

    const reader = response.body?.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader!.read()
      if (done) break

      const chunk = decoder.decode(value)
      const lines = chunk.split('\n')

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6))
          if (data.content) {
            onChunk(data.content)
          }
        }
      }
    }
  }
}
```

### 6. 聊天 Store (`src/stores/chat.ts`)

```typescript
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { chatAPI } from '@/api/chat'

export const useChatStore = defineStore('chat', () => {
  const messages = ref<any[]>([])
  const isLoading = ref(false)

  async function sendMessage(content: string) {
    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date().toISOString()
    }
    messages.value.push(userMessage)

    const assistantMessage = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString()
    }
    messages.value.push(assistantMessage)

    isLoading.value = true

    try {
      await chatAPI.sendMessageStream(
        {
          message: content,
          user_id: 'default_user',
          conversation_id: 'default_conv',
          history: messages.value.slice(0, -2)
        },
        (chunk: string) => {
          assistantMessage.content += chunk
        }
      )
    } catch (error) {
      console.error(error)
      assistantMessage.content = '抱歉，发生了错误'
    } finally {
      isLoading.value = false
    }
  }

  return { messages, isLoading, sendMessage }
})
```

### 7. 聊天页面 (`src/views/Chat.vue`)

```vue
<template>
  <div class="chat-container">
    <div class="header">
      <h1>🦛 Hippo - 智能记忆助手</h1>
    </div>

    <div class="messages">
      <div
        v-for="msg in chatStore.messages"
        :key="msg.id"
        :class="['message', msg.role]"
      >
        <div class="avatar">
          {{ msg.role === 'user' ? '👤' : '🦛' }}
        </div>
        <div class="content">
          <div class="role">{{ msg.role === 'user' ? '用户' : 'Hippo' }}</div>
          <div class="text">{{ msg.content }}</div>
        </div>
      </div>
    </div>

    <div class="input-box">
      <el-input
        v-model="inputText"
        placeholder="输入消息..."
        @keyup.enter="handleSend"
        :disabled="chatStore.isLoading"
      />
      <el-button
        type="primary"
        @click="handleSend"
        :loading="chatStore.isLoading"
      >
        发送
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useChatStore } from '@/stores/chat'

const chatStore = useChatStore()
const inputText = ref('')

const handleSend = () => {
  if (!inputText.value.trim() || chatStore.isLoading) return
  chatStore.sendMessage(inputText.value)
  inputText.value = ''
}
</script>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.header {
  padding: 20px;
  border-bottom: 1px solid #eee;
  text-align: center;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.message {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
}

.avatar {
  font-size: 32px;
}

.content {
  flex: 1;
}

.role {
  font-weight: bold;
  margin-bottom: 5px;
}

.text {
  background: #f5f5f5;
  padding: 10px;
  border-radius: 8px;
}

.message.user .text {
  background: #409eff;
  color: white;
}

.input-box {
  display: flex;
  gap: 10px;
  padding: 20px;
  border-top: 1px solid #eee;
}
</style>
```

## 🚀 启动前端

```bash
cd frontend

# 首次运行：安装依赖
npm install

# 启动开发服务器
npm run dev
```

访问: http://localhost:3000

## 📚 扩展功能

完整实现需要添加:

1. **记忆管理页面** - 查看、搜索、删除记忆
2. **Markdown 渲染** - 使用 markdown-it 渲染助手回复
3. **代码高亮** - 使用 highlight.js 高亮代码块
4. **会话管理** - 创建、切换、删除会话
5. **用户设置** - 配置 API Key、模型参数等

这些功能可以根据需要逐步添加。

## 💡 提示

1. 先确保后端正常运行
2. 测试 API 连接: http://localhost:8000/docs
3. 查看浏览器控制台的错误信息
4. 使用 Vue DevTools 调试组件状态

---

以上是前端的基础实现方案，可根据实际需求调整和扩展。

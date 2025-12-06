<template>
  <div class="memory-page">
    <!-- 头部 -->
    <div class="header">
      <div class="header-content">
        <div class="title">
          <el-button text @click="goBack">
            <el-icon><ArrowLeft /></el-icon>
          </el-button>
          <h1>📚 记忆管理</h1>
        </div>
        <div class="header-actions">
          <el-button @click="refreshMemories" :loading="memoryStore.isLoading">
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </div>
    </div>

    <!-- 搜索栏 -->
    <div class="search-bar">
      <div class="search-content">
        <el-input
          v-model="searchQuery"
          placeholder="搜索记忆..."
          clearable
          @input="handleSearch"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>
      </div>
    </div>

    <!-- 记忆列表 -->
    <div class="memories-container">
      <div class="memories-content">
        <div v-if="memoryStore.isLoading" class="loading">
          <el-icon class="is-loading" size="32"><Loading /></el-icon>
          <p>加载中...</p>
        </div>

        <div v-else-if="memoryStore.memories.length === 0" class="empty">
          <el-empty description="暂无记忆" />
        </div>

        <div v-else class="memory-list">
          <div
            v-for="memory in memoryStore.memories"
            :key="memory.id"
            class="memory-card"
          >
            <div class="memory-content">
              <div class="memory-text">{{ memory.memory }}</div>
              <div class="memory-meta">
                <span class="memory-time">
                  {{ formatDate(memory.created_at) }}
                </span>
              </div>
            </div>
            <div class="memory-actions">
              <el-button
                text
                type="danger"
                @click="handleDelete(memory.id)"
              >
                <el-icon><Delete /></el-icon>
              </el-button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useMemoryStore } from '@/stores/memory'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  ArrowLeft,
  Refresh,
  Search,
  Loading,
  Delete
} from '@element-plus/icons-vue'

const router = useRouter()
const memoryStore = useMemoryStore()
const searchQuery = ref('')

// 返回聊天页面
function goBack() {
  router.push('/')
}

// 刷新记忆
async function refreshMemories() {
  try {
    await memoryStore.fetchMemories()
    ElMessage.success('刷新成功')
  } catch (error) {
    ElMessage.error('刷新失败')
  }
}

// 搜索记忆
async function handleSearch() {
  if (searchQuery.value.trim()) {
    try {
      await memoryStore.searchMemories(searchQuery.value)
    } catch (error) {
      ElMessage.error('搜索失败')
    }
  } else {
    await refreshMemories()
  }
}

// 删除记忆
async function handleDelete(memoryId: string) {
  try {
    await ElMessageBox.confirm(
      '确定要删除这条记忆吗？',
      '确认删除',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    await memoryStore.deleteMemory(memoryId)
    ElMessage.success('删除成功')
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

// 格式化日期
function formatDate(dateString: string): string {
  const date = new Date(dateString)
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

onMounted(() => {
  refreshMemories()
})
</script>

<style scoped lang="scss">
.memory-page {
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
  
  .title {
    display: flex;
    align-items: center;
    gap: 12px;
    
    h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 600;
      color: #303133;
    }
  }
}

.search-bar {
  background: white;
  border-bottom: 1px solid #e4e7ed;
  padding: 16px 24px;
  
  .search-content {
    max-width: 1200px;
    margin: 0 auto;
  }
}

.memories-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  
  .memories-content {
    max-width: 1200px;
    margin: 0 auto;
  }
}

.loading,
.empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  color: #909399;
  
  p {
    margin-top: 12px;
    font-size: 14px;
  }
}

.memory-list {
  display: grid;
  gap: 16px;
}

.memory-card {
  background: white;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
  display: flex;
  gap: 12px;
  transition: all 0.3s;
  
  &:hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
    transform: translateY(-2px);
  }
  
  .memory-content {
    flex: 1;
    min-width: 0;
  }
  
  .memory-text {
    font-size: 15px;
    line-height: 1.6;
    color: #303133;
    margin-bottom: 8px;
    word-wrap: break-word;
  }
  
  .memory-meta {
    display: flex;
    gap: 16px;
    
    .memory-time {
      font-size: 13px;
      color: #909399;
    }
  }
  
  .memory-actions {
    flex-shrink: 0;
  }
}
</style>

<template>
  <div class="settings-page">
    <!-- 头部 -->
    <div class="header">
      <div class="header-content">
        <div class="title">
          <el-button text @click="goBack">
            <el-icon><ArrowLeft /></el-icon>
          </el-button>
          <h1>⚙️ 设置</h1>
        </div>
      </div>
    </div>

    <!-- 设置内容 -->
    <div class="settings-container">
      <div class="settings-content">
        <!-- API 配置 -->
        <el-card class="setting-section">
          <template #header>
            <div class="section-header">
              <h2>🔑 API 配置</h2>
            </div>
          </template>
          
          <el-form label-width="120px" label-position="left">
            <el-form-item label="API Base URL">
              <el-input
                v-model="settings.apiBaseUrl"
                placeholder="http://localhost:8000"
              />
            </el-form-item>
            
            <el-form-item label="超时时间（秒）">
              <el-input-number
                v-model="settings.timeout"
                :min="5"
                :max="300"
                :step="5"
              />
            </el-form-item>
          </el-form>
        </el-card>

        <!-- 聊天配置 -->
        <el-card class="setting-section">
          <template #header>
            <div class="section-header">
              <h2>💬 聊天配置</h2>
            </div>
          </template>
          
          <el-form label-width="120px" label-position="left">
            <el-form-item label="AI 模型">
              <el-select v-model="settings.model" placeholder="选择模型">
                <el-option label="GPT-4" value="gpt-4" />
                <el-option label="GPT-4 Turbo" value="gpt-4-turbo-preview" />
                <el-option label="GPT-3.5 Turbo" value="gpt-3.5-turbo" />
              </el-select>
            </el-form-item>
            
            <el-form-item label="温度">
              <el-slider
                v-model="settings.temperature"
                :min="0"
                :max="2"
                :step="0.1"
                show-stops
              />
              <div class="slider-value">{{ settings.temperature }}</div>
            </el-form-item>
            
            <el-form-item label="最大历史轮数">
              <el-input-number
                v-model="settings.maxHistory"
                :min="1"
                :max="50"
              />
            </el-form-item>
            
            <el-form-item label="自动滚动">
              <el-switch v-model="settings.autoScroll" />
            </el-form-item>
          </el-form>
        </el-card>

        <!-- 记忆配置 -->
        <el-card class="setting-section">
          <template #header>
            <div class="section-header">
              <h2>🧠 记忆配置</h2>
            </div>
          </template>
          
          <el-form label-width="120px" label-position="left">
            <el-form-item label="启用记忆">
              <el-switch v-model="settings.memoryEnabled" />
            </el-form-item>
            
            <el-form-item label="检索数量">
              <el-input-number
                v-model="settings.memoryTopK"
                :min="1"
                :max="20"
                :disabled="!settings.memoryEnabled"
              />
            </el-form-item>
          </el-form>
        </el-card>

        <!-- 界面配置 -->
        <el-card class="setting-section">
          <template #header>
            <div class="section-header">
              <h2>🎨 界面配置</h2>
            </div>
          </template>
          
          <el-form label-width="120px" label-position="left">
            <el-form-item label="主题">
              <el-select v-model="settings.theme" placeholder="选择主题">
                <el-option label="浅色" value="light" />
                <el-option label="深色" value="dark" />
                <el-option label="跟随系统" value="auto" />
              </el-select>
            </el-form-item>
            
            <el-form-item label="显示时间戳">
              <el-switch v-model="settings.showTimestamp" />
            </el-form-item>
            
            <el-form-item label="代码高亮">
              <el-switch v-model="settings.codeHighlight" />
            </el-form-item>
          </el-form>
        </el-card>

        <!-- 操作按钮 -->
        <div class="actions">
          <el-button type="primary" @click="saveSettings">
            <el-icon><Select /></el-icon>
            保存设置
          </el-button>
          <el-button @click="resetSettings">
            <el-icon><RefreshLeft /></el-icon>
            恢复默认
          </el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  ArrowLeft,
  Select,
  RefreshLeft
} from '@element-plus/icons-vue'

const router = useRouter()

// 默认设置
const defaultSettings = {
  // API 配置
  apiBaseUrl: 'http://localhost:8000',
  timeout: 30,
  
  // 聊天配置
  model: 'gpt-3.5-turbo',
  temperature: 0.7,
  maxHistory: 10,
  autoScroll: true,
  
  // 记忆配置
  memoryEnabled: true,
  memoryTopK: 5,
  
  // 界面配置
  theme: 'light',
  showTimestamp: true,
  codeHighlight: true
}

// 当前设置
const settings = ref({ ...defaultSettings })

// 返回聊天页面
function goBack() {
  router.push('/')
}

// 加载设置
function loadSettings() {
  const savedSettings = localStorage.getItem('hippo_settings')
  if (savedSettings) {
    try {
      settings.value = { ...defaultSettings, ...JSON.parse(savedSettings) }
    } catch (error) {
      console.error('Failed to load settings:', error)
    }
  }
}

// 保存设置
function saveSettings() {
  try {
    localStorage.setItem('hippo_settings', JSON.stringify(settings.value))
    ElMessage.success('设置已保存')
    
    // 应用 API Base URL 配置
    if (import.meta.env.DEV) {
      // 开发环境可以动态修改
      (window as any).__HIPPO_API_BASE_URL__ = settings.value.apiBaseUrl
    }
  } catch (error) {
    console.error('Failed to save settings:', error)
    ElMessage.error('保存失败')
  }
}

// 恢复默认设置
function resetSettings() {
  settings.value = { ...defaultSettings }
  saveSettings()
  ElMessage.success('已恢复默认设置')
}

onMounted(() => {
  loadSettings()
})
</script>

<style scoped lang="scss">
.settings-page {
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
    max-width: 1000px;
    margin: 0 auto;
    padding: 16px 24px;
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

.settings-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  
  .settings-content {
    max-width: 1000px;
    margin: 0 auto;
  }
}

.setting-section {
  margin-bottom: 20px;
  
  .section-header {
    h2 {
      margin: 0;
      font-size: 18px;
      font-weight: 600;
      color: #303133;
    }
  }
  
  .slider-value {
    text-align: center;
    color: #606266;
    font-size: 14px;
    margin-top: 8px;
  }
}

.actions {
  display: flex;
  gap: 12px;
  justify-content: center;
  padding: 24px 0;
}
</style>

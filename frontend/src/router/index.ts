import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
    {
        path: '/',
        name: 'Home',
        component: () => import('@/views/Chat.vue'),
        meta: {
            title: 'Hippo - 智能记忆助手'
        }
    },
    {
        path: '/memory',
        name: 'Memory',
        component: () => import('@/views/Memory.vue'),
        meta: {
            title: '记忆管理'
        }
    },
    {
        path: '/settings',
        name: 'Settings',
        component: () => import('@/views/Settings.vue'),
        meta: {
            title: '设置'
        }
    }
]

const router = createRouter({
    history: createWebHistory(),
    routes
})

// 路由守卫 - 更新页面标题
router.beforeEach((to, from, next) => {
    if (to.meta.title) {
        document.title = to.meta.title as string
    }
    next()
})

export default router

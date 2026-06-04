<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { useTheme } from '../composables/useTheme'
import { fetchHealth } from '../api/client'

const route = useRoute()
const { theme, toggleTheme } = useTheme()
const device = ref('cpu')

/** 切换「预测 / 训练」时保留页面状态（输入、进度、结果等） */
const cachedViews = ['PredictView', 'TrainView']

onMounted(async () => {
  try {
    const h = await fetchHealth()
    device.value = h.device
  } catch {
    /* backend offline */
  }
})
</script>

<template>
  <div class="app-shell">
    <div class="bg-mesh" aria-hidden="true" />

    <header class="header">
      <div class="brand">
        <router-link to="/" class="brand-link">
          <div class="logo-mark">
            <svg viewBox="0 0 24 24" width="28" height="28" fill="none">
              <circle cx="8" cy="12" r="2.5" fill="currentColor" />
              <circle cx="16" cy="7" r="2" fill="currentColor" opacity="0.7" />
              <circle cx="16" cy="17" r="2" fill="currentColor" opacity="0.7" />
              <path stroke="currentColor" stroke-width="1.2" d="M8 12 L16 7 M8 12 L16 17" />
            </svg>
          </div>
          <div>
            <h1>AtomProp</h1>
            <p class="tagline">基于 GeAT 的分子性质建模平台</p>
          </div>
        </router-link>
      </div>

      <nav class="nav">
        <router-link to="/" class="nav-link" :class="{ active: route.name === 'predict' }">
          <el-icon><DataAnalysis /></el-icon>
          性质预测
        </router-link>
        <router-link to="/train" class="nav-link" :class="{ active: route.name === 'train' }">
          <el-icon><Setting /></el-icon>
          模型训练
        </router-link>
      </nav>

      <div class="header-actions">
        <span class="badge" :class="{ on: device === 'cuda' }">
          <el-icon><Cpu /></el-icon>
          {{ device === 'cuda' ? 'GPU' : 'CPU' }}
        </span>
        <el-tooltip :content="theme === 'dark' ? '切换为日间模式' : '切换为夜间模式'">
          <el-button circle class="theme-btn" @click="toggleTheme">
            <el-icon v-if="theme === 'dark'"><Sunny /></el-icon>
            <el-icon v-else><Moon /></el-icon>
          </el-button>
        </el-tooltip>
      </div>
    </header>

    <main class="main-content">
      <router-view v-slot="{ Component }">
        <keep-alive :include="cachedViews">
          <component :is="Component" />
        </keep-alive>
      </router-view>
    </main>
  </div>
</template>

<style scoped>
.app-shell {
  position: relative;
  min-height: 100vh;
  padding: 0 1.5rem 2rem;
  max-width: 1400px;
  margin: 0 auto;
}

.bg-mesh {
  position: fixed;
  inset: 0;
  z-index: -1;
  background:
    radial-gradient(ellipse 80% 50% at 50% -20%, var(--mesh-accent), transparent),
    radial-gradient(ellipse 60% 40% at 100% 50%, var(--mesh-blue), transparent),
    var(--bg-deep);
  pointer-events: none;
  transition: background 0.3s ease;
}

.header {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  padding: 1.5rem 0 1.25rem;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
}

.brand-link {
  display: flex;
  align-items: center;
  gap: 1rem;
  text-decoration: none;
  color: inherit;
}

.logo-mark {
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, var(--accent-dim), #0891b2);
  border-radius: 12px;
  color: #fff;
  box-shadow: 0 8px 24px var(--accent-glow);
}

.brand h1 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
}

.tagline {
  margin: 0.15rem 0 0;
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.nav {
  display: flex;
  gap: 0.5rem;
  flex: 1;
  justify-content: center;
}

.nav-link {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  text-decoration: none;
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--text-secondary);
  transition: all 0.2s;
}

.nav-link:hover {
  color: var(--text-primary);
  background: var(--bg-card-hover);
}

.nav-link.active {
  color: var(--accent);
  background: var(--accent-glow);
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-left: auto;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  color: var(--text-secondary);
}

.badge.on {
  color: var(--accent);
  border-color: var(--accent);
  background: var(--accent-glow);
}

.theme-btn {
  border-color: var(--border-subtle) !important;
  background: var(--bg-card) !important;
  color: var(--text-primary) !important;
}

.main-content {
  min-width: 0;
}

@media (max-width: 768px) {
  .nav {
    order: 3;
    width: 100%;
    justify-content: flex-start;
  }
  .header-actions {
    margin-left: 0;
  }
}
</style>

<script setup lang="ts">
import { onActivated, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'

defineOptions({ name: 'PredictView' })
import ModelPanel from '../components/ModelPanel.vue'
import InputPanel from '../components/InputPanel.vue'
import ResultsTable from '../components/ResultsTable.vue'
import {
  exportResultsCsv,
  fetchHealth,
  getErrorMessage,
  predictFile,
  predictSmiles,
  type PredictResult,
} from '../api/client'

const modelLoaded = ref(false)
const modelName = ref<string | null>(null)
const smilesText = ref('')
const results = ref<PredictResult[]>([])
const predicting = ref(false)
const statusText = ref('就绪 — 请先加载模型')

function parseLines(text: string): string[] {
  return text.split(/\r?\n/).map((s) => s.trim()).filter(Boolean)
}

async function refreshHealth() {
  try {
    const h = await fetchHealth()
    modelLoaded.value = h.model_loaded
    modelName.value = h.model_name
  } catch {
    statusText.value = '无法连接后端，请确认 Django 服务已启动'
  }
}

function onModelLoaded(name: string) {
  modelLoaded.value = true
  modelName.value = name
  statusText.value = `模型已加载: ${name}`
  ElMessage.success('模型加载成功')
}

function onFileLoaded(content: string, filename: string) {
  smilesText.value = content
  statusText.value = `已从 ${filename} 导入 ${parseLines(content).length} 条 SMILES`
}

async function runPredict() {
  if (!modelLoaded.value) {
    ElMessage.warning('请先加载 .pth 模型文件')
    return
  }
  const lines = parseLines(smilesText.value)
  if (!lines.length) {
    ElMessage.warning('请输入或导入 SMILES')
    return
  }
  predicting.value = true
  statusText.value = `正在预测 ${lines.length} 条分子…`
  try {
    const res = await predictSmiles(lines)
    results.value = res.results
    statusText.value = `完成: ${res.count} / ${res.input_count} 条有效预测`
    if (res.count < res.input_count) {
      ElMessage.info(`${res.input_count - res.count} 条 SMILES 无法解析，已跳过`)
    }
  } catch (e) {
    ElMessage.error(getErrorMessage(e))
    statusText.value = '预测失败'
  } finally {
    predicting.value = false
  }
}

async function runPredictFile(file: File) {
  if (!modelLoaded.value) {
    ElMessage.warning('请先加载 .pth 模型文件')
    return
  }
  predicting.value = true
  statusText.value = `正在从 ${file.name} 预测…`
  try {
    const res = await predictFile(file)
    results.value = res.results
    smilesText.value = res.results.map((r) => r.smiles).join('\n')
    statusText.value = `完成: ${res.count} / ${res.input_count} 条 (${file.name})`
  } catch (e) {
    ElMessage.error(getErrorMessage(e))
    statusText.value = '预测失败'
  } finally {
    predicting.value = false
  }
}

async function exportCsv() {
  if (!results.value.length) {
    ElMessage.warning('暂无结果可导出')
    return
  }
  try {
    const blob = await exportResultsCsv(results.value)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'predictions.csv'
    a.click()
    URL.revokeObjectURL(url)
    ElMessage.success('CSV 已下载')
  } catch (e) {
    ElMessage.error(getErrorMessage(e))
  }
}

function clearAll() {
  smilesText.value = ''
  results.value = []
  statusText.value = '已清空'
}

onMounted(refreshHealth)

/** 从训练页切回时同步后端已加载的模型状态 */
onActivated(refreshHealth)
</script>

<template>
  <div class="main-grid">
    <aside class="sidebar">
      <ModelPanel
        :model-loaded="modelLoaded"
        :model-name="modelName"
        @loaded="onModelLoaded"
      />
      <section class="card tips-card">
        <h3>使用说明</h3>
        <ul>
          <li>加载训练好的 <code>.pth</code> 模型</li>
          <li>每行输入一条 SMILES，或拖入 TXT/CSV</li>
          <li>CSV 自动识别 smiles / smi 等列名</li>
          <li>预测完成后可导出 CSV</li>
        </ul>
      </section>
    </aside>

    <div class="workspace">
      <InputPanel
        v-model="smilesText"
        :predicting="predicting"
        :disabled="!modelLoaded"
        @predict="runPredict"
        @predict-file="runPredictFile"
        @file-loaded="onFileLoaded"
        @clear="clearAll"
      />
      <ResultsTable :results="results" :loading="predicting" @export="exportCsv" />
      <footer class="status-bar">
        <el-icon><InfoFilled /></el-icon>
        <span>{{ statusText }}</span>
      </footer>
    </div>
  </div>
</template>

<style scoped>
.main-grid {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 1.5rem;
  align-items: start;
}

@media (max-width: 960px) {
  .main-grid {
    grid-template-columns: 1fr;
  }
}

.sidebar {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.tips-card h3 {
  margin: 0 0 0.75rem;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
}

.tips-card ul {
  margin: 0;
  padding-left: 1.1rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
  line-height: 1.65;
}

.tips-card code {
  font-family: var(--font-mono);
  font-size: 0.8em;
  color: var(--accent);
  background: var(--code-bg);
  padding: 0.1em 0.35em;
  border-radius: 4px;
}

.workspace {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  min-width: 0;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.65rem 1rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
}
</style>

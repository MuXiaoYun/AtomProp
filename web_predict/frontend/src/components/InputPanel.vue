<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { parseSmilesFromCsvClient, parseSmilesFromTextClient } from '../utils/parseFile'

const props = defineProps<{
  modelValue: string
  predicting: boolean
  disabled?: boolean
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string]
  predict: []
  'predict-file': [file: File]
  'file-loaded': [content: string, filename: string]
  clear: []
}>()

const dragOver = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)

function updateText(v: string) {
  emit('update:modelValue', v)
}

async function handleFile(file: File) {
  const name = file.name.toLowerCase()
  if (!name.endsWith('.txt') && !name.endsWith('.csv')) {
    ElMessage.warning('仅支持 .txt 或 .csv 文件')
    return
  }
  try {
    let smiles: string[]
    if (name.endsWith('.csv')) {
      smiles = await parseSmilesFromCsvClient(file)
    } else {
      const text = await file.text()
      smiles = parseSmilesFromTextClient(text)
    }
    if (!smiles.length) {
      ElMessage.warning('文件中未找到有效 SMILES')
      return
    }
    const content = smiles.join('\n')
    updateText(content)
    emit('file-loaded', content, file.name)
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : '文件读取失败')
  }
}

function onDrop(e: DragEvent) {
  dragOver.value = false
  const file = e.dataTransfer?.files?.[0]
  if (file) void handleFile(file)
}

function onDragOver(e: DragEvent) {
  e.preventDefault()
  dragOver.value = true
}

function onDragLeave() {
  dragOver.value = false
}

function onPickFile() {
  fileInput.value?.click()
}

function onInputChange(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) void handleFile(file)
  input.value = ''
}
</script>

<template>
  <section
    class="card input-panel"
    :class="{ 'drag-over': dragOver }"
    @drop.prevent="onDrop"
    @dragover.prevent="onDragOver"
    @dragleave="onDragLeave"
  >
    <div class="panel-head">
      <h2>SMILES 输入</h2>
      <span class="sub">每行一条，或拖入 TXT / CSV</span>
    </div>

    <input
      ref="fileInput"
      type="file"
      accept=".txt,.csv"
      hidden
      @change="onInputChange"
    />

    <div class="textarea-wrap">
      <el-input
        :model-value="modelValue"
        type="textarea"
        :rows="12"
        placeholder="CCO&#10;c1ccccc1&#10;或拖放文件到此处…"
        :disabled="predicting"
        @update:model-value="updateText"
      />
      <div v-if="dragOver" class="drop-overlay">
        <el-icon :size="40"><Upload /></el-icon>
        <span>释放以导入文件</span>
      </div>
    </div>

    <div class="actions">
      <el-button :disabled="predicting" @click="onPickFile">
        <el-icon><FolderOpened /></el-icon>
        选择文件
      </el-button>
      <el-button :disabled="predicting" @click="emit('clear')">
        <el-icon><Delete /></el-icon>
        清空
      </el-button>
      <el-button
        type="primary"
        :loading="predicting"
        :disabled="disabled"
        class="predict-btn"
        @click="emit('predict')"
      >
        <el-icon v-if="!predicting"><Promotion /></el-icon>
        {{ predicting ? '预测中…' : '开始预测' }}
      </el-button>
    </div>
  </section>
</template>

<style scoped>
.input-panel {
  position: relative;
  transition: box-shadow 0.2s, border-color 0.2s;
}

.input-panel.drag-over {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent-glow);
}

.panel-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 1rem;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.panel-head h2 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
}

.sub {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.textarea-wrap {
  position: relative;
  margin-bottom: 1rem;
}

.textarea-wrap :deep(.el-textarea__inner) {
  font-family: var(--font-mono);
  font-size: 0.85rem;
  line-height: 1.55;
  min-height: 280px !important;
}

.drop-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  background: rgba(13, 148, 136, 0.15);
  border: 2px dashed var(--accent);
  border-radius: 8px;
  color: var(--accent);
  font-weight: 600;
  pointer-events: none;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
}

.predict-btn {
  margin-left: auto;
  min-width: 140px;
}

@media (max-width: 600px) {
  .predict-btn {
    margin-left: 0;
    width: 100%;
  }
}
</style>

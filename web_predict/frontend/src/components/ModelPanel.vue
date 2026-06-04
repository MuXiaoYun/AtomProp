<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { uploadModel, getErrorMessage } from '../api/client'

defineProps<{
  modelLoaded: boolean
  modelName: string | null
}>()

const emit = defineEmits<{
  loaded: [name: string]
}>()

const uploading = ref(false)

async function onFileChange(uploadFile: { raw?: File }) {
  const file = uploadFile?.raw
  if (!file) return
  if (!file.name.toLowerCase().endsWith('.pth')) {
    ElMessage.warning('请选择 .pth 模型文件')
    return
  }
  uploading.value = true
  try {
    const res = await uploadModel(file)
    emit('loaded', res.model_name)
  } catch (e) {
    ElMessage.error(getErrorMessage(e))
  } finally {
    uploading.value = false
  }
}
</script>

<template>
  <section class="card model-panel">
    <h3>模型</h3>
    <div v-if="modelLoaded && modelName" class="model-active">
      <el-icon class="ok"><CircleCheckFilled /></el-icon>
      <span class="name" :title="modelName">{{ modelName }}</span>
    </div>
    <p v-else class="hint">尚未加载模型</p>

    <el-upload
      class="upload-wrap"
      drag
      :auto-upload="false"
      :show-file-list="false"
      accept=".pth"
      :disabled="uploading"
      @change="onFileChange"
    >
      <div class="drop-inner">
        <el-icon :size="32"><UploadFilled /></el-icon>
        <span>{{ uploading ? '上传中…' : '拖入或点击选择 .pth' }}</span>
      </div>
    </el-upload>
  </section>
</template>

<style scoped>
.model-panel h3 {
  margin: 0 0 1rem;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
}

.model-active {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
  padding: 0.6rem 0.75rem;
  background: rgba(20, 184, 166, 0.1);
  border: 1px solid rgba(20, 184, 166, 0.25);
  border-radius: 8px;
}

.model-active .ok {
  color: var(--accent);
  flex-shrink: 0;
}

.model-active .name {
  font-size: 0.8rem;
  font-family: var(--font-mono);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.hint {
  margin: 0 0 1rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.upload-wrap {
  width: 100%;
}

.drop-inner {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  padding: 1.25rem;
  border: 2px dashed var(--border-subtle);
  border-radius: 10px;
  color: var(--text-secondary);
  font-size: 0.8rem;
  transition: border-color 0.2s, background 0.2s;
  cursor: pointer;
}

.drop-inner:hover {
  border-color: var(--accent);
  background: var(--bg-card-hover);
  color: var(--text-primary);
}
</style>

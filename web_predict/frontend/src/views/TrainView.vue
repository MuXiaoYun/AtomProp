<script setup lang="ts">
import { onActivated, onMounted, onUnmounted, ref } from 'vue'
import { ElMessage } from 'element-plus'

defineOptions({ name: 'TrainView' })
import LossChart from '../components/LossChart.vue'
import {
  downloadTrainedModel,
  fetchTrainDefaults,
  getErrorMessage,
  pollTrainJob,
  startTraining,
  uploadTrainDataset,
  uploadTrainInitModel,
  type TrainJobStatus,
} from '../api/client'

const defaults = ref({ num_epochs: 100, learning_rate: 0.0001 })
const datasetId = ref<string | null>(null)
const datasetName = ref('')
const smilesColumn = ref('')
const targetColumns = ref<string[]>([])
const targetColumn = ref('')
const preview = ref<Record<string, string>[]>([])
const rowCount = ref(0)

const taskType = ref<'regression' | 'classification'>('regression')
const initMode = ref<'scratch' | 'checkpoint' | 'pretrain'>('scratch')
const initModelId = ref<string | null>(null)
const initModelName = ref('')
const numEpochs = ref(100)
const learningRate = ref(0.0001)

const training = ref(false)
const jobId = ref<string | null>(null)
const jobStatus = ref<TrainJobStatus | null>(null)
let pollTimer: ReturnType<typeof setInterval> | null = null

async function loadDefaults() {
  try {
    const d = await fetchTrainDefaults()
    defaults.value = d
    numEpochs.value = d.num_epochs
    learningRate.value = d.learning_rate
  } catch {
    /* use hardcoded fallbacks */
  }
}

async function onDatasetChange(uploadFile: { raw?: File }) {
  const file = uploadFile?.raw
  if (!file) return
  try {
    const res = await uploadTrainDataset(file)
    datasetId.value = res.dataset_id
    datasetName.value = res.filename
    smilesColumn.value = res.smiles_column
    targetColumns.value = res.target_columns
    targetColumn.value = res.target_columns[0] ?? ''
    preview.value = res.preview
    rowCount.value = res.row_count
    ElMessage.success(`已加载 ${res.row_count} 行数据`)
  } catch (e) {
    ElMessage.error(getErrorMessage(e))
  }
}

async function onInitModelChange(uploadFile: { raw?: File }) {
  const file = uploadFile?.raw
  if (!file) return
  try {
    const res = await uploadTrainInitModel(file)
    initModelId.value = res.model_id
    initModelName.value = res.model_name
    initMode.value = 'checkpoint'
    ElMessage.success('初始模型已上传')
  } catch (e) {
    ElMessage.error(getErrorMessage(e))
  }
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

function startPolling(id: string) {
  stopPolling()
  pollTimer = setInterval(async () => {
    try {
      const st = await pollTrainJob(id)
      jobStatus.value = st
      if (st.status === 'completed') {
        stopPolling()
        training.value = false
        ElMessage.success('训练完成！可下载模型')
      } else if (st.status === 'failed') {
        stopPolling()
        training.value = false
        ElMessage.error(st.error || '训练失败')
      }
    } catch {
      /* ignore transient errors */
    }
  }, 1500)
}

async function handleStartTrain() {
  if (!datasetId.value) {
    ElMessage.warning('请先上传训练数据集 CSV')
    return
  }
  if (!targetColumn.value) {
    ElMessage.warning('请选择目标列（真实值）')
    return
  }
  if (initMode.value === 'checkpoint' && !initModelId.value) {
    ElMessage.warning('继续训练模式下请上传 .pth 模型')
    return
  }

  training.value = true
  jobStatus.value = null
  try {
    const res = await startTraining({
      dataset_id: datasetId.value,
      target_column: targetColumn.value,
      task_type: taskType.value,
      init_mode: initMode.value,
      init_model_id: initModelId.value ?? undefined,
      num_epochs: numEpochs.value,
      learning_rate: learningRate.value,
    })
    jobId.value = res.job_id
    startPolling(res.job_id)
    ElMessage.info('训练已在后台启动')
  } catch (e) {
    training.value = false
    ElMessage.error(getErrorMessage(e))
  }
}

async function handleDownloadModel() {
  if (!jobId.value) return
  try {
    const blob = await downloadTrainedModel(jobId.value)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `atomprop_${taskType.value}_${jobId.value}.pth`
    a.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    ElMessage.error(getErrorMessage(e))
  }
}

async function refreshJobStatus() {
  if (!jobId.value) return
  try {
    const st = await pollTrainJob(jobId.value)
    jobStatus.value = st
    if (st.status === 'completed') {
      training.value = false
      stopPolling()
    } else if (st.status === 'failed') {
      training.value = false
      stopPolling()
    } else if (st.status === 'running' || st.status === 'pending') {
      training.value = true
      if (!pollTimer) startPolling(jobId.value)
    }
  } catch {
    /* ignore */
  }
}

onMounted(() => {
  loadDefaults()
  refreshJobStatus()
})

/** 切回训练页时立即刷新进度；后台轮询在页面缓存期间不中断 */
onActivated(refreshJobStatus)

onUnmounted(stopPolling)
</script>

<template>
  <div class="train-page">
    <div class="train-grid">
      <section class="card">
        <h2>1. 上传数据集</h2>
        <p class="hint">CSV 需包含 SMILES 列与至少一列真实值</p>
        <el-upload
          drag
          :auto-upload="false"
          :show-file-list="false"
          accept=".csv"
          :disabled="training"
          @change="onDatasetChange"
        >
          <div class="drop-zone">
            <el-icon :size="36"><UploadFilled /></el-icon>
            <span v-if="datasetName">{{ datasetName }}（{{ rowCount }} 行）</span>
            <span v-else>拖入或点击上传 CSV</span>
          </div>
        </el-upload>

        <template v-if="datasetId">
          <el-divider />
          <p class="field-label">
            SMILES 列（自动识别）: <code>{{ smilesColumn }}</code>
          </p>
          <p class="field-label">目标列（真实值）</p>
          <el-select v-model="targetColumn" placeholder="选择目标列" style="width: 100%">
            <el-option
              v-for="col in targetColumns"
              :key="col"
              :label="col"
              :value="col"
            />
          </el-select>

          <div v-if="preview.length" class="preview-table">
            <p class="field-label">数据预览</p>
            <el-table :data="preview" :stripe="false" size="small" max-height="160">
              <el-table-column
                v-for="key in Object.keys(preview[0] || {})"
                :key="key"
                :prop="key"
                :label="key"
                min-width="100"
                show-overflow-tooltip
              />
            </el-table>
          </div>
        </template>
      </section>

      <section class="card">
        <h2>2. 训练配置</h2>

        <p class="field-label">任务类型</p>
        <el-radio-group v-model="taskType" :disabled="training">
          <el-radio value="regression">回归 (finetune_regression)</el-radio>
          <el-radio value="classification">分类 (finetune)</el-radio>
        </el-radio-group>

        <p class="field-label">初始化方式</p>
        <el-radio-group v-model="initMode" :disabled="training" class="init-group">
          <el-radio value="scratch">从头训练（随机初始化）</el-radio>
          <el-radio value="pretrain">使用预训练权重</el-radio>
          <el-radio value="checkpoint">从已有模型继续训练</el-radio>
        </el-radio-group>

        <div v-if="initMode === 'checkpoint'" class="init-upload">
          <el-upload
            :auto-upload="false"
            :show-file-list="false"
            accept=".pth"
            :disabled="training"
            @change="onInitModelChange"
          >
            <el-button type="primary" plain>
              <el-icon><Upload /></el-icon>
              {{ initModelName || '上传 .pth 模型' }}
            </el-button>
          </el-upload>
        </div>

        <el-divider />

        <div class="param-row">
          <div class="param-field">
            <label>训练 Epoch 数</label>
            <el-input-number
              v-model="numEpochs"
              :min="1"
              :max="500"
              :disabled="training"
              controls-position="right"
            />
          </div>
          <div class="param-field">
            <label>学习率（Head）</label>
            <el-input-number
              v-model="learningRate"
              :min="1e-7"
              :max="1"
              :step="1e-5"
              :precision="7"
              :disabled="training"
              controls-position="right"
            />
          </div>
        </div>
        <p class="hint small">
          其余参数（batch size、模型结构、早停等）与 config_reg.py / config_finetune.py 保持一致
        </p>

        <el-button
          type="primary"
          size="large"
          class="start-btn"
          :loading="training"
          :disabled="!datasetId"
          @click="handleStartTrain"
        >
          {{ training ? '训练中…' : '开始训练' }}
        </el-button>
      </section>
    </div>

    <section v-if="training || jobStatus" class="card progress-section">
      <h2>训练进度</h2>
      <el-progress
        :percentage="jobStatus?.progress ?? 0"
        :status="jobStatus?.status === 'failed' ? 'exception' : jobStatus?.status === 'completed' ? 'success' : undefined"
        :stroke-width="14"
        striped
        :striped-flow="training"
      />
      <p class="progress-msg">{{ jobStatus?.message || '等待开始…' }}</p>
      <p v-if="jobStatus" class="epoch-info">
        Epoch {{ jobStatus.epoch }} / {{ jobStatus.total_epochs }}
      </p>
    </section>

    <LossChart
      v-if="jobStatus && (jobStatus.train_losses.length || training)"
      :train-losses="jobStatus.train_losses"
      :val-losses="jobStatus.val_losses"
    />

    <section
      v-if="jobStatus?.status === 'completed' && jobStatus.model_ready"
      class="card done-section"
    >
      <el-result icon="success" title="训练完成">
        <template #sub-title>
          <span>最佳验证 Loss: {{ jobStatus.best_metric?.toFixed(6) ?? '—' }}</span>
        </template>
        <template #extra>
          <el-button type="primary" size="large" @click="handleDownloadModel">
            <el-icon><Download /></el-icon>
            下载模型 (.pth)
          </el-button>
        </template>
      </el-result>
    </section>
  </div>
</template>

<style scoped>
.train-page {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.train-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
}

@media (max-width: 900px) {
  .train-grid {
    grid-template-columns: 1fr;
  }
}

h2 {
  margin: 0 0 0.75rem;
  font-size: 1.1rem;
  font-weight: 600;
}

.hint {
  margin: 0 0 1rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.hint.small {
  margin-top: 0.5rem;
  font-size: 0.8rem;
}

.field-label {
  margin: 1rem 0 0.5rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.field-label code {
  color: var(--accent);
  font-family: var(--font-mono);
  background: var(--code-bg);
  padding: 0.1em 0.35em;
  border-radius: 4px;
}

.drop-zone {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  padding: 2rem;
  border: 2px dashed var(--border-subtle);
  border-radius: 10px;
  color: var(--text-secondary);
  transition: border-color 0.2s;
}

.drop-zone:hover {
  border-color: var(--accent);
  color: var(--text-primary);
}

.init-group {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.35rem;
}

.init-upload {
  margin-top: 0.75rem;
}

.param-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.param-field label {
  display: block;
  font-size: 0.85rem;
  color: var(--text-secondary);
  margin-bottom: 0.35rem;
}

.start-btn {
  width: 100%;
  margin-top: 1.25rem;
}

.progress-section h2 {
  margin-bottom: 1rem;
}

.progress-msg {
  margin: 0.75rem 0 0;
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.epoch-info {
  margin: 0.35rem 0 0;
  font-size: 0.85rem;
  font-family: var(--font-mono);
  color: var(--accent);
}

.preview-table {
  margin-top: 1rem;
}

.done-section {
  text-align: center;
}
</style>

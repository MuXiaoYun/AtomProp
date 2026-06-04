<script setup lang="ts">
import { computed } from 'vue'
import type { PredictResult } from '../api/client'

const props = defineProps<{
  results: PredictResult[]
  loading?: boolean
}>()

defineEmits<{ export: [] }>()

const tableData = computed(() =>
  props.results.map((r, i) => ({
    index: i + 1,
    smiles: r.smiles,
    value: r.predicted_value,
  })),
)
</script>

<template>
  <section class="card results-panel">
    <div class="panel-head">
      <div>
        <h2>预测结果</h2>
        <span class="count">{{ results.length }} 条记录</span>
      </div>
      <el-button type="primary" plain :disabled="!results.length" @click="$emit('export')">
        <el-icon><Download /></el-icon>
        导出 CSV
      </el-button>
    </div>

    <el-table
      v-loading="loading"
      :data="tableData"
      :stripe="false"
      empty-text="暂无数据 — 完成预测后将在此显示"
      max-height="420"
      style="width: 100%"
    >
      <el-table-column prop="index" label="#" width="56" align="center" />
      <el-table-column prop="smiles" label="SMILES" min-width="280">
        <template #default="{ row }">
          <code class="smiles-cell">{{ row.smiles }}</code>
        </template>
      </el-table-column>
      <el-table-column prop="value" label="预测值" width="160" align="right">
        <template #default="{ row }">
          <span class="value-cell">{{ row.value.toFixed(6) }}</span>
        </template>
      </el-table-column>
    </el-table>
  </section>
</template>

<style scoped>
.results-panel .panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.results-panel h2 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
}

.count {
  display: block;
  font-size: 0.8rem;
  color: var(--text-secondary);
  margin-top: 0.2rem;
}

.smiles-cell {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  word-break: break-all;
  color: var(--text-code);
}

.value-cell {
  font-family: var(--font-mono);
  font-weight: 500;
  color: var(--accent);
}
</style>

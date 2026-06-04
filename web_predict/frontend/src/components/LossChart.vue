<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  trainLosses: number[]
  valLosses: number[]
}>()

const width = 640
const height = 260
const pad = { top: 24, right: 24, bottom: 36, left: 52 }

const chart = computed(() => {
  const train = props.trainLosses
  const val = props.valLosses
  if (!train.length) return null

  const all = [...train, ...val]
  const minY = Math.min(...all) * 0.95
  const maxY = Math.max(...all) * 1.05 || 1
  const n = train.length
  const innerW = width - pad.left - pad.right
  const innerH = height - pad.top - pad.bottom

  const x = (i: number) => pad.left + (n <= 1 ? innerW / 2 : (i / (n - 1)) * innerW)
  const y = (v: number) => pad.top + innerH - ((v - minY) / (maxY - minY || 1)) * innerH

  const line = (vals: number[]) =>
    vals.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(v)}`).join(' ')

  const yTicks = 5
  const ticks = Array.from({ length: yTicks }, (_, i) => {
    const v = minY + ((maxY - minY) * i) / (yTicks - 1)
    return { v, py: y(v) }
  })

  return { trainPath: line(train), valPath: val.length ? line(val) : '', ticks, minY, maxY, n }
})
</script>

<template>
  <div class="loss-chart card">
    <h3>Loss 曲线</h3>
    <svg
      v-if="chart"
      :viewBox="`0 0 ${width} ${height}`"
      class="chart-svg"
      role="img"
      aria-label="训练与验证 loss 曲线"
    >
      <line
        v-for="t in chart.ticks"
        :key="t.v"
        :x1="pad.left"
        :y1="t.py"
        :x2="width - pad.right"
        :y2="t.py"
        class="grid-line"
      />
      <text
        v-for="t in chart.ticks"
        :key="'l' + t.v"
        :x="pad.left - 8"
        :y="t.py + 4"
        class="axis-label"
        text-anchor="end"
      >
        {{ t.v.toFixed(3) }}
      </text>
      <path :d="chart.trainPath" class="line-train" fill="none" stroke-width="2" />
      <path v-if="chart.valPath" :d="chart.valPath" class="line-val" fill="none" stroke-width="2" />
      <text :x="width / 2" :y="height - 6" class="axis-label" text-anchor="middle">Epoch</text>
    </svg>
    <p v-else class="empty">训练开始后将显示 loss 曲线</p>
    <div class="legend">
      <span class="leg-item"><i class="dot train" />训练 Loss</span>
      <span class="leg-item"><i class="dot val" />验证 Loss</span>
    </div>
  </div>
</template>

<style scoped>
.loss-chart h3 {
  margin: 0 0 1rem;
  font-size: 1rem;
  font-weight: 600;
}

.chart-svg {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
}

.grid-line {
  stroke: var(--border-subtle);
  stroke-width: 1;
}

.axis-label {
  fill: var(--text-secondary);
  font-size: 10px;
  font-family: var(--font-mono);
}

.line-train {
  stroke: var(--chart-train);
}

.line-val {
  stroke: var(--chart-val);
}

.empty {
  color: var(--text-secondary);
  font-size: 0.9rem;
  text-align: center;
  padding: 3rem 0;
}

.legend {
  display: flex;
  gap: 1.5rem;
  margin-top: 0.75rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 0.35rem;
  vertical-align: middle;
}

.dot.train {
  background: var(--chart-train);
}

.dot.val {
  background: var(--chart-val);
}
</style>

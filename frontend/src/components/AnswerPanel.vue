<script setup lang="ts">
import { computed, ref } from 'vue'
import type { RunResult } from '../types'
import { formatNumber, formatTime, resultStatus } from '../utils/presentation'
import AppIcon from './AppIcon.vue'

const props = defineProps<{ result: RunResult | null; running: boolean; elapsed: number; question: string; error: string }>()
const status = computed(() => props.result ? resultStatus(props.result) : null)
const copyLabel = ref('Copy answer')
const copyError = ref('')
async function copy() {
  try { await navigator.clipboard.writeText(props.result?.answer || ''); copyLabel.value = 'Copied'; setTimeout(() => { copyLabel.value = 'Copy answer' }, 1600) }
  catch { copyError.value = 'Couldn’t access the clipboard. Select the answer and copy it manually.' }
}
function download() {
  if (!props.result) return
  const blob = new Blob([JSON.stringify({ question: props.question, ...props.result }, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a'); link.href = url; link.download = `bm25-result-${new Date().toISOString().replace(/[:.]/g, '-')}.json`; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
</script>

<template>
  <section v-if="running" class="answer-card running-card" role="status" aria-live="polite">
    <span class="running-orbit"><AppIcon name="search" :size="26" /></span>
    <div><h2>Working through your question</h2><p>TRACE is searching the corpus and assessing the available evidence. The answer, sources, and activity will appear when the run completes.</p><span class="running-time">Elapsed {{ formatTime(elapsed) }}</span></div>
  </section>
  <section v-else-if="result" class="answer-card" aria-labelledby="answer-heading">
    <div class="section-heading"><span class="eyebrow">RESEARCH RESULT</span><span class="status-pill" :class="status?.tone"><AppIcon :name="status?.tone === 'success' ? 'check' : 'info'" :size="14" />{{ status?.label }}</span></div>
    <h2 id="answer-heading">{{ result.answer === 'Unknown' ? 'The available evidence doesn’t establish an answer' : result.answer }}</h2>
    <p class="answer-explanation">{{ status?.detail }}</p>
    <div class="answer-metrics"><div><AppIcon name="network" :size="17" /><span>Tool calls</span><strong>{{ formatNumber(result.tool_calls) }}</strong></div><div><AppIcon name="layers" :size="17" /><span>API tokens</span><strong>{{ formatNumber(result.total_tokens) }}</strong></div><div><AppIcon name="clock" :size="17" /><span>Duration</span><strong>{{ formatTime(result.elapsed_seconds) }}</strong></div></div>
    <div class="answer-actions"><button class="button text" @click="copy"><AppIcon name="copy" :size="16" />{{ copyLabel }}</button><button class="button text" @click="download"><AppIcon name="download" :size="16" />Export run</button></div>
    <p v-if="copyError" class="muted" role="status">{{ copyError }}</p>
  </section>
  <section v-else-if="error && question" class="answer-card failed-card" role="status"><AppIcon name="info" :size="24" /><h2>The run didn’t finish</h2><p>No answer was returned. Address the error above, then try again.</p></section>
</template>

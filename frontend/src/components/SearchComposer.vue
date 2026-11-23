<script setup lang="ts">
import { nextTick, ref } from 'vue'
import AppIcon from './AppIcon.vue'

defineProps<{ running: boolean; ready: boolean; canRun: boolean; error: string; compact: boolean }>()
const question = defineModel<string>('question', { required: true })
const confirmed = defineModel<boolean>('confirmed', { required: true })
const emit = defineEmits<{ submit: [] }>()
const input = ref<HTMLTextAreaElement>()
function shortcut(event: KeyboardEvent) {
  if (!event.isComposing && event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
    event.preventDefault(); emit('submit')
  }
}
async function focus() { await nextTick(); input.value?.focus() }
defineExpose({ focus })
</script>

<template>
  <form class="composer-card" @submit.prevent="emit('submit')" :aria-busy="running">
    <div class="composer-heading"><label for="question">Research question</label><span class="small-tag">BM25 BASELINE</span></div>
    <textarea id="question" ref="input" v-model="question" maxlength="6000" :disabled="running" :rows="compact ? 3 : 5" aria-describedby="question-help form-error" placeholder="Describe the question, the clues you have, and what you need to establish." @keydown="shortcut" />
    <div class="composer-input-footer"><span id="question-help">Include relevant names, dates, and relationships to guide the search.</span><span class="character-count">{{ question.length.toLocaleString('en-US') }} / 6,000</span></div>
    <div class="composer-actions">
      <label class="confirmation"><input v-model="confirmed" type="checkbox" :disabled="running" /><span>Use the configured model for this run<span class="confirmation-sub">API charges may apply.</span></span></label>
      <button class="button primary run-button" type="submit" :disabled="!canRun"><span v-if="running" class="spinner" /><AppIcon v-else name="search" :size="18" />{{ running ? 'Running' : 'Start research' }}<kbd v-if="!running">Ctrl ↵</kbd></button>
    </div>
    <p v-if="!ready && !running" class="setup-hint"><AppIcon name="info" :size="16" />You can prepare a question now. Open System to complete the connection and configuration before starting a run.</p>
    <p id="form-error" v-show="error" role="alert" class="form-error"><AppIcon name="info" :size="17" />{{ error }}</p>
  </form>
</template>

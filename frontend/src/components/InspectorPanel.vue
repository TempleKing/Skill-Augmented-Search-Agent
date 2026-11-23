<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { InspectorTab, RunResult } from '../types'
import { describeEvent, safeSourceUrl, skillNames, toolNames } from '../utils/presentation'
import AppIcon from './AppIcon.vue'

const props = defineProps<{ result: RunResult | null; running: boolean }>()
const emit = defineEmits<{ openDocument: [docid: string] }>()
const tab = ref<InspectorTab>('evidence')
const filter = ref('')
const evidence = computed(() => (props.result?.evidence || []).filter((item) => `${item.title || ''} ${item.docid || ''} ${item.text || ''}`.toLowerCase().includes(filter.value.toLowerCase())))
const tabs: { key: InspectorTab; label: string }[] = [{ key: 'evidence', label: 'Evidence' }, { key: 'trace', label: 'Activity' }, { key: 'skills', label: 'Skills' }]
const skills = [
  { id: 'multi-hop-search', icon: 'network', description: 'Decompose a question into related searches and connect evidence across sources.' },
  { id: 'entity-disambiguation', icon: 'search', description: 'Distinguish between similarly named entities using their attributes and context.' },
  { id: 'search-recovery', icon: 'refresh', description: 'Revise queries when retrieval stalls and follow alternative lines of evidence.' },
]
watch(() => props.result, () => { filter.value = ''; tab.value = 'evidence' })
function moveTab(direction: number) {
  tab.value = tabs[(tabs.findIndex(item => item.key === tab.value) + direction + tabs.length) % tabs.length].key
  void nextTick(() => document.getElementById(`tab-${tab.value}`)?.focus())
}
function highlight(text: string) {
  return text.split(/(\[[^\]\n]{1,100}\])/g).map((value) => ({ text: value.startsWith('[') && value.endsWith(']') ? value.slice(1, -1) : value, marked: value.startsWith('[') && value.endsWith(']') }))
}
</script>

<template>
  <aside class="inspector" aria-label="Sources & activity">
    <div class="inspector-heading"><span class="inspector-symbol"><AppIcon name="shield" :size="20" /></span><div><h2>Sources & activity</h2><p>Inspect the sources and steps behind an answer.</p></div></div>
    <div class="inspector-tabs" role="tablist" aria-label="Source and activity tabs"><button v-for="item in tabs" :id="`tab-${item.key}`" :key="item.key" role="tab" :aria-selected="tab === item.key" :aria-controls="`panel-${item.key}`" :tabindex="tab === item.key ? 0 : -1" :class="{ active: tab === item.key }" @click="tab = item.key" @keydown.right.prevent="moveTab(1)" @keydown.left.prevent="moveTab(-1)">{{ item.label }}<span v-if="item.key === 'evidence' && result">{{ result.evidence.length }}</span></button></div>
    <div v-show="tab === 'evidence'" id="panel-evidence" role="tabpanel" aria-labelledby="tab-evidence" class="inspector-content">
      <template v-if="result?.evidence.length"><label class="filter-input"><AppIcon name="search" :size="16" /><input v-model="filter" aria-label="Filter evidence" placeholder="Filter by document, name, or phrase" /></label><p class="result-count">{{ evidence.length }} {{ evidence.length === 1 ? 'source excerpt' : 'source excerpts' }}</p>
        <article v-for="(item, index) in evidence" :key="item.evidence_id || index" class="evidence-card"><div class="evidence-card-top"><span class="source-label"><AppIcon name="file" :size="14" />Document {{ item.docid || 'Not reported' }}</span><span class="evidence-rank" v-if="item.rank">#{{ item.rank }}</span></div><h3><a v-if="safeSourceUrl(item.url)" :href="safeSourceUrl(item.url)!" target="_blank" rel="noopener noreferrer">{{ item.title || 'View original source' }}<AppIcon name="external" :size="13" /></a><span v-else>{{ item.title || 'Source link unavailable' }}</span></h3><p class="evidence-excerpt"><template v-for="(part, i) in highlight(item.text || '')" :key="i"><mark v-if="part.marked">{{ part.text }}</mark><template v-else>{{ part.text }}</template></template></p><div class="evidence-card-bottom"><span>{{ toolNames[item.source_tool || ''] || 'Retrieved excerpt' }}</span><button class="button text" :disabled="!item.docid" @click="item.docid && emit('openDocument', item.docid)">Open document<AppIcon name="book" :size="14" /></button></div></article>
        <p v-if="!evidence.length" class="empty-message">No sources match this filter. Try another name or phrase.</p>
      </template>
      <div v-else class="inspector-empty"><span class="empty-illustration"><AppIcon name="file" :size="34" /></span><h3>{{ running ? 'Waiting for sources' : 'The evidence will appear here' }}</h3><p>{{ running ? 'Sources will be available when the run completes.' : 'Start a run to review source excerpts and open the documents behind them.' }}</p><div class="evidence-guide"><span><AppIcon name="check" :size="15" />Source excerpts and references</span><span><AppIcon name="check" :size="15" />Tool calls and results</span><span><AppIcon name="check" :size="15" />Skills loaded for the question</span></div></div>
    </div>
    <div v-show="tab === 'trace'" id="panel-trace" role="tabpanel" aria-labelledby="tab-trace" class="inspector-content"><ol v-if="result?.trajectory.length" class="trace-list"><li v-for="(event, index) in result.trajectory" :key="index" class="trace-item"><span class="trace-dot" :class="describeEvent(event).tone" /><h3>{{ describeEvent(event).title }}</h3><p>{{ describeEvent(event).detail }}</p><details><summary>View event details</summary><pre>{{ JSON.stringify(event, null, 2) }}</pre></details></li></ol><div v-else class="inspector-empty"><span class="empty-illustration"><AppIcon name="network" :size="34" /></span><h3>Review the search activity</h3><p>Review tool calls, their results, and the checks applied to the answer.</p></div></div>
    <div v-show="tab === 'skills'" id="panel-skills" role="tabpanel" aria-labelledby="tab-skills" class="inspector-content"><p class="skills-description">{{ result ? 'Review which skills were loaded during this run.' : 'Skills are selected automatically to match the question and search progress.' }}</p><article v-for="skill in skills" :key="skill.id" class="skill-card" :class="{ selected: result?.selected_skills.includes(skill.id) }"><span class="skill-icon"><AppIcon :name="skill.icon" :size="20" /></span><div><h3>{{ skillNames[skill.id] }}</h3><p>{{ skill.description }}</p><span class="skill-state">{{ result?.selected_skills.includes(skill.id) ? 'Loaded' : result ? 'Not loaded' : 'Selected as needed' }}</span></div></article></div>
    <div class="inspector-footer"><AppIcon name="shield" :size="15" />Open a source to review the surrounding context.</div>
  </aside>
</template>

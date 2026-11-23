<script setup lang="ts">
import { nextTick, onUnmounted, ref } from 'vue'
import type { DocumentPage } from '../types'
import { fetchDocument } from '../services/api'
import AppIcon from './AppIcon.vue'

const dialog = ref<HTMLDialogElement>()
const docid = ref('')
const page = ref<DocumentPage | null>(null)
const loading = ref(false)
const error = ref('')
const previousOffsets = ref<number[]>([])
let controller: AbortController | undefined
async function load(offset: number) {
  controller?.abort(); const current = new AbortController(); controller = current
  loading.value = true; error.value = ''
  try { const result = await fetchDocument(docid.value, offset, current.signal); if (controller === current) page.value = result }
  catch (e) { if (controller === current && !(e instanceof DOMException && e.name === 'AbortError')) error.value = e instanceof Error ? e.message : 'The source document could not be loaded.' }
  finally { if (controller === current) loading.value = false }
}
async function open(id: string) { docid.value = id; page.value = null; previousOffsets.value = []; await nextTick(); dialog.value?.showModal(); void load(0) }
function close() { controller?.abort(); dialog.value?.close() }
async function next() { if (!page.value || page.value.next_offset == null) return; const oldOffset = page.value.start_offset; await load(page.value.next_offset); if (!error.value) previousOffsets.value.push(oldOffset) }
async function previous() { const offset = previousOffsets.value.at(-1); if (offset == null) return; await load(offset); if (!error.value) previousOffsets.value.pop() }
onUnmounted(() => controller?.abort())
defineExpose({ open })
</script>

<template>
  <dialog ref="dialog" class="document-dialog" aria-labelledby="document-title" @click="event => { if (event.target === dialog) close() }" @cancel="controller?.abort()">
    <header class="dialog-header"><div><span class="eyebrow">SOURCE DOCUMENT</span><h2 id="document-title">Document {{ docid }}</h2></div><button class="icon-button" aria-label="Close document" @click="close"><AppIcon name="close" /></button></header>
    <div class="document-body" :aria-busy="loading"><p v-if="loading" class="document-notice" role="status">Loading the source document…</p><p v-else-if="error" class="form-error" role="alert">{{ error }}</p><pre v-else>{{ page?.content || 'No source text is available in this range.' }}</pre></div>
    <footer class="dialog-footer"><span>{{ page ? `Characters ${page.start_offset.toLocaleString('en-US')}–${page.end_offset.toLocaleString('en-US')}` : 'Paginated source document' }}</span><div><button class="button secondary" :disabled="loading || !previousOffsets.length" @click="previous">Previous page</button><button class="button secondary" :disabled="loading || !page?.has_more || !!error" @click="next">Next page</button></div></footer>
  </dialog>
</template>

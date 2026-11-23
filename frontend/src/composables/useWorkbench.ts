import { computed, onMounted, onUnmounted, ref } from 'vue'
import { fetchHealth, runQuestion } from '../services/api'
import type { Health, RunRecord, RunResult, View } from '../types'

const HISTORY_KEY = 'bm25-workbench.history.v1'
function readHistory(): RunRecord[] {
  try {
    const value: unknown = JSON.parse(sessionStorage.getItem(HISTORY_KEY) || '[]')
    return Array.isArray(value) ? value.filter((r) => r && typeof r.id === 'string' && typeof r.question === 'string' && r.result && typeof r.result.answer === 'string' && Array.isArray(r.result.evidence) && Array.isArray(r.result.trajectory)).slice(0, 10) : []
  } catch { return [] }
}

export function useWorkbench() {
  const view = ref<View>('search')
  const question = ref('')
  const submittedQuestion = ref('')
  const confirmed = ref(false)
  const running = ref(false)
  const elapsed = ref(0)
  const error = ref('')
  const health = ref<Health | null>(null)
  const healthError = ref(false)
  const checkingHealth = ref(false)
  const result = ref<RunResult | null>(null)
  const history = ref<RunRecord[]>(readHistory())
  const storageNotice = ref('')
  let healthTimer: ReturnType<typeof setInterval> | undefined
  let elapsedTimer: ReturnType<typeof setInterval> | undefined
  const ready = computed(() => !healthError.value && !!health.value?.llm_configured && !!health.value?.bm25_ready)
  const canRun = computed(() => ready.value && confirmed.value && !!question.value.trim() && !running.value && question.value.length <= 6000)

  async function refreshHealth() {
    if (checkingHealth.value) return
    checkingHealth.value = true
    try { health.value = await fetchHealth(); healthError.value = false }
    catch { healthError.value = true; health.value = null }
    finally { checkingHealth.value = false }
  }

  function persistHistory() {
    try { sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history.value)); storageNotice.value = '' }
    catch { storageNotice.value = 'Browser storage is full. This run is available on the current page but may be lost when you refresh.' }
  }

  async function submit() {
    if (running.value) return
    if (!question.value.trim()) { error.value = 'Add a question before starting a run.'; return }
    if (!ready.value) { error.value = 'The service isn’t ready. Open System to check the connection and configuration.'; return }
    if (!confirmed.value) { error.value = 'Confirm API usage to start this run.'; return }
    if (question.value.length > 6000) { error.value = 'Questions can contain up to 6,000 characters.'; return }
    const query = question.value.trim()
    running.value = true; error.value = ''; result.value = null; view.value = 'search'
    submittedQuestion.value = query; elapsed.value = 0
    const started = performance.now()
    elapsedTimer = setInterval(() => { elapsed.value = (performance.now() - started) / 1000 }, 250)
    try {
      const response = await runQuestion(query)
      result.value = response
      history.value = [{ id: crypto.randomUUID(), question: query, createdAt: new Date().toISOString(), result: response }, ...history.value].slice(0, 10)
      persistHistory()
    } catch (e) { error.value = e instanceof Error ? e.message : 'The run could not be completed. Try again.' }
    finally { clearInterval(elapsedTimer); running.value = false; confirmed.value = false; void refreshHealth() }
  }

  function newSearch() {
    if (running.value) return
    view.value = 'search'; question.value = ''; submittedQuestion.value = ''; result.value = null; error.value = ''; confirmed.value = false
  }
  function openRecord(record: RunRecord) {
    if (running.value) return
    question.value = record.question; submittedQuestion.value = record.question; result.value = record.result; error.value = ''; confirmed.value = false; view.value = 'search'
  }
  function clearHistory() { history.value = []; persistHistory() }

  onMounted(() => { void refreshHealth(); healthTimer = setInterval(refreshHealth, 20000) })
  onUnmounted(() => { clearInterval(healthTimer); clearInterval(elapsedTimer) })
  return { view, question, submittedQuestion, confirmed, running, elapsed, error, health, healthError, checkingHealth, result, history, storageNotice, ready, canRun, refreshHealth, submit, newSearch, openRecord, clearHistory }
}

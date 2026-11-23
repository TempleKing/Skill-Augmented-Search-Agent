import type { RunResult, TraceEvent } from '../types'

export const formatNumber = (value?: number | null) => value == null ? 'Not reported' : new Intl.NumberFormat('en-US').format(value)
export const formatTime = (seconds: number) => seconds >= 60 ? `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s` : `${seconds.toFixed(1)}s`
export const toolNames: Record<string, string> = {
  search_documents: 'Search documents', open_document: 'Open document', find_in_document: 'Find in document',
}
const toolResultNames: Record<string, string> = {
  search_documents: 'Search results', open_document: 'Source document', find_in_document: 'Passage matches',
}
export const skillNames: Record<string, string> = {
  'multi-hop-search': 'Multi-hop search', 'entity-disambiguation': 'Entity disambiguation', 'search-recovery': 'Search recovery',
}
export function resultStatus(result: RunResult): { label: string; tone: string; detail: string } {
  if (result.answer === 'Unknown' || ['insufficient_evidence', 'incomplete_evidence'].includes(result.termination_reason)) {
    return { label: 'Insufficient evidence', tone: 'warning', detail: 'The retrieved evidence does not fully support an answer. Review the sources or refine the question.' }
  }
  if (result.termination_reason === 'token_budget_exhausted') return { label: 'Token threshold reached', tone: 'warning', detail: 'This run reached its token threshold. Review the evidence collected before the run stopped.' }
  if (result.termination_reason === 'completed') return { label: 'Completed', tone: 'success', detail: 'Review the source excerpts and activity to assess how this answer was established.' }
  return { label: 'Run ended', tone: 'neutral', detail: 'Review the result alongside its source evidence and termination details.' }
}

export function describeEvent(event: TraceEvent): { title: string; detail: string; tone: string } {
  if (event.type === 'tool_call') return { title: toolNames[event.tool || ''] || 'Tool request', detail: String(event.input?.query || event.input?.keyword || (event.input?.docid ? `Document ${event.input.docid}` : 'Request issued')), tone: 'neutral' }
  if (event.type === 'tool_result') return { title: toolResultNames[event.tool || ''] || 'Tool response', detail: event.executed === false ? 'Request blocked; not executed' : event.status === 'error' ? 'The tool returned an error. Open the event details to inspect it.' : Array.isArray(event.output) ? `${event.output.length} ${event.output.length === 1 ? 'result' : 'results'} returned` : 'Tool response received', tone: event.status === 'error' ? 'warning' : 'success' }
  if (event.type === 'skill_loaded') return { title: 'Research skill loaded', detail: skillNames[event.skill || ''] || event.skill || 'Skill name not returned', tone: 'neutral' }
  if (event.type === 'answer_validation') return { title: 'Answer validation', detail: event.valid ? 'The answer passed evidence validation.' : 'The current answer has not passed evidence validation.', tone: event.valid ? 'success' : 'warning' }
  if (event.type === 'search_progress') return { title: 'Search progress', detail: String(event.reason || 'Evidence and clue state updated'), tone: 'neutral' }
  if (event.type === 'llm_error') return { title: 'Model request error', detail: 'Open the event details to review the error and any retries.', tone: 'warning' }
  return { title: event.type === 'budget_stop' ? 'Run stopped at a budget check' : 'Run state updated', detail: String(event.reason || 'See the event details.'), tone: 'neutral' }
}

export function safeSourceUrl(value?: string): string | null {
  try { const url = new URL(value || ''); return ['https:', 'http:'].includes(url.protocol) ? url.href : null } catch { return null }
}

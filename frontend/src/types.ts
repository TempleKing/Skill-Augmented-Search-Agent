export interface Health {
  status: string
  llm_configured: boolean
  bm25_ready: boolean
  corpus_documents: number | null
  retrieval_method: 'bm25'
  retrieval_top_k: number
  retrieval_candidate_k: number
  retrieval_rrf_k: number
  model: string
  max_agent_steps: number
  max_tool_calls: number
  max_stalled_searches: number
  max_total_tokens_per_run: number
}

export interface Evidence {
  evidence_id?: string
  docid?: string
  title?: string
  text: string
  url?: string
  source_tool?: string
  query?: string
  rank?: number
  score?: number
  start_offset?: number
  end_offset?: number
}

export interface TraceEvent {
  type: string
  tool?: string
  skill?: string
  reason?: string
  status?: string
  valid?: boolean
  executed?: boolean
  timestamp?: string
  input?: Record<string, unknown>
  output?: unknown
  [key: string]: unknown
}

export interface RunResult {
  answer: string
  termination_reason: string
  selected_skills: string[]
  evidence: Evidence[]
  trajectory: TraceEvent[]
  tool_calls: number
  total_tokens: number
  elapsed_seconds: number
}

export interface RunRecord {
  id: string
  question: string
  createdAt: string
  result: RunResult
}

export interface DocumentPage {
  docid: string
  content: string
  start_offset: number
  end_offset: number
  next_offset: number | null
  has_more: boolean
}

export type View = 'search' | 'history' | 'baseline'
export type InspectorTab = 'evidence' | 'trace' | 'skills'

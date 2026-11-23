import type { DocumentPage, Health, RunResult } from '../types'

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message) }
}

async function request<T>(url: string, options: RequestInit = {}): Promise<T> {
  let response: Response
  try {
    response = await fetch(url, { cache: 'no-store', ...options })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError('Couldn’t reach the backend. Check that the service is running, then try again.', 0)
  }
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const messages: Record<number, string> = {
      409: 'A run is already in progress. Wait for it to finish, then try again.',
      503: 'The service isn’t ready. Check the model configuration and corpus index in System.',
    }
    throw new ApiError(messages[response.status] || body?.error || 'The request could not be completed. Try again.', response.status)
  }
  if (!body || typeof body !== 'object') throw new ApiError('The service returned a response that could not be read.', response.status)
  return body as T
}

export const fetchHealth = () => request<Health>('/api/health')
export const runQuestion = (question: string) => request<RunResult>('/api/runs', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question, confirm_api_use: true }),
})
export const fetchDocument = (docid: string, offset: number, signal?: AbortSignal) =>
  request<DocumentPage>(`/api/documents/${encodeURIComponent(docid)}?offset=${offset}&limit=4000`, { signal })

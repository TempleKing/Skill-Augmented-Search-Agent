import { expect, test, type Page } from '@playwright/test'
import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'

// Fixtures are intercepted only by the test runner, never shipped as application results.
const health = { status: 'ok', llm_configured: true, bm25_ready: true, corpus_documents: 100195, retrieval_method: 'bm25', retrieval_top_k: 8, retrieval_candidate_k: 50, retrieval_rrf_k: 60, model: 'deepseek-v4-flash', max_agent_steps: 10, max_tool_calls: 15, max_stalled_searches: 3, max_total_tokens_per_run: 95000 }
const result = { answer: 'Example Researcher', termination_reason: 'completed', selected_skills: ['multi-hop-search'], evidence: [
  { evidence_id: 'e1', docid: 'sample-1', title: 'Research archive', text: 'A source identifies [Example Researcher].', url: 'https://example.test/archive', source_tool: 'search_documents', rank: 1 },
  { evidence_id: 'e2', docid: 'sample-2', title: 'Unsafe link remains plain text', text: '<img src=x onerror="window.injected=true">', url: 'javascript:alert(1)', source_tool: 'open_document', rank: 2 },
], trajectory: [{ type: 'tool_call', tool: 'search_documents', input: { query: 'sample archive' } }, { type: 'tool_result', tool: 'search_documents', status: 'success', executed: true, output: [{ docid: 'sample-1' }] }, { type: 'answer_validation', valid: true }], tool_calls: 1, total_tokens: 1234, elapsed_seconds: 2.5 }

async function navigate(page: Page, name: string) {
  const menu = page.getByRole('button', { name: 'Open navigation' })
  if (await menu.isVisible()) await menu.click()
  await page.getByRole('navigation').getByRole('button', { name: new RegExp(name) }).click()
}
async function completeRun(page: Page) {
  await page.route('**/api/runs', route => route.fulfill({ json: result }))
  await page.getByRole('textbox', { name: 'Research question', exact: true }).fill('Find the researcher with supporting sources.')
  await page.getByRole('checkbox').check()
  await page.getByRole('button', { name: /Start research/ }).click()
  await expect(page.getByRole('heading', { name: 'Example Researcher' })).toBeVisible()
}

test.beforeEach(async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({ json: health }))
  await page.goto('/')
  await expect(page.getByRole('button', { name: 'View service status' })).toContainText('Ready to run')
})

test('uses PolyU red, shows honest empty states, supports mobile without overflow', async ({ page }, info) => {
  await expect(page.getByRole('heading', { name: /What would you like/ })).toBeVisible()
  expect(await page.locator('html').getAttribute('lang')).toBe('en')
  expect(await page.locator('body').innerText()).not.toMatch(/[\u3400-\u9fff]/)
  await expect(page.getByRole('button', { name: /Start research/ })).toBeDisabled()
  await expect(page.getByRole('heading', { name: 'The evidence will appear here' })).toBeVisible()
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--brand').trim())).toBe('#a02337')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy()
  if (process.env.UI_ARTIFACT_DIR) {
    await mkdir(process.env.UI_ARTIFACT_DIR, { recursive: true })
    await page.screenshot({ path: join(process.env.UI_ARTIFACT_DIR, `${info.project.name}-empty.png`), fullPage: true })
  }
})

test('sample fills the question without calling the paid endpoint', async ({ page }) => {
  let calls = 0
  await page.route('**/api/runs', route => { calls++; return route.fulfill({ json: result }) })
  await page.getByRole('button', { name: /Publication history/ }).click()
  await expect(page.getByRole('textbox', { name: 'Research question', exact: true })).toHaveValue(/Pride and Prejudice/)
  expect(calls).toBe(0)
  await expect(page.getByRole('button', { name: /Start research/ })).toBeDisabled()
})

test('requires confirmation, prevents duplicate requests, and clears confirmation after completion', async ({ page }) => {
  let release: () => void = () => {}
  const pending = new Promise<void>(resolve => { release = resolve })
  let calls = 0
  await page.route('**/api/runs', async route => { calls++; await pending; await route.fulfill({ json: result }) })
  await page.getByRole('textbox', { name: 'Research question', exact: true }).fill('Find a researcher.')
  await page.getByRole('checkbox').check()
  await page.getByRole('button', { name: /Start research/ }).click()
  await expect(page.getByRole('heading', { name: 'Working through your question' })).toBeVisible()
  await page.locator('form').evaluate(form => { (form as HTMLFormElement).requestSubmit(); (form as HTMLFormElement).requestSubmit() })
  await expect(page.getByRole('checkbox')).toBeDisabled()
  await expect(page.getByRole('button', { name: /Running/, exact: false })).toBeDisabled()
  release()
  await expect(page.getByRole('heading', { name: 'Example Researcher' })).toBeVisible()
  expect(calls).toBe(1)
  await expect(page.getByRole('checkbox')).not.toBeChecked()
  await expect(page.getByRole('button', { name: /Start research/ })).toBeDisabled()
})

test('shows evidence safely, filters sources, and supports keyboard-operated inspector tabs', async ({ page }, info) => {
  await completeRun(page)
  if (info.project.name === 'desktop') {
    await expect(page.getByRole('heading', { name: 'Example Researcher' })).toBeInViewport()
    expect(await page.locator('.recent-list button').first().evaluate(button => button.getBoundingClientRect().right <= document.querySelector('.sidebar')!.getBoundingClientRect().right)).toBeTruthy()
  }
  await expect(page.locator('.evidence-card')).toHaveCount(2)
  await expect(page.locator('.evidence-excerpt img')).toHaveCount(0)
  await expect(page.locator('a[href^="javascript:"]')).toHaveCount(0)
  await expect(page.locator('mark')).toContainText('Example Researcher')
  await page.getByLabel('Filter evidence').fill('Research archive')
  await expect(page.locator('.evidence-card')).toHaveCount(1)
  await expect(page.locator('.result-count')).toHaveText('1 source excerpt')
  await page.getByLabel('Filter evidence').clear()
  await page.getByRole('tab', { name: /Evidence/ }).focus()
  await page.keyboard.press('ArrowRight')
  await expect(page.getByRole('tab', { name: 'Activity' })).toBeFocused()
  await expect(page.getByRole('tabpanel', { name: 'Activity' })).toContainText('The answer passed evidence validation.')
  await page.getByRole('tab', { name: 'Skills' }).click()
  await expect(page.getByRole('tabpanel', { name: 'Skills' })).toContainText('Loaded')
  await page.getByRole('tab', { name: /Evidence/ }).click()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy()
  if (process.env.UI_ARTIFACT_DIR) await page.screenshot({ path: join(process.env.UI_ARTIFACT_DIR, `${info.project.name}-result.png`), fullPage: true })
})

test('reads document pages with the returned offsets', async ({ page }) => {
  const offsets: number[] = []
  await page.route('**/api/documents/**', route => {
    const offset = Number(new URL(route.request().url()).searchParams.get('offset'))
    offsets.push(offset)
    return route.fulfill({ json: { docid: 'sample-1', content: offset ? 'Original content on page two' : 'Original content on page one', start_offset: offset, end_offset: offset + 4000, has_more: offset === 0, next_offset: offset === 0 ? 4000 : null } })
  })
  await completeRun(page)
  await page.getByRole('button', { name: /Open document/ }).first().click()
  await expect(page.getByRole('dialog')).toContainText('Original content on page one')
  await page.getByRole('button', { name: 'Next page' }).click()
  await expect(page.getByRole('dialog')).toContainText('Original content on page two')
  await page.getByRole('button', { name: 'Previous page' }).click()
  await expect(page.getByRole('dialog')).toContainText('Original content on page one')
  expect(offsets).toEqual([0, 4000, 0])
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).not.toBeVisible()
})

test('restores session history without a new model request, and clears local records', async ({ page }) => {
  await completeRun(page)
  await navigate(page, 'History')
  await expect(page.locator('.history-card')).toHaveCount(1)
  await page.locator('.history-card').click()
  await expect(page.getByRole('heading', { name: 'Example Researcher' })).toBeVisible()
  await page.reload()
  await navigate(page, 'History')
  await expect(page.locator('.history-card')).toHaveCount(1)
  await page.getByRole('button', { name: 'Clear history' }).click()
  await expect(page.getByRole('heading', { name: 'No completed runs yet' })).toBeVisible()
})

test('shows request failures instead of a fabricated answer', async ({ page }) => {
  await page.route('**/api/runs', route => route.fulfill({ status: 503, json: { error: 'Unavailable' } }))
  await page.getByRole('textbox', { name: 'Research question', exact: true }).fill('Find a researcher.')
  await page.getByRole('checkbox').check()
  await page.getByRole('button', { name: /Start research/ }).click()
  await expect(page.getByRole('alert')).toContainText('The service isn’t ready')
  await expect(page.getByRole('heading', { name: 'The run didn’t finish' })).toBeVisible()
  await expect(page.locator('.history-card')).toHaveCount(0)
})

test('baseline details use server configuration and distinguish the historical score', async ({ page }) => {
  await navigate(page, 'System')
  await expect(page.getByRole('heading', { name: /Research configuration/ })).toBeVisible()
  await expect(page.locator('.configuration-grid')).toContainText('95,000')
  await expect(page.locator('.readiness-grid')).toContainText('100,195')
  await expect(page.locator('.baseline-evaluation')).toContainText('not a new evaluation or a full-set score')
  await page.route('**/api/health', route => route.fulfill({ json: { ...health, llm_configured: false, bm25_ready: false, corpus_documents: null } }))
  await page.getByRole('button', { name: 'Refresh status' }).click()
  await expect(page.getByRole('button', { name: 'View service status' })).toContainText('Configuration needed')
  await expect(page.locator('.readiness-grid')).toContainText('Document count unavailable')
})

test('clears stale readiness and counts when the health connection is lost', async ({ page }) => {
  await navigate(page, 'System')
  await page.route('**/api/health', route => route.abort('connectionfailed'))
  await page.getByRole('button', { name: 'Refresh status' }).click()
  await expect(page.getByRole('button', { name: 'View service status' })).toContainText('Connection unavailable')
  await expect(page.locator('.readiness-grid')).toContainText('Document count unavailable')
  await expect(page.locator('.readiness-grid')).not.toContainText('100,195')
})

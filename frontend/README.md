# TRACE — Evidence-first research

A modern, English-only research workspace for the BM25 baseline. Vue 3 and TypeScript keep presentation, API access, state, and shared types separate.

## Design

Graphite surfaces, self-hosted Inter typography, monospaced metadata, and restrained red accents create a focused research-tool interface. The first screen introduces the task; once a run starts, the heading and editor compact to give the result more space. Desktop uses a persistent navigation rail and evidence inspector. Mobile uses a single column and a collapsible navigation drawer.

| Token | Value | Purpose |
| --- | --- | --- |
| PolyU Red | `#A02337` | Primary action and brand mark |
| Accent text | `#ED9AAA` | Readable red-derived text on dark surfaces |
| Navigation | `#0B0C10` | Persistent navigation rail |
| Page | `#111318` | Main workspace |
| Surface | `#191C23` | Input and result panels |
| Main text | `#F1F2F5` | Primary content |
| Secondary text | `#A0A4B2` | Supporting information |
| Border | `#2A2D36` | Structural separation |

PolyU Red is referenced from the official [UBSN Branding Guide](https://www.polyu.edu.hk/ubsn/docdrive/download/UBSN_Branding_Guide.pdf). The product uses its own name and symbol. The [Inter variable font](https://github.com/rsms/inter) is self-hosted under SIL OFL 1.1; its license is included in `public/fonts/LICENSE-Inter.txt`.

Tokens live in `src/styles/tokens.css`. Focus indicators, keyboard-operated tabs, native document dialogs, reduced-motion support, and responsive layouts are retained. All application-owned labels, errors, loading states, help text, and controls are English. User questions and returned source text are displayed faithfully rather than automatically translated.

Interface copy follows the [TRACE writing guidelines](COPY_GUIDELINES.md): direct, conversational English; precise research terminology; explicit limits; and actionable state and error messages.

## Development

Requires Node.js 22.12+; Node.js 24 is recommended. Versions are pinned by `package-lock.json`.

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` to the Python backend at `http://127.0.0.1:4173`. Configure the model and corpus index as described in the repository README, then run `python web_server.py` from the repository root in another terminal.

## Production build

```powershell
cd frontend
npm run build
cd ..
python web_server.py
```

Open `http://127.0.0.1:4173`. Python serves the compiled `frontend/dist/` directory. Build output is local and ignored by Git. The standalone Vite preview server does not proxy the Python API.

## Modules

```text
src/components/    Question editor, answers, inspector, icons, source dialog
src/composables/   Run state, confirmation, health checks, session history
src/services/      Typed API requests and error handling
src/utils/         Status copy, number/time formatting, safe source links
src/styles/        Design tokens and responsive styles
src/types.ts       Backend response contracts
e2e/              Browser regression tests and test-only API fixtures
```

## Interaction contracts

- Starter prompts fill the editor without making a model request.
- Each run requires explicit API confirmation. Confirmation resets when the run completes.
- The UI prevents duplicate submission. The backend also serializes runs.
- The existing API returns a complete result when a run finishes. The UI displays actual waiting time without inventing intermediate progress or sources.
- Answers, evidence, activity, and loaded skills come from the API. Insufficient evidence is reported honestly as Unknown.
- Source documents use backend-provided offsets for pagination. Closing the source dialog aborts an outstanding read.
- Returned text is rendered safely; source links accept only HTTP and HTTPS.
- The last ten completed runs stay in `sessionStorage`. Export downloads local JSON. Neither action uploads results to the repository.
- Keys remain on the backend. Health checks inspect local configuration and do not send paid model requests.
- The backend does not support cancellation; closing the page does not guarantee that an existing run stops.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Configuration, actual index size, and baseline parameters |
| `POST /api/runs` | Submit `{question, confirm_api_use: true}` |
| `GET /api/documents/{docid}?offset=0&limit=4000` | Read a source document page |

## Verification

```powershell
npm run typecheck
npm run build
npm run test:e2e
```

Browser tests use an installed Google Chrome with desktop and mobile viewports. Model responses are intercepted exclusively in the test runner; no fixtures are bundled as application results and no model usage is billed. Run Python tests from the repository root with `python -m pytest -q`.

Set `UI_ARTIFACT_DIR` to save screenshots locally. Browser reports, screenshots, dependencies, and generated bundles are excluded from Git. Result screenshots from browser tests contain synthetic fixtures and must not be presented as measured model performance.

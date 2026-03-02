# Arkain Atlas — CRE Deal Intelligence Platform

**Truth Accretion Engine v3.3.1** powering a full-stack deal analysis platform.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Next.js Frontend (web/)                                     │
│  ├── 3-pane UI: Chat + Artifacts + Admin                     │
│  ├── Streaming NDJSON for real-time analysis                 │
│  └── Strategic intake wizard                                 │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Backend (engine/api_server.py)                       │
│  ├── 104 v1 endpoints (legacy)                               │
│  ├── 7 v3 endpoints (new engine)                             │
│  ├── NDJSON streaming at /api/v1/stream                      │
│  └── Strategic dispatcher at /api/v1/strategic               │
├─────────────────────────────────────────────────────────────┤
│  V3 Truth Accretion Engine (engine/brain/)                   │
│  ├── 17-agent cognitive pipeline                             │
│  ├── Correlated Monte Carlo (Cholesky decomposition)         │
│  ├── Capital stack waterfall + stress testing                │
│  ├── Scenario trees with CRRA utility                        │
│  ├── Cost-aware epistemic foraging (EIG/$)                   │
│  ├── Hash-chained audit ledger                               │
│  ├── Active inference + SEAL/CECA + OODA loops               │
│  └── Deterministic replay                                    │
├─────────────────────────────────────────────────────────────┤
│  Domain Modules                                              │
│  ├── EGM gaming data (IL Gaming Board connector)             │
│  ├── Financial calculators (IRR, DSCR, amortization)         │
│  ├── Contract engine (Monte Carlo, gaming contracts)         │
│  ├── Real estate pipeline (7-stage deal filter)              │
│  ├── Portfolio analytics (concentration, dashboard)          │
│  ├── Construction (drawings, MEP, structural)                │
│  └── Strategic intelligence (5-stage LLM pipeline)           │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure                                              │
│  ├── PostgreSQL (SQLAlchemy + Alembic)                       │
│  ├── Redis (cache + job queue)                               │
│  └── Multi-LLM (Anthropic + OpenAI routing)                  │
└─────────────────────────────────────────────────────────────┘
```

## Railway Deployment

### Services

| Service | Build | Port |
|---------|-------|------|
| **api** | `Dockerfile` | 8000 |
| **web** | `web/Dockerfile` | 3000 |
| **Postgres** | Railway managed | 5432 |
| **Redis** | Railway managed | 6379 |

### Environment Variables

**API Service:**
```
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}
JWT_SECRET=<generate-random>
AUTH_ENABLED=false
CORS_ORIGINS=*
ANTHROPIC_API_KEY=<your-key>
OPENAI_API_KEY=<your-key>
```

**Web Service:**
```
NEXT_PUBLIC_API_BASE_URL=/api/v1
API_BACKEND_URL=http://api.railway.internal:8000
```

### Deploy

1. Push this repo to GitHub
2. Connect repo to Railway
3. Railway auto-detects `Dockerfile` for API, `web/Dockerfile` for frontend
4. Set environment variables per above
5. Deploy

## V3 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/stream` | NDJSON streaming chat |
| POST | `/api/v1/strategic` | Strategic analysis dispatcher |
| POST | `/api/v3/deals/run` | Full v3 pipeline |
| POST | `/api/v3/deals/monte-carlo` | Correlated Monte Carlo |
| POST | `/api/v3/deals/capital-stack/waterfall` | Capital stack waterfall |
| POST | `/api/v3/deals/capital-stack/stress-test` | Stress test capital structure |
| POST | `/api/v3/deals/scenarios/evaluate` | Scenario tree evaluation |
| GET | `/api/v3/engine/health` | V3 engine health check |

## Local Development

```bash
# Backend
docker-compose up -d postgres redis
pip install -r requirements.txt
python -m engine.api_server

# Frontend
cd web && npm install && npm run dev

# Engine smoke test
cd engine/brain && python smoke_test.py
```

## Stats

- **112 Python files**, 44,292 lines
- **26 TypeScript files** (frontend)
- **39 brain modules** (v3 engine)
- **111 API endpoints** total
- **17 AI agents** with specific task assignments
- **18 data tools** registered

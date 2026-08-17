# API and Frontend Contract

The FastAPI application is currently a read-only HTTP surface. It does not contact Binance or run ingestion, decisions, simulations, or refinement from GET requests. Parameters use UTC timestamps and complete instrument identity where ambiguity is possible. Decimal financial values remain exact serialized values.

## Endpoint inventory

This inventory lists all 93 routes exposed by `main.py` (`auth_router`, `health_router`, `market_router`, `paper_router`, `paper_deployment_router`, `dashboard_router`, `optimization_router`, `protected_ml_router`), grouped by subsystem. `app/ml/protected/router.py` mounts a second `APIRouter(prefix="/optimization")`, so protected-validation and development-re-audit routes share the `/optimization` path space with `app/optimization/router.py` rather than nesting under it. FastAPI's generated `/docs`, `/redoc`, and `/openapi.json` (disabled once auth enforcement is on) are the schema-level reference; this inventory intentionally summarizes rather than duplicates Pydantic models.

| Group | Method + path | Purpose | Mutation |
| --- | --- | --- | --- |
| Auth | `POST /auth/login` | Authenticate admin credentials; set session and CSRF cookies | yes |
| Auth | `GET /auth/me` | Return current session principal; refresh CSRF header | no |
| Auth | `POST /auth/logout` | Clear session and CSRF cookies (CSRF-protected) | yes |
| Health | `GET /health` | Process liveness | no |
| Health | `GET /health/database` | PostgreSQL connectivity; 503 when unreachable | yes when enforcement is enabled |
| Market | `GET /market/instruments` | Catalog identities/capabilities/display metadata | no |
| Market | `GET /market/candles` | Persisted candles for an instrument over a UTC range | no |
| Market | `GET /market/status` | Binance Spot 5m count/latest/freshness | no |
| Market | `GET /market/features/latest` | Latest complete Feature snapshot | no |
| Market | `GET /market/scores/latest` | Latest baseline technical Score | no |
| Market | `GET /market/scores` | Score history | no |
| Market | `GET /market/decisions/latest` | Latest Decision Snapshot | no |
| Market | `GET /market/decisions` | Decision Snapshot history | no |
| Dashboard | `GET /dashboard/overview` | One composed catalog/market/Paper overview | no |
| Paper | `GET /paper/account` | Active run account summary | no |
| Paper | `GET /paper/portfolio` | Current persisted-candle valuation (optional `run_id`) | no |
| Paper | `GET /paper/performance` | Equity-history performance (optional `run_id`) | no |
| Paper | `GET /paper/evaluation` | Strategy vs. buy-and-hold comparison (optional `run_id`) | no |
| Paper | `GET /paper/equity-snapshots` | Immutable snapshot history, limit/offset/order | no |
| Paper | `GET /paper/equity-snapshots/{snapshot_id}` | One snapshot and its positions | no |
| Paper | `GET /paper/runs` | Active and stopped runs | no |
| Paper | `GET /paper/runs/{run_id}` | One historical run | no |
| Paper | `GET /paper/positions` | Positions, including open-at-stop history | no |
| Paper | `GET /paper/trades` | Immutable simulated trades | no |
| Paper | `GET /paper/decision-cycles` | Auditable cycle history | no |
| Paper | `GET /paper/decision-cycles/{cycle_id}` | One decision cycle | no |
| Paper Deployments | `GET /paper/deployments` | Page deployments by lifecycle/strategy kind | no |
| Paper Deployments | `GET /paper/deployments/{deployment_id}` | One deployment: cash, open positions, artifact | no |
| Paper Deployments | `POST /paper/deployments` | Create a deployment from a passed final validation | yes |
| Paper Deployments | `POST /paper/deployments/{deployment_id}/{action}` | `pause`/`resume`/`archive`/`make-primary` | yes |
| Optimization (legacy) | `GET /optimization/refinement-runs` | Legacy refinement run summaries | no |
| Optimization (legacy) | `GET /optimization/refinement-runs/{run_id}` | Bounded run detail | no |
| Optimization (legacy) | `GET /optimization/refinement-runs/{run_id}/evaluations` | Paged evaluations | no |
| Optimization (legacy) | `GET /optimization/candidates` | Canonical candidate profiles | no |
| Optimization (legacy) | `GET /optimization/candidates/{stable_candidate_id}` | Candidate definition and paged evidence | no |
| Optimization (legacy) | `GET /optimization/refinement-runs/{run_id}/walk-forward` | Linked walk-forward summary | no |
| Optimization (legacy) | `GET /optimization/walk-forward-runs/{run_id}` | Walk-forward run detail | no |
| Optimization (legacy) | `GET /optimization/walk-forward-runs/{run_id}/candidates` | Candidate page, optional classification | no |
| Optimization (legacy) | `GET /optimization/walk-forward-runs/{run_id}/selections` | Selection page | no |
| Optimization (legacy) | `GET /optimization/walk-forward-runs/{run_id}/candidates/{candidate_id}/windows` | Per-window evidence | no |
| Predictive Edge Map | `GET /optimization/predictive-edge-runs` | Runs (`actionable`/`diagnostic`), instrument/status filters | no |
| Predictive Edge Map | `GET /optimization/predictive-edge-runs/{run_id}` | Run detail | no |
| Predictive Edge Map | `GET /optimization/predictive-edge-runs/{run_id}/edges` | Edge evidence page | no |
| Predictive Edge Map | `GET /optimization/predictive-edge-runs/{run_id}/edges/{edge_id}/windows` | Edge window evidence | no |
| Predictive Edge Map | `GET /optimization/predictive-edge-runs/{run_id}/hypotheses` | Strategy hypotheses | no |
| Predictive Edge Map | `GET /optimization/predictive-edge-runs/{run_id}/diagnostics` | Diagnostic statistics | no |
| Predictive Edge Map | `GET /optimization/predictive-edge-runs/{run_id}/near-misses` | Near-miss evidence (`actionable`/`diagnostic`) | no |
| Predictive Edge Map | `GET /optimization/predictive-edge-runs/{run_id}/near-misses/{near_miss_id}` | One near-miss detail | no |
| Predictive Edge Map | `GET /optimization/predictive-edge-runs/{run_id}/targeted-refinement` | Targeted stage summary | no |
| Predictive Edge Map | `GET /optimization/predictive-edge-runs/{run_id}/targeted-refinement/candidates` | Targeted candidate evidence | no |
| Predictive Edge Map | `GET /optimization/predictive-edge-runs/{run_id}/targeted-refinement/candidates/{candidate_id}` | Targeted candidate detail | no |
| Predictive Edge Map | `GET /optimization/predictive-edge-runs/{run_id}/composite-search` | Composite (multi-feature v4.1) summary | no |
| Predictive Edge Map | `GET /optimization/predictive-edge-runs/{run_id}/composite-search/candidates` | Composite candidate page | no |
| Predictive Edge Map | `GET /optimization/predictive-edge-runs/{run_id}/composite-search/candidates/{candidate_id}` | Composite candidate detail | no |
| Edge-Guided Candidates | `GET /optimization/edge-guided/runs` | Qualification runs | no |
| Edge-Guided Candidates | `GET /optimization/edge-guided/runs/{run_id}` | One qualification run | no |
| Edge-Guided Candidates | `GET /optimization/edge-guided/candidates` | Candidate evaluations | no |
| Edge-Guided Candidates | `GET /optimization/edge-guided/candidates/{candidate_evaluation_id}` | Candidate plus window evidence | no |
| ML Screening | `GET /optimization/ml-screening-runs` | Screening runs | no |
| ML Screening | `GET /optimization/ml-screening-runs/latest` | Latest completed run for an instrument | no |
| ML Screening | `GET /optimization/ml-screening-runs/{run_id}` | Run detail | no |
| ML Screening | `GET /optimization/ml-screening-runs/{run_id}/results` | Results page | no |
| ML Screening | `GET /optimization/ml-screening-runs/{run_id}/results/{result_id}` | One result detail | no |
| ML Refinement | `GET /optimization/ml-refinement-contract` | Request/response contract | no |
| ML Refinement | `POST /optimization/ml-refinement-runs` | Enqueue from a completed screening result | yes |
| ML Refinement | `POST /optimization/ml-refinement-runs/{run_id}/cancel` | Cancel queued/running run | yes |
| ML Refinement | `POST /optimization/ml-refinement-runs/{run_id}/retry` | Retry failed/cancelled run | yes |
| ML Refinement | `GET /optimization/ml-refinement-runs` | List runs | no |
| ML Refinement | `GET /optimization/ml-refinement-runs/{run_id}` | Run detail | no |
| ML Refinement | `GET /optimization/ml-refinement-runs/{run_id}/evaluations` | Evaluations page | no |
| ML Refinement | `GET /optimization/ml-refinement-runs/{run_id}/evaluations/{evaluation_id}` | One evaluation detail | no |
| ML Search Campaigns | `GET /optimization/ml-search-campaign-contract` | v1 (`exhaustive_ohlcv_v1`) contract | no |
| ML Search Campaigns | `GET /optimization/ml-search-campaign-contract/v2` | v2 (`economic_policy_search_v2`) contracts | no |
| ML Search Campaigns | `POST /optimization/ml-search-campaigns` | Enqueue or reuse a campaign | yes |
| ML Search Campaigns | `GET /optimization/ml-search-campaigns` | Page campaigns | no |
| ML Search Campaigns | `GET /optimization/ml-search-campaigns/{campaign_id}` | Campaign with family/qualification counts | no |
| ML Search Campaigns | `GET /optimization/ml-search-campaigns/{campaign_id}/families` | Page families | no |
| ML Search Campaigns | `GET /optimization/ml-search-campaigns/{campaign_id}/families/{family_id}` | One family | no |
| ML Search Campaigns | `GET /optimization/ml-search-campaigns/{campaign_id}/qualifications` | Page economic qualifications | no |
| ML Search Campaigns | `POST /optimization/ml-search-campaigns/{campaign_id}/diagnostic-qualifications` | Enqueue diagnostic qualifications | yes |
| ML Search Campaigns | `GET /optimization/ml-search-campaigns/{campaign_id}/qualifications/{qualification_id}` | One qualification | no |
| ML Search Campaigns | `POST /optimization/ml-search-campaigns/{campaign_id}/cancel` | Cancel running campaign | yes |
| ML Search Campaigns | `POST /optimization/ml-search-campaigns/{campaign_id}/retry` | Retry failed/cancelled campaign | yes |
| Protected Final Validation | `GET /optimization/ml-search-campaigns/{campaign_id}/families/{family_id}/audit` | Development re-audit report for a family | no |
| Protected Final Validation | `POST /optimization/ml-search-campaigns/{campaign_id}/families/{family_id}/development-manifest-reconstructions` | Enqueue manifest reconstruction | yes |
| Protected Final Validation | `GET /optimization/ml-development-manifest-reconstructions/{manifest_id}` | One reconstruction record | no |
| Protected Final Validation | `POST /optimization/ml-development-manifest-reconstructions/{manifest_id}/cancel` | Cancel queued/running reconstruction | yes |
| Protected Final Validation | `POST /optimization/ml-development-manifest-reconstructions/{manifest_id}/retry` | Retry failed reconstruction | yes |
| Protected Final Validation | `POST /optimization/ml-search-campaigns/{campaign_id}/families/{family_id}/final-validations` | Enqueue protected final validation | yes |
| Protected Final Validation | `GET /optimization/ml-final-validations` | Page validations | no |
| Protected Final Validation | `GET /optimization/ml-final-validations/{validation_id}` | One validation record | no |
| Protected Final Validation | `POST /optimization/ml-final-validations/{validation_id}/retry` | Retry failed/cancelled validation | yes |
| Protected Final Validation | `POST /optimization/ml-final-validations/{validation_id}/cancel` | Cancel queued/running validation | yes |

Edge-guided candidate and run responses add complete-Strategy family, entry/exit,
cadence, holding, cooldown, allocation, generation-contract, ranking, funnel, timing,
and outcome fields. They remain additive and nullable for legacy rows.
`test_metrics` is null when validation did not authorize isolated test; it is never a
synthetic zero.

## CORS and `argos-web`

`CORS_ALLOWED_ORIGINS` is a comma-separated list of exact HTTP/HTTPS origins. Wildcards, credentials, paths, queries, fragments, and empty entries are rejected. Allowed methods are GET, POST, and OPTIONS; credentials are enabled for the HttpOnly session cookie. Allowed request headers are Accept, Content-Type, and X-CSRF-Token. The only exposed application response header is X-CSRF-Token, allowing the separately hosted SPA to recover the in-memory CSRF proof from login and `/auth/me` without reading API-domain cookies. The Railway hosts are cross-site because `up.railway.app` is a verified [Public Suffix List](https://publicsuffix.org/list/public_suffix_list.dat) entry, so production host-only API cookies use `Secure; SameSite=None`; local HTTP cookies use `SameSite=Lax` without Secure. Explicit cookie/header/JWT-digest binding continues to protect unsafe cross-site requests. Production must list the public `argos-web` origin, not a temporary or private Railway hostname. See [API authentication](015-authentication.md).

The frontend consumes the aggregate dashboard, instrument catalog, Paper portfolio/performance/history, Scores, Decisions, and optimization candidate APIs. Its implementation lives in `EduBeghe/argos-web`. It is data-driven by complete catalog identities, does not hardcode BTC/ETH behavior, keeps visible copy in English, uses accessible low-brightness presentation and metric help, and gives charts axes, units, UTC references, gaps for absent evidence, and textual summaries. Authoritative financial calculations remain in the backend.

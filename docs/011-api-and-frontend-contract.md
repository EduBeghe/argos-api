# API and Frontend Contract

The FastAPI application is currently a read-only HTTP surface. It does not contact Binance or run ingestion, decisions, simulations, or refinement from GET requests. Parameters use UTC timestamps and complete instrument identity where ambiguity is possible. Decimal financial values remain exact serialized values.

## Endpoint inventory

| Group | Method and path | Main parameters and purpose | Result | Mutation |
| --- | --- | --- | --- | --- |
| Health | `GET /health` | Process liveness | service status/version | no |
| Health | `GET /health/database` | PostgreSQL connectivity | status or 503 | no |
| Market | `GET /market/instruments` | none; catalog | configured identities/capabilities/display metadata | no |
| Market | `GET /market/candles` | instrument, UTC range, order, limit | persisted candles | no |
| Market | `GET /market/status` | none | Binance Spot 5m count/latest/freshness | no |
| Market | `GET /market/features/latest` | instrument | latest complete Feature snapshot | no |
| Market | `GET /market/scores/latest` | instrument | latest Score | no |
| Market | `GET /market/scores` | instrument, UTC range, order, limit | Score history | no |
| Market | `GET /market/decisions/latest` | instrument | latest Decision Snapshot | no |
| Market | `GET /market/decisions` | instrument, run/range/order/limit | Decision history | no |
| Dashboard | `GET /dashboard/overview` | none | one composed catalog/market/Paper overview | no |
| Paper | `GET /paper/account` | active run | account/run summary | no |
| Paper | `GET /paper/portfolio` | optional `run_id` | current persisted-candle valuation | no |
| Paper | `GET /paper/performance` | optional `run_id` | equity-history performance | no |
| Paper | `GET /paper/evaluation` | optional `run_id` | Strategy and buy-and-hold comparison | no |
| Paper | `GET /paper/equity-snapshots` | run, limit/offset/order | immutable snapshot history | no |
| Paper | `GET /paper/equity-snapshots/{snapshot_id}` | snapshot ID | one snapshot and positions | no |
| Paper | `GET /paper/runs` | limit/offset | active and stopped runs | no |
| Paper | `GET /paper/runs/{run_id}` | run ID | historical run | no |
| Paper | `GET /paper/positions` | run/status/limit/offset | positions, including open-at-stop history | no |
| Paper | `GET /paper/trades` | run, optional instrument/range/order/page | immutable simulated trades | no |
| Paper | `GET /paper/decision-cycles` | run/asset/page | auditable cycle history | no |
| Paper | `GET /paper/decision-cycles/{cycle_id}` | cycle ID | one cycle | no |
| Optimization | `GET /optimization/refinement-runs` | instrument/status/recommendation/range/page filters | refinement run summaries | no |
| Optimization | `GET /optimization/refinement-runs/{run_id}` | run ID | bounded run detail | no |
| Optimization | `GET /optimization/refinement-runs/{run_id}/evaluations` | limit/cursor | paged evaluations | no |
| Optimization | `GET /optimization/candidates` | instrument/classification/model/evidence filters, cursor | canonical candidate profiles | no |
| Optimization | `GET /optimization/candidates/{stable_candidate_id}` | ID, limit/cursor | definition and paged evidence | no |
| Optimization | `GET /optimization/refinement-runs/{run_id}/walk-forward` | refinement run ID | linked walk-forward summary | no |
| Optimization | `GET /optimization/walk-forward-runs/{run_id}` | run ID | walk-forward detail | no |
| Optimization | `GET /optimization/walk-forward-runs/{run_id}/candidates` | classification/limit/cursor | candidate page | no |
| Optimization | `GET /optimization/walk-forward-runs/{run_id}/selections` | selected flag/page | selection page | no |
| Optimization | `GET /optimization/walk-forward-runs/{run_id}/candidates/{candidate_id}/windows` | page | per-window evidence | no |
| Optimization | `GET /optimization/predictive-edge-runs` | instrument/status/cursor | analysis runs | no |
| Optimization | `GET /optimization/predictive-edge-runs/{run_id}` | run ID | run detail | no |
| Optimization | `GET /optimization/predictive-edge-runs/{run_id}/edges` | predictor/horizon/condition/classification/page | edge evidence | no |
| Optimization | `GET /optimization/predictive-edge-runs/{run_id}/edges/{edge_id}/windows` | page | edge window evidence | no |
| Optimization | `GET /optimization/predictive-edge-runs/{run_id}/hypotheses` | page | Strategy hypotheses | no |
| Optimization | `GET /optimization/predictive-edge-runs/{run_id}/targeted-refinement` | run ID | targeted stage summary | no |
| Optimization | `GET /optimization/predictive-edge-runs/{run_id}/targeted-refinement/candidates` | status/page | targeted candidate evidence | no |
| Optimization | `GET /optimization/predictive-edge-runs/{run_id}/targeted-refinement/candidates/{candidate_id}` | candidate ID | targeted candidate detail | no |
| Optimization | `GET /optimization/edge-guided/runs` | instrument/status/page | qualification runs | no |
| Optimization | `GET /optimization/edge-guided/runs/{run_id}` | run ID | one qualification run | no |
| Optimization | `GET /optimization/edge-guided/candidates` | run/instrument/status/action/predictor/condition/page | candidate evaluations | no |
| Optimization | `GET /optimization/edge-guided/candidates/{candidate_evaluation_id}` | evaluation ID | candidate plus window evidence | no |

FastAPI’s generated `/docs`, `/redoc`, and `/openapi.json` are the schema-level reference. This inventory intentionally summarizes rather than duplicates Pydantic models.

Edge-guided candidate and run responses add complete-Strategy family, entry/exit,
cadence, holding, cooldown, allocation, generation-contract, ranking, funnel, timing,
and outcome fields. They remain additive and nullable for legacy rows.
`test_metrics` is null when validation did not authorize isolated test; it is never a
synthetic zero.

## CORS and `argos-web`

`CORS_ALLOWED_ORIGINS` is a comma-separated list of exact HTTP/HTTPS origins. Wildcards, credentials, paths, queries, fragments, and empty entries are rejected. Allowed methods are GET and OPTIONS; credentials are disabled. Production should list the public `argos-web` origin, not a temporary or private Railway hostname.

The frontend consumes the aggregate dashboard, instrument catalog, Paper portfolio/performance/history, Scores, Decisions, and optimization candidate APIs. Its implementation lives in `EduBeghe/argos-web`. It is data-driven by complete catalog identities, does not hardcode BTC/ETH behavior, keeps visible copy in English, uses accessible low-brightness presentation and metric help, and gives charts axes, units, UTC references, gaps for absent evidence, and textual summaries. Authoritative financial calculations remain in the backend.

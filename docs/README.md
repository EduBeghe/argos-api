# Argos Documentation Index

This index separates current system truth from historical decisions and future work. Read the first five documents in order when changing the system.

| Document | Purpose | Status |
| --- | --- | --- |
| [Project vision](000-project-vision.md) | Product purpose, safety, and evidence principles | current |
| [MVP scope](001-mvp-scope.md) | Implemented research loop and explicit exclusions | current |
| [Architecture](002-architecture.md) | Runtime processes, boundaries, transactions, and data flow | current |
| [Current system reference](010-current-system.md) | Instruments, market data, analytics, Strategy, Paper Trading, optimization, and roadmap | current |
| [API and frontend contract](011-api-and-frontend-contract.md) | Endpoint inventory, CORS, and `argos-web` integration | current |
| [Railway operations](012-railway-operations.md) | Deployment, cron, diagnostics, and safe historical-job commands | current |
| [Weekly Strategy research](014-weekly-strategy-research.md) | Complete candidate search, reuse, outcomes, and research runbook | current |
| [API authentication](015-authentication.md) | Session, CSRF, CORS, and staged Railway rollout | current |
| [Glossary](013-glossary.md) | Accessible domain and financial vocabulary | current |
| [Domain model](003-domain-model.md) | Provider-independent vocabulary and relationships | current |
| [Binance market-data contract](004-binance-market-data-contract.md) | Binance positional kline boundary and normalization | current |
| [Analytical Features](005-analytical-features.md) | Exact version-1 Feature definitions | current |
| [Baseline Score](006-baseline-score-model.md) | Exact `baseline_technical` version-1 model | current |
| [Paper Strategy evaluation](007-paper-strategy-evaluation.md) | Read-only run evaluation and benchmark evidence | current |
| [Experiment memory](008-experiment-memory.md) | Refinement persistence, reuse, and experience-guided planning | current |
| [Predictive Edge Map](009-predictive-edge-map.md) | Walk-forward predictive evidence and hypotheses | current |
| [ML experiment dataset contract](ml-experiment-contract.md) | Leak-free variables, targets, temporal folds, and read-only validation | current |
| [ML model screening](ml-model-screening.md) | Reproducible Ridge, LightGBM, and CatBoost development-fold screening | current |

## Architecture decisions

ADRs are historical decisions: they explain why a choice was made and are not a substitute for the current reference. ADR-001 through ADR-011 remain accepted. Where implementation has resolved an earlier â€œopen decisionâ€ sentence, the implementation and current reference are authoritative; the original decision record remains useful history. No ADR is currently marked superseded.

## Future work

Future work is listed only in [Current system reference](010-current-system.md#current-state-and-next-steps). It is not a production commitment.

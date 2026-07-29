# API Authentication and Railway Rollout

## Scope and threat model

Argos uses one configured administrator, not a user-management system. A successful
login creates an HMAC-signed, eight-hour session JWT in the HttpOnly `argos_session`
cookie and an independently readable `argos_csrf` cookie. The design protects private
API data from unauthenticated access, password disclosure, token tampering, and
cross-site unsafe requests. It does not provide refresh tokens, per-device revocation,
multiple users, OAuth, or protection after an attacker steals a valid JWT.

Logout removes both browser cookies. Because sessions are stateless, a stolen JWT remains
valid until expiration. Rotating `ARGOS_JWT_SECRET` is the emergency global-revocation
mechanism and invalidates every existing session.

## Public and private routes

With `ARGOS_AUTH_ENFORCEMENT_ENABLED=true`, the exact unauthenticated allowlist is:

- `GET /health`;
- `POST /auth/login`;
- CORS preflight `OPTIONS` requests.

Every other current or future route is private by default, including
`GET /health/database`. Public health returns only `{"status":"ok"}`. FastAPI docs,
ReDoc, and OpenAPI routes are not registered while enforcement is enabled.

When enforcement is disabled, existing endpoints and developer documentation remain
accessible. Authentication routes work only when valid credentials are configured. The
API logs one startup warning. This mode exists only to avoid frontend downtime during
the staged rollout and must be removed after login is verified.

## Environment variables

| Variable | Contract |
| --- | --- |
| `ARGOS_AUTH_ENFORCEMENT_ENABLED` | Temporary rollout switch; defaults to `false`. |
| `ARGOS_ADMIN_USERNAME` | Administrator username, 1-128 characters. |
| `ARGOS_ADMIN_PASSWORD_HASH` | Argon2id hash generated interactively; never plaintext. |
| `ARGOS_JWT_SECRET` | Random secret with at least 32 effective UTF-8 bytes. |
| `ARGOS_SESSION_LIFETIME_MINUTES` | Session lifetime, 15-1440 minutes; defaults to 480. |
| `ARGOS_AUTH_COOKIE_SECURE` | `true` in Railway/production; `false` only for local HTTP. |
| `CORS_ALLOWED_ORIGINS` | Comma-separated exact public frontend origins; no wildcard. |

Generate a password hash without placing the password in shell history:

```sh
python -m app.jobs.hash_admin_password
```

The command reads and confirms the password with `getpass` and prints only its Argon2id
hash. API authentication settings are separate from shared job settings, so ingestion,
decision, research, migrations, and Alembic do not require these secrets.

## Browser contract

`POST /auth/login` accepts exactly `username` and `password`, returns `204`, and sets:

- `argos_session`: HttpOnly, `SameSite=Strict`, `Path=/`, Secure in production;
- `argos_csrf`: readable by the SPA, `SameSite=Strict`, `Path=/`, Secure in production.

The JWT itself is never returned in JSON. `GET /auth/me` returns only `username`, `role`,
and the UTC expiration timestamp. For authenticated `POST`, `PUT`, `PATCH`, and `DELETE`,
the SPA must send the `argos_csrf` cookie value in `X-CSRF-Token`. `POST /auth/logout`
requires that proof and deletes both cookies. Browser requests must use credentials.

## Railway rollout and rollback

1. Deploy this backend with `ARGOS_AUTH_ENFORCEMENT_ENABLED=false`.
2. Deploy the separate frontend login change.
3. Configure all authentication variables in the `argos-api` Railway service. Set
   `ARGOS_AUTH_COOKIE_SECURE=true` and use only the public web origin for CORS.
4. Set `ARGOS_AUTH_ENFORCEMENT_ENABLED=true`.
5. Redeploy `argos-api`.
6. Verify `GET /health` remains public and exact, then verify unauthenticated requests to
   `/health/database`, `/market/instruments`, `/paper/runs`, `/dashboard/overview`, and
   `/optimization/refinement-runs` return `401`.

If the frontend rollout fails, temporarily set enforcement back to `false` and redeploy;
this restores compatibility but also restores public data access. Correct the frontend
or configuration promptly, re-enable enforcement, and repeat verification. Rotate the
JWT secret and redeploy if session compromise is suspected.

No database migration is involved in authentication rollout or rollback.

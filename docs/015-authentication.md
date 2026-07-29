# API Authentication and Railway Rollout

## Scope and threat model

Argos uses one configured administrator, not a user-management system. A successful
login creates an HMAC-signed, eight-hour session JWT in the HttpOnly `argos_session`
cookie and a CSRF cookie scoped to the API origin. The design protects private
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

- production (`ARGOS_AUTH_COOKIE_SECURE=true`): both cookies are host-only, `Secure`,
  `SameSite=None`, and `Path=/`; only `argos_session` is HttpOnly;
- local HTTP (`ARGOS_AUTH_COOKIE_SECURE=false`): both cookies are host-only,
  `SameSite=Lax`, not Secure, and `Path=/`; only `argos_session` is HttpOnly.

The Railway deployment hosts `argos-web-production-e6cf.up.railway.app` and
`argos-api-production-b48a.up.railway.app` are cross-site: `up.railway.app` is listed
in the [Public Suffix List](https://publicsuffix.org/list/public_suffix_list.dat).
Production therefore requires `Secure; SameSite=None`; browsers reject insecure
`SameSite=None`, so local HTTP uses `SameSite=Lax`. No `Domain` attribute is set, and
both cookies remain host-only API cookies rather than shared Railway cookies.

The JWT itself is never returned in JSON or a readable response header. Because
`argos-web` and `argos-api` have different origins, frontend JavaScript cannot and must
not attempt to read the API-domain CSRF cookie through `document.cookie`. Successful
login returns the same raw CSRF value in the exposed `X-CSRF-Token` response header.
`GET /auth/me` returns only `username`, `role`, and the UTC expiration timestamp in JSON,
and restores the same header after validating the submitted CSRF cookie against the JWT.
Both responses use `Cache-Control: no-store`.

`SameSite=None` is transport compatibility, not CSRF protection. Unsafe requests still
require the session cookie, CSRF cookie, matching `X-CSRF-Token`, and the CSRF digest
bound into the signed JWT. Exact-origin credentialed CORS limits which configured
frontend origin may read the response. Browser policies that globally block third-party
cookies may still prevent this Railway-domain arrangement.

The SPA calls every API request with browser credentials enabled, keeps the response
header value in memory only, and sends it as `X-CSRF-Token` on authenticated `POST`,
`PUT`, `PATCH`, and `DELETE`. It never stores the CSRF value or JWT in local storage,
session storage, IndexedDB, URLs, or logs. After a page reload it restores identity and
the in-memory CSRF value through `/auth/me`; a `401` returns the user to login.
`POST /auth/logout` requires the original cookie/header/JWT proof, deletes both cookies,
returns no CSRF header, and uses `Cache-Control: no-store`.

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

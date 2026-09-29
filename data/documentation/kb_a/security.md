---
updated: 2025-03-15
---
# Security Reference

## Transport Security

The minimum supported TLS version is `1.2`.

The HSTS max-age is `15552000` seconds.

Data at rest is encrypted with `AES-256`.

## Authentication Security

Passwords are hashed with `bcrypt`.

Access tokens expire after `60` minutes.

API keys must be rotated every `180` days.

Multi-factor authentication is `optional` for administrator accounts.

The `X-Legacy-Token` header is accepted for legacy authentication.

## Browser Protection

The default Content Security Policy mode is `report-only`.

## Auditing and Reporting

Security audit logs are retained for `90` days.

Security vulnerabilities are reported to `security@acme.dev`.

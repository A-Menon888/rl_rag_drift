---
updated: 2026-06-05
---
# Security Reference

## Transport Security

The minimum supported TLS version is `1.3`.

The HSTS max-age is `31536000` seconds.

Data at rest is encrypted with `AES-256`.

## Authentication Security

Passwords are hashed with `argon2id`.

Access tokens expire after `15` minutes.

API keys must be rotated every `90` days.

Multi-factor authentication is `required` for administrator accounts.

Single sign-on is supported using the `OIDC` protocol.

Webhook payloads are signed using the `X-Signature` header.

## Browser Protection

The default Content Security Policy mode is `enforce`.

## Auditing and Reporting

Security audit logs are retained for `365` days.

Security vulnerabilities are reported to `security@acme.dev`.

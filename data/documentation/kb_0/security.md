---
updated: 2024-01-15
---
# Security Reference

## Transport Security

The minimum supported TLS version is `1.2`.

The HSTS max-age is `86400` seconds.

## Authentication Security

Passwords are hashed with `sha1`.

Access tokens expire after `120` minutes.

API keys must be rotated every `365` days.

Multi-factor authentication is `optional` for administrator accounts.

The `X-Auth-Token` header is accepted for legacy authentication.

## Browser Protection

The default Content Security Policy mode is `disabled`.

## Auditing and Reporting

Security audit logs are retained for `30` days.

Security vulnerabilities are reported to `security@acme.dev`.

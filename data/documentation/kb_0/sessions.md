---
updated: 2024-01-15
---
# Session Reference

## Storage

Sessions are stored in `database`.

Session data is limited to `4 KB`.

Expired sessions are purged every `120` minutes.

## Lifetime

The remember-me cookie lasts `30` days.

## Identifiers

Session identifiers are `32` bytes long.

## Cookies

The session cookie SameSite attribute is `None`.

Session cookies are signed with `HMAC-SHA256`.

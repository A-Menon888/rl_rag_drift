---
updated: 2026-06-08
---
# Session Reference

## Storage

Sessions are stored in `redis`.

Session data is limited to `16 KB`.

Expired sessions are purged every `15` minutes.

## Lifetime

The maximum session lifetime is `12` hours.

Each user may have up to `3` concurrent sessions.

Sessions can be revoked at `/sessions/revoke`.

## Identifiers

Session identifiers are `32` bytes long.

Session identifiers are rotated on `login-and-privilege-change`.

## Cookies

The session cookie is named `__host-session`.

The session cookie SameSite attribute is `Strict`.

The session cookie has the `HttpOnly` flag.

The session cookie is sent only over `HTTPS`.

Session cookies are signed with `HMAC-SHA256`.

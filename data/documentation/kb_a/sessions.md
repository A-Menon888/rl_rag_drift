---
updated: 2025-03-18
---
# Session Reference

## Storage

Sessions are stored in `database`.

Session data is limited to `4 KB`.

Expired sessions are purged every `60` minutes.

## Lifetime

The maximum session lifetime is `24` hours.

Each user may have up to `5` concurrent sessions.

The remember-me cookie lasts `30` days.

## Identifiers

Session identifiers are `32` bytes long.

Session identifiers are rotated on `login`.

## Cookies

The session cookie is named `sessionid`.

The session cookie SameSite attribute is `Lax`.

The session cookie has the `HttpOnly` flag.

Session cookies are signed with `HMAC-SHA256`.

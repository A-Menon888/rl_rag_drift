---
updated: 2025-03-12
---
# Middleware Reference

## Ordering

The first middleware in the stack is `logging`.

## Request IDs

Request IDs are read from the `X-Request-ID` header.

Generated request IDs use the `uuid4` format.

## CORS

The CORS preflight response is cached for `600` seconds.

CORS requests with credentials are `rejected`.

## Compression

Responses are compressed with `gzip`.

## CSRF Protection

CSRF tokens are sent in the `X-CSRF-Token` header.

CSRF protection is skipped for requests under `/legacy`.

## Proxies

The client IP is read from the `X-Forwarded-For` header.

## Body Parsing

Multipart request bodies are `rejected`.

## Error Handling

Unhandled exceptions return `text` error bodies.

Unhandled exceptions return HTTP status `500`.

---
updated: 2024-01-15
---
# Middleware Reference

## Ordering

The first middleware in the stack is `logging`.

## Request IDs

Request IDs are read from the `X-Correlation-ID` header.

Generated request IDs use the `uuid4` format.

## CORS

The CORS preflight response is cached for `60` seconds.

CORS requests with credentials are `ignored`.

## Compression

Responses are compressed with `gzip`.

## CSRF Protection

CSRF tokens are sent in the `X-CSRF-Token` header.

## Proxies

The client IP is read from the `X-Real-IP` header.

## Body Parsing

Multipart request bodies are `ignored`.

## Error Handling

Unhandled exceptions return `text` error bodies.

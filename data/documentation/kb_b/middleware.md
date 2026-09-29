---
updated: 2026-06-02
---
# Middleware Reference

## Ordering

The first middleware in the stack is `request-id`.

## Request IDs

Request IDs are read from the `X-Request-ID` header.

Generated request IDs use the `ulid` format.

## CORS

The CORS preflight response is cached for `86400` seconds.

CORS requests with credentials are `allowed`.

## Compression

Responses are compressed with `brotli`.

## CSRF Protection

CSRF tokens are sent in the `X-CSRF-Token` header.

## Proxies

The client IP is read from the `Forwarded` header.

Up to `2` trusted proxy hops are honored.

## Body Parsing

Multipart request bodies are `accepted`.

## Response Headers

Response time is reported in the `Server-Timing` header.

## Error Handling

Unhandled exceptions return `problem-json` error bodies.

Unhandled exceptions return HTTP status `500`.

---
updated: 2025-03-05
---
# API Reference

## Base URL

All API requests use the base URL:

`https://api.acme.dev/v1`

The `/v1` path identifies the current stable API version.

## User API

Create a user with:

`POST /v1/users`

Retrieve a user with:

`GET /v1/users/{id}`

Delete a user with:

`DELETE /v1/users/{id}`

The maximum number of users returned by a single list request is 100.

## Payment API

Create a payment with:

`POST /v1/payments`

Retrieve a payment with:

`GET /v1/payments/{id}`

The payment creation endpoint supports an `Idempotency-Key` header.

Supported payment currencies are `USD`, `EUR`, and `GBP`.

## API Rate Limits

The default API rate limit is 100 requests per minute per API key.

Requests that exceed the rate limit receive HTTP status `429`.

## API Responses

Successful resource creation returns HTTP status `201`.

Successful resource retrieval returns HTTP status `200`.
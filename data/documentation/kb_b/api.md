---
updated: 2026-05-20
---
# API Reference

## Base URL

All API requests use the base URL:

`https://api.acme.dev/v2`

The `/v2` path identifies the current stable API version.

## User API

Create a user with:

`POST /v2/users`

Retrieve a user with:

`GET /v2/users/{id}`

Delete a user with:

`DELETE /v2/users/{id}`

The maximum number of users returned by a single list request is 200.

## Payment API

Create a payment with:

`POST /v2/payments`

Retrieve a payment with:

`GET /v2/payments/{id}`

The payment creation endpoint requires an `Idempotency-Key` header.

Supported payment currencies are `USD`, `EUR`, and `INR`.

## API Rate Limits

The default API rate limit is 60 requests per minute per API key.

Requests that exceed the rate limit receive HTTP status `429`.

## API Responses

Successful resource creation returns HTTP status `201`.

Successful resource retrieval returns HTTP status `200`.

## Webhook API

Create a webhook subscription with:

`POST /v2/webhooks`

Webhook subscriptions can be deleted with:

`DELETE /v2/webhooks/{id}`
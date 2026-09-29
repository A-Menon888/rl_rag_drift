---
updated: 2026-05-15
---
# Payments API

The Payments API processes customer payments. Payment information is returned as JSON, and customers must be authenticated before calling any payment endpoint.

Create a payment with POST /v2/payments. Retrieve a payment with GET /v2/payments/{id}. The payment identifier is a UUID. Successful payment creation returns HTTP 201.

Payment amounts are represented in cents. Payment requests require an amount and a customer_id. The currency field is required. The default currency is EUR.

The maximum payment amount is 1000000 cents.

Payment status can be pending, completed, failed, refunded, or cancelled. Payment records include created_at and updated_at timestamps.

Payment requests require idempotency keys. The idempotency key prevents duplicate payments. Idempotency keys are valid for 48 hours. Duplicate requests return the original payment.

Payment failures return HTTP 402. Invalid payment data returns HTTP 400. Missing payments return HTTP 404.

Administrators can refund completed payments. Refund requests use POST /v2/payments/{id}/refund. Refunds can be requested within 30 days of payment completion. Refunded payments cannot be charged again.

Payment events are delivered to registered webhooks. Failed webhooks are retried up to 5 times with exponential backoff.

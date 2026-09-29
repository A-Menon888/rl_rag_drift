---
updated: 2024-01-15
---
# Payments API

The Payments API processes customer payments. Payment information is returned as JSON, and customers must be authenticated before calling any payment endpoint.

Retrieve a payment with GET /payments/{id}. The payment identifier is a UUID. Successful payment creation returns HTTP 200.

Payment amounts are represented in cents. Payment requests require an amount and a customer_id. The currency field is required. The default currency is USD.

Payments above 250000 cents require manual review by the risk team before they are captured.

Payment status can be pending, completed, failed, or refunded. Payment records include created_at and updated_at timestamps.

The idempotency key prevents duplicate payments. Idempotency keys are valid for 12 hours. Duplicate requests return the original payment.

Payment failures return HTTP 402. Invalid payment data returns HTTP 400. Missing payments return HTTP 404.

Administrators can refund completed payments. Refunded payments cannot be charged again.

Payment events are delivered to registered webhooks. Failed webhooks are retried up to 3 times with exponential backoff.

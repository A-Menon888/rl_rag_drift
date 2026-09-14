# Payments API

The Payments API processes customer payments.
Create a payment with POST /v2/payments.
Retrieve a payment with GET /v2/payments/{id}.
The payment identifier is a UUID.
Payment amounts are represented in cents.
The currency field is required.
The default currency is EUR.
Payment requests require an amount.
Payment requests require a customer_id.
Successful payment creation returns HTTP 201.
Payment status can be pending.
Payment status can be completed.
Payment status can be failed.
Payment status can be cancelled.
Payment requests require idempotency keys.
The idempotency key prevents duplicate payments.
Idempotency keys are valid for 48 hours.
Duplicate requests return the original payment.
Payment records include created_at.
Payment records include updated_at.
Payment failures return HTTP 402.
Invalid payment data returns HTTP 400.
Missing payments return HTTP 404.
Payment information is returned as JSON.
Customers must be authenticated.
Administrators can refund completed payments.
Refund requests use POST /v2/payments/{id}/refund.
Refunded payments cannot be charged again.
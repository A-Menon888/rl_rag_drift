---
updated: 2025-03-20
---
# Frequently Asked Questions

This FAQ collects common support answers. The API reference pages are authoritative; this page is updated less often.

Which currency is used if I leave it out? The default currency is USD for every payment request that omits it.

Can I safely retry a payment request? Yes. Send the same idempotency key again. Idempotency keys are valid for 24 hours, after which a retry creates a new payment.

How many users can I fetch per page? The maximum page size is 100 users; larger values are rejected.

Do I need to authenticate? Yes. All user and payment endpoints require an authenticated client.

What format do responses use? Every endpoint returns JSON.

---
updated: 2024-01-15
---
# Users API

The Users API manages application user accounts. The API returns JSON responses, and passwords are never returned in user responses.

Create a user with POST /users. Retrieve a user with GET /users/{id}. Update a user with PUT /users/{id}. Delete a user with DELETE /users/{id}.

The user identifier is an integer. New users receive a unique identifier. The email field is required when creating a user, and the name field is optional.

Email addresses must be unique. Email addresses are stored in lowercase.

The default account status is active.

A successful creation returns HTTP 201. A successful retrieval returns HTTP 200. Missing users return HTTP 404. Invalid request data returns HTTP 400.

User deletion is permanent. Deleted users cannot access the application. Administrators can delete any user.

User creation requires authentication. User updates require authentication. Regular users can update their own profile.

The API supports pagination for user listings. The default page size is 10 users. The maximum page size is 100 users.

User records contain created_at and updated_at timestamps.

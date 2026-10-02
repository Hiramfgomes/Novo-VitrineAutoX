# Security Specification - Multi-Vendor Vehicle Marketplace

## Data Invariants
1. **User Identity Invariant**: A user profile's `uid` must strictly match the authenticated user's UID.
2. **Ad Ownership Invariant**: Every `vehicle`, `product`, and `service` listing must have an `ownerId` that matches the authenticated user's ID.
3. **Admin Exclusivity**: Only a verified admin (hiramfgomes@gmail.com) can modify `settings`, `events`, and `coupons`.
4. **Subscription Integrity**: Lojista-specific features require an active subscription state in the user profile.
5. **Modification Immutability**: `createdAt` and `ownerId` fields can never be changed after document creation.
6. **Relational Sync**: Notifications and Alerts must be stored within the user's subcollection and be accessible only by that user.

## The "Dirty Dozen" Payloads (Red Team Test Cases)

1. **Identity Spoofing**: Attempt to create a document in `users` with a `uid` different from `request.auth.uid`.
2. **Role Escalation**: Attempt to update own profile to set `role: 'admin'`.
3. **Shadow Update**: Attempt to update a vehicle listing with extra fields not defined in the schema (e.g., `isVerified: true`).
4. **Price Poisoning**: Attempt to set a negative `price` or a price that is not a number.
5. **Orphaned Write**: Attempt to create a vehicle with an `ownerId` that does not exist in the `users` collection.
6. **Admin Bypass**: Attempt to update `settings/floating_banner` as a regular authenticated user.
7. **PII Leak**: Attempt to read the full `users` profile of another user who is NOT a lojista.
8. **Recursive Cost Attack**: Large-scale `list` query attempt without proper filtering by `status == 'active'`.
9. **Status Shortcutting**: Attempt to change a vehicle `status` from `pending` to `active` without being an admin (if that logic is enforced).
10. **Timestamp Fraud**: Attempt to provide a manual `updatedAt` instead of `request.time`.
11. **ID Injection**: Attempt to create a document with a massive 1MB string as the document ID.
12. **Notification Scraping**: Attempt to list notifications from another user's subcollection.

## Required Rule Helpers

- `isSignedIn()`: Basic auth check.
- `isOwner(userId)`: Verification against `request.auth.uid`.
- `isAdmin()`: Email verification and role check.
- `isValidId(id)`: Regex and size check for path variables.
- `incoming()`: Shortcut for `request.resource.data`.
- `existing()`: Shortcut for `resource.data`.

# Deferred Android work

This file parks verified follow-up work that is not required for the current behavior.

## Localization copy review

The 13 locale catalogs are functionally complete and automatically checked for current
source coverage, placeholders, and plurals. Before a language-specific marketing push,
schedule native-speaker review of long legal, license, FAQ, and measurement-guidance
copy. That review is editorial QA; it is not a missing-resource or runtime fallback.

## Retained legacy error resources

The following default resources are currently unreferenced after UI errors were changed
to localized, non-technical messages. They remain translated so older branches can be
merged without reintroducing English fallback. Remove them together in a later resource
cleanup after confirming no external feature branch consumes them:

- `camera_error_opening`
- `glb_room_error`
- `home_failed_delete`
- `model_detail_failed_load`
- `model_detail_failed_save_screenshot_message`
- `photo_room_back`
- `photo_room_error_load`
- `smartypants_camera_error`
- `smartypants_failed_save`
- `smartypants_screenshot_failed`

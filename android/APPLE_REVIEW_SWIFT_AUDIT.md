# Apple Review Audit For iPhone App

Date: 2026-04-01
Project reviewed: `../Furnit`
Scope: Swift iPhone app review for Apple submission readiness and App Review risk

## Summary

This review focuses on Apple App Review blockers, privacy/compliance mismatches, and release-readiness issues that can affect acceptance or make the app risky to ship.

## Findings

### 1. High: Runtime-downloaded JavaScript in `WKWebView`

The app loads JavaScript modules from public CDNs inside `WKWebView`. This is a real App Review risk under Guideline `2.5.2`, which requires apps to be self-contained and restricts downloaded executable code.

Files:
- [SharpRoomView.swift](/Users/al/Documents/tries01/Furnit/Furnit/Views/SharpRoomView.swift#L1210)
- [SharpRoomView.swift](/Users/al/Documents/tries01/Furnit/Furnit/Views/SharpRoomView.swift#L1307)
- [GLBRoomView.swift](/Users/al/Documents/tries01/Furnit/Furnit/Views/GLBRoomView.swift#L501)
- [GLBRoomView.swift](/Users/al/Documents/tries01/Furnit/Furnit/Views/GLBRoomView.swift#L649)
- [MeshRoomView.swift](/Users/al/Documents/tries01/Furnit/Furnit/Views/MeshRoomView.swift#L444)
- [MeshRoomView.swift](/Users/al/Documents/tries01/Furnit/Furnit/Views/MeshRoomView.swift#L599)

Examples found:
- `cdnjs.cloudflare.com`
- `cdn.jsdelivr.net`
- `sparkjs.dev`

Why this matters:
- Apple may treat CDN-loaded JS modules as downloaded code.
- Review can reject this even if the app otherwise works.

Recommended action:
- Bundle `three.js`, Spark, and related JS assets inside the app.
- Load only local packaged resources in all `WKWebView` viewers.

### 2. High: Mandatory account login without in-app account deletion

The app forces login before the user can access the app. The login flow collects phone number and name, and I did not find any in-app account deletion flow.

Files:
- [FurnitApp.swift](/Users/al/Documents/tries01/Furnit/Furnit/FurnitApp.swift#L195)
- [LoginView.swift](/Users/al/Documents/tries01/Furnit/Furnit/Authentication/LoginView.swift#L151)
- [LoginView.swift](/Users/al/Documents/tries01/Furnit/Furnit/Authentication/LoginView.swift#L328)
- [SettingsView.swift](/Users/al/Documents/tries01/Furnit/Furnit/Views/SettingsView.swift#L335)
- [AuthenticationManager.swift](/Users/al/Documents/tries01/Furnit/Furnit/Authentication/AuthenticationManager.swift#L244)

Why this matters:
- Apple is strict about requiring login only when necessary for core functionality.
- If the app creates an account, Apple expects users to be able to delete that account in-app.

Recommended action:
- Either allow core features without login, or justify login as essential.
- Add in-app account deletion.
- Make the reviewer path easy to test.
- Reconsider whether `name` is actually required.

### 3. High: Privacy declarations do not match app behavior

The privacy manifest declares no collected data, but the app clearly handles personal data and auth data.

Files:
- [PrivacyInfo.xcprivacy](/Users/al/Documents/tries01/Furnit/Furnit/PrivacyInfo.xcprivacy#L5)
- [AuthenticationManager.swift](/Users/al/Documents/tries01/Furnit/Furnit/Authentication/AuthenticationManager.swift#L17)
- [AuthenticationManager.swift](/Users/al/Documents/tries01/Furnit/Furnit/Authentication/AuthenticationManager.swift#L81)
- [AuthenticationManager.swift](/Users/al/Documents/tries01/Furnit/Furnit/Authentication/AuthenticationManager.swift#L244)

Observed behavior:
- Stores `userName`, `userPhone`, `userId`
- Uses Firebase phone authentication
- Requires phone-based OTP login

Why this matters:
- Apple checks whether app privacy disclosures match actual app behavior.
- Mismatch can trigger rejection or follow-up questions during review.

Recommended action:
- Update App Store Connect privacy answers.
- Ensure privacy policy matches actual collection, storage, retention, and sharing behavior.
- Review privacy manifest coverage and SDK disclosures for Firebase usage.

### 4. Medium: Hidden feature gating by specific phone numbers

Share functionality is only enabled for a hardcoded set of phone numbers.

Files:
- [AuthenticationManager.swift](/Users/al/Documents/tries01/Furnit/Furnit/Authentication/AuthenticationManager.swift#L390)
- [SharpRoomView.swift](/Users/al/Documents/tries01/Furnit/Furnit/Views/SharpRoomView.swift#L212)

Why this matters:
- This looks like restricted or hidden functionality.
- Reviewers may see different behavior from internal users and treat it as incomplete or misleading.

Recommended action:
- Remove the allowlist before submission, or
- Clearly disclose it in-product and in App Review notes.

### 5. Medium: Build and test configuration mismatch

The app target and test target do not use the same deployment target.

Files:
- [project.pbxproj](/Users/al/Documents/tries01/Furnit/Furnit.xcodeproj/project.pbxproj#L420)
- [project.pbxproj](/Users/al/Documents/tries01/Furnit/Furnit.xcodeproj/project.pbxproj#L497)

Observed:
- App target deployment target: `26.0`
- Test target deployment target: `18.0`

Why this matters:
- Not directly an App Review rejection item.
- It weakens release confidence because tests do not align with the shipping target.

Recommended action:
- Align deployment targets.
- Run clean build, device test, and release validation using the same target matrix.

## Additional Notes

### Sign in with Apple

I did not flag missing Sign in with Apple as a blocker at this stage.

Reason:
- I found first-party phone authentication, not third-party social login.
- Apple Guideline `4.8` generally applies when third-party login services are offered.

Still worth checking before submission if auth scope changes.

### Legal and commercial clarity

Your own strings and license text describe this as a non-commercial Phase 1 release and mention separate commercial licensing for YOLO-related use.

Files:
- [Localizable.strings](/Users/al/Documents/tries01/Furnit/Furnit/en.lproj/Localizable.strings#L154)
- [ContentView.swift](/Users/al/Documents/tries01/Furnit/Furnit/Views/ContentView.swift#L934)

Why this matters:
- If the app is intended for public App Store distribution, licensing and commercial-use wording should be internally consistent and legally settled.

## Submission Checklist

- Bundle all JS and WebGL dependencies locally.
- Remove runtime CDN dependency from all `WKWebView`-based viewers.
- Decide whether login is truly required for core use.
- Add in-app account deletion if accounts remain mandatory.
- Make reviewer access easy and explicit.
- Align app privacy answers with actual data collection.
- Verify privacy policy, terms, and support URLs are live and complete.
- Remove or disclose hidden phone-number-based feature restrictions.
- Align test and app deployment targets.
- Run release build, device tests, and reviewer-path QA.
- Add reviewer notes for anything non-obvious.

## Apple Sources

- App Review Guidelines: https://developer.apple.com/app-store/review/guidelines/
- App privacy metadata reference: https://developer.apple.com/help/app-store-connect/reference/app-information/app-privacy
- Account deletion requirement note: https://developer.apple.com/news/?id=i71db0mv

## Suggested Fix Order

1. Replace all CDN-loaded JavaScript with bundled local assets.
2. Add account deletion or remove mandatory login for non-essential flows.
3. Correct privacy disclosures and privacy policy.
4. Remove hidden share allowlist or document it clearly.
5. Clean up build/test target mismatch and run submission QA.

import XCTest
@testable import Furnit

final class AuthenticationDebugSettingsTests: XCTestCase {

    func testNumberAllowsDebugSettingsAcrossDisplayFormatting() {
        XCTAssertTrue(AuthenticationManager.isDebugSettingsPhoneNumber("+16505553434"))
        XCTAssertTrue(AuthenticationManager.isDebugSettingsPhoneNumber("+1 650-555-3434"))
    }

    func testOtherNumbersDoNotAllowDebugSettings() {
        XCTAssertFalse(AuthenticationManager.isDebugSettingsPhoneNumber(nil))
        XCTAssertFalse(AuthenticationManager.isDebugSettingsPhoneNumber(""))
        XCTAssertFalse(AuthenticationManager.isDebugSettingsPhoneNumber("+16505553435"))
        XCTAssertFalse(AuthenticationManager.isDebugSettingsPhoneNumber("+916505553434"))
        XCTAssertFalse(AuthenticationManager.isDebugSettingsPhoneNumber("(650) 555-3434"))
    }
}

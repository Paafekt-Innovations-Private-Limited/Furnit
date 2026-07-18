import XCTest
@testable import Furnit

final class SplatRoomViewTests: XCTestCase {
    private let autoOrbitKey = "roomViewer.oscillation"

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: autoOrbitKey)
        super.tearDown()
    }

    func testAutoOrbitDefaultIsFalse() {
        UserDefaults.standard.removeObject(forKey: autoOrbitKey)

        XCTAssertFalse(UserDefaults.standard.bool(forKey: autoOrbitKey))
    }

    func testAutoOrbitCanBeEnabled() {
        UserDefaults.standard.set(true, forKey: autoOrbitKey)

        XCTAssertTrue(UserDefaults.standard.bool(forKey: autoOrbitKey))
    }

    func testAutoOrbitCanBeDisabled() {
        UserDefaults.standard.set(true, forKey: autoOrbitKey)
        UserDefaults.standard.set(false, forKey: autoOrbitKey)

        XCTAssertFalse(UserDefaults.standard.bool(forKey: autoOrbitKey))
    }

    func testPhotoOrientationRawValues() {
        XCTAssertEqual(PhotoOrientation.portrait.rawValue, "portrait")
        XCTAssertEqual(PhotoOrientation.landscape.rawValue, "landscape")
    }

    func testPhotoOrientationHints() {
        XCTAssertTrue(PhotoOrientation.portrait.hint.contains("Portrait"))
        XCTAssertTrue(PhotoOrientation.landscape.hint.contains("Landscape"))
    }
}

package com.furnit.android.models.roomintelligence

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AestheticAdvisorTest {
    @Test
    fun `empty palette reports unavailable instead of fabricated scores`() {
        val result = AestheticAdvisor.evaluate(
            palette = SurfacePalette.EMPTY,
            roomStyleTags = emptyList(),
            furniture = profile(StraightSrgbColor(0.5f, 0.4f, 0.3f)),
        )

        assertTrue(result === AestheticEvaluation.Unavailable)
    }

    @Test
    fun `neutral furniture is classified as neutral and scores stay bounded`() {
        val result = available(
            palette = palette(StraightSrgbColor(0.75f, 0.72f, 0.70f)),
            roomTags = listOf("modern"),
            furniture = FurnitureAestheticProfile(
                primaryColor = StraightSrgbColor(0.45f, 0.45f, 0.45f),
                styleTags = listOf("minimalist"),
            ),
        )

        assertEquals(HarmonyType.NEUTRAL, result.harmonyType)
        assertTrue(result.harmonyScore in 0f..1f)
        assertTrue(result.contrastScore in 0f..1f)
        assertTrue(result.styleCompatibilityScore in 0f..1f)
    }

    @Test
    fun `similar lightness produces low contrast recommendation`() {
        val color = StraightSrgbColor(0.55f, 0.35f, 0.20f)
        val result = available(
            palette = palette(color),
            roomTags = listOf("modern"),
            furniture = FurnitureAestheticProfile(color, listOf("modern")),
        )

        assertTrue(result.contrastScore < 0.15f)
        assertTrue(AestheticRecommendationCode.INCREASE_CONTRAST in result.recommendations)
        assertEquals(1f, result.styleCompatibilityScore, 0.0001f)
    }

    @Test
    fun `incompatible known styles score zero`() {
        val result = available(
            palette = palette(StraightSrgbColor(0.8f, 0.8f, 0.8f)),
            roomTags = listOf("industrial"),
            furniture = FurnitureAestheticProfile(
                StraightSrgbColor(0.2f, 0.3f, 0.4f),
                listOf("traditional"),
            ),
        )

        assertEquals(0f, result.styleCompatibilityScore, 0f)
        assertTrue(AestheticRecommendationCode.STYLE_MISMATCH in result.recommendations)
    }

    private fun profile(color: StraightSrgbColor) =
        FurnitureAestheticProfile(color, listOf("modern"))

    private fun palette(color: StraightSrgbColor) =
        SurfacePalette(walls = SurfaceColors(listOf(color), MaterialHint.PLASTER))

    private fun available(
        palette: SurfacePalette,
        roomTags: List<String>,
        furniture: FurnitureAestheticProfile,
    ): AestheticEvaluation.Available =
        AestheticAdvisor.evaluate(palette, roomTags, furniture) as AestheticEvaluation.Available
}

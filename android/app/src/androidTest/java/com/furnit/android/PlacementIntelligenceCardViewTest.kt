package com.furnit.android

import android.content.Context
import android.view.View
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.furnit.android.theme.PlacementIntelligenceCardState
import com.furnit.android.theme.PlacementIntelligenceCardView
import com.furnit.android.theme.PaafektColors
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class PlacementIntelligenceCardViewTest {
    private val context: Context = ApplicationProvider.getApplicationContext()

    @Test
    fun renderExpandAndClearFollowViewerLifecycle() {
        val card = PlacementIntelligenceCardView(context)
        assertEquals(View.GONE, card.visibility)

        card.render(
            PlacementIntelligenceCardState(
                statusText = "Fits by dimensions",
                statusColor = PaafektColors.success,
                dimensionsText = "1.00 × 0.80 × 0.72 m",
                harmonyScore = 80,
                contrastScore = 70,
                styleScore = 90,
                notes = listOf("Depth is estimated."),
            ),
        )
        assertEquals(View.VISIBLE, card.visibility)
        assertFalse(card.isExpanded())

        card.setExpanded(true)
        assertTrue(card.isExpanded())

        card.clear()
        assertEquals(View.GONE, card.visibility)
        assertFalse(card.isExpanded())
    }
}

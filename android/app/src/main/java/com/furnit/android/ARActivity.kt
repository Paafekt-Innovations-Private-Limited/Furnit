package com.furnit.android

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.widget.FrameLayout
import android.widget.TextView

class ARActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Placeholder AR screen: integrates with ARCore or Sceneform in future
        val container = FrameLayout(this)
        val modelId = intent?.getStringExtra("MODEL_ID")
        val display = if (modelId != null) "AR Placeholder — model: $modelId" else "AR Placeholder — integrate ARCore here"
        val tv = TextView(this).apply { text = display }
        container.addView(tv)
        setContentView(container)
    }
}

package com.furnit.android

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Start the Login screen (mirrors iOS entry point)
        startActivity(Intent(this, LoginActivity::class.java))
        finish()
    }
}

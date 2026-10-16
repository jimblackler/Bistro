package com.android.demo.bistro

import android.app.Activity
import android.content.Intent
import android.content.res.AssetManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.widget.TextView

/**
 * Hosts the native Vulkan renderer that flies a camera through the Amazon Lumberyard Bistro.
 *
 * Rendering happens on a dedicated native thread; this activity only forwards the surface lifecycle
 * and shows a small overlay with status information and the asset attribution.
 *
 * Debug extras (useful with `adb shell am start`):
 *  - `startTime` (float): start the fly-through at this many seconds into the loop.
 *  - `freeze` (boolean): hold the camera at `startTime`.
 *  - `camera` (string "x,y,z,tx,ty,tz"): fixed eye and target position.
 *  - `hud` (boolean): show (default) or hide the overlay.
 */
class MainActivity : Activity(), SurfaceHolder.Callback {

    private lateinit var statusText: TextView
    private lateinit var overlay: View
    private val handler = Handler(Looper.getMainLooper())

    private val statusUpdater = object : Runnable {
        override fun run() {
            statusText.text = nativeGetStatus()
            handler.postDelayed(this, STATUS_INTERVAL_MS)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.attributes = window.attributes.apply {
            layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS
        }

        statusText = findViewById(R.id.status_text)
        overlay = findViewById(R.id.overlay)
        val surfaceView = findViewById<SurfaceView>(R.id.surface_view)
        surfaceView.holder.addCallback(this)
        surfaceView.setOnClickListener {
            overlay.visibility = if (overlay.isVisible()) View.GONE else View.VISIBLE
        }

        nativeCreate(assets)
        applyDebugExtras()
    }

    private fun View.isVisible() = visibility == View.VISIBLE

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        applyDebugExtras()
    }

    private fun applyDebugExtras() {
        val extras = intent?.extras ?: return
        val startTime = extras.getFloat("startTime", 0f)
        val freeze = extras.getBoolean("freeze", false)
        val camera = extras.getString("camera")
            ?.split(",")
            ?.mapNotNull { it.trim().toFloatOrNull() }
            ?.takeIf { it.size == 6 }
            ?.toFloatArray()
        overlay.visibility = if (extras.getBoolean("hud", true)) View.VISIBLE else View.GONE
        nativeSetDebugOptions(startTime, freeze, camera)
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) hideSystemBars()
    }

    private fun hideSystemBars() {
        window.insetsController?.let {
            it.hide(WindowInsets.Type.systemBars())
            it.systemBarsBehavior = WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }

    override fun onResume() {
        super.onResume()
        nativeSetPaused(false)
        handler.post(statusUpdater)
    }

    override fun onPause() {
        super.onPause()
        nativeSetPaused(true)
        handler.removeCallbacks(statusUpdater)
    }

    override fun onDestroy() {
        nativeDestroy()
        super.onDestroy()
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        nativeSetSurface(holder.surface)
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        // Size/orientation changes are detected by the swapchain (out-of-date / suboptimal).
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        nativeSetSurface(null)
    }

    private external fun nativeCreate(assetManager: AssetManager)
    private external fun nativeSetSurface(surface: android.view.Surface?)
    private external fun nativeSetPaused(paused: Boolean)
    private external fun nativeDestroy()
    private external fun nativeGetStatus(): String
    private external fun nativeSetDebugOptions(startTime: Float, freeze: Boolean, camera: FloatArray?)

    companion object {
        private const val STATUS_INTERVAL_MS = 500L

        init {
            System.loadLibrary("bistro")
        }
    }
}

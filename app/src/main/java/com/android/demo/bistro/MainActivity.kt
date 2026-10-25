package com.android.demo.bistro

import android.app.Activity
import android.content.Intent
import android.content.res.AssetManager
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Outline
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.TextUtils
import android.util.TypedValue
import android.view.Gravity
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import android.view.ViewOutlineProvider
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import org.json.JSONObject

/**
 * Hosts the native Vulkan renderer that flies a camera through 3D ORCA environments
 * discovered dynamically from `assets/environments/<id>/manifest.json`.
 *
 * Rendering happens on a dedicated native thread; this activity manages the Start screen
 * (with environment selection cards and a 10-second auto-start countdown that alternates
 * between environments after each fly-through completes), forwards the surface lifecycle,
 * and shows an overlay with live rendering statistics and asset attribution.
 *
 * Debug extras (useful with `adb shell am start`):
 *  - `env` (string): environment id or folder name to launch directly.
 *  - `startTime` (float): start the fly-through at this many seconds into the loop.
 *  - `freeze` (boolean): hold the camera at `startTime`.
 *  - `camera` (string "x,y,z,tx,ty,tz"): fixed eye and target position.
 *  - `hud` (boolean): show (default) or hide the overlay.
 *  - `startScreen` (boolean): force showing the Start screen even when debug extras are passed.
 */
class MainActivity : Activity(), SurfaceHolder.Callback {

    private data class EnvironmentEntry(
        val id: String,
        val order: Int,
        val title: String,
        val shortTitle: String,
        val subtitle: String,
        val attribution: String,
        val assetDir: String,
        val previewAssetPath: String
    )

    private lateinit var statusText: TextView
    private lateinit var envTitleText: TextView
    private lateinit var attributionText: TextView
    private lateinit var overlay: View
    private lateinit var startOverlay: View
    private lateinit var cardsContainer: LinearLayout
    private lateinit var countdownText: TextView
    private lateinit var pageSubtitleText: TextView
    private lateinit var prevEnvButton: View
    private lateinit var nextEnvButton: View

    private val handler = Handler(Looper.getMainLooper())
    private val environments = mutableListOf<EnvironmentEntry>()
    private val cardViews = mutableListOf<View>()
    private val cardBadgeViews = mutableListOf<TextView>()

    private var activeEnvIndex = 0
    private var nextDefaultIndex = 0
    private var currentPage = 0
    private var inFlyby = false
    private var hudEnabled = true
    private var countdownRemaining = COUNTDOWN_SECONDS

    private val statusUpdater = object : Runnable {
        override fun run() {
            statusText.text = nativeGetStatus()
            if (inFlyby && nativeIsFlybyCompleted()) {
                if (environments.isNotEmpty()) {
                    nextDefaultIndex = (activeEnvIndex + 1) % environments.size
                }
                showStartScreen()
            }
            handler.postDelayed(this, STATUS_INTERVAL_MS)
        }
    }

    private val countdownTick = object : Runnable {
        override fun run() {
            if (inFlyby || environments.isEmpty()) return
            if (countdownRemaining <= 0) {
                launchEnvironment(nextDefaultIndex)
                return
            }
            updateCountdownUi()
            countdownRemaining--
            handler.postDelayed(this, 1000L)
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
        envTitleText = findViewById(R.id.env_title_text)
        attributionText = findViewById(R.id.attribution_text)
        overlay = findViewById(R.id.overlay)
        startOverlay = findViewById(R.id.start_overlay)
        cardsContainer = findViewById(R.id.environment_cards_container)
        countdownText = findViewById(R.id.countdown_text)
        pageSubtitleText = findViewById(R.id.page_subtitle_text)
        prevEnvButton = findViewById(R.id.prev_env_button)
        nextEnvButton = findViewById(R.id.next_env_button)

        prevEnvButton.setOnClickListener { stepStartScreenPage(-1) }
        nextEnvButton.setOnClickListener { stepStartScreenPage(1) }

        findViewById<View>(R.id.prev_flyby_button).setOnClickListener {
            if (environments.isNotEmpty()) {
                launchEnvironment((activeEnvIndex - 1 + environments.size) % environments.size)
            }
        }
        findViewById<View>(R.id.next_flyby_button).setOnClickListener {
            if (environments.isNotEmpty()) {
                launchEnvironment((activeEnvIndex + 1) % environments.size)
            }
        }

        findViewById<View>(R.id.menu_button).setOnClickListener {
            if (environments.isNotEmpty()) {
                nextDefaultIndex = (activeEnvIndex + 1) % environments.size
            }
            showStartScreen()
        }

        onBackInvokedDispatcher.registerOnBackInvokedCallback(
            android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT
        ) {
            if (inFlyby) {
                if (environments.isNotEmpty()) {
                    nextDefaultIndex = (activeEnvIndex + 1) % environments.size
                }
                showStartScreen()
            } else {
                finish()
            }
        }

        val surfaceView = findViewById<SurfaceView>(R.id.surface_view)
        surfaceView.holder.addCallback(this)
        surfaceView.setOnClickListener {
            if (inFlyby) {
                overlay.visibility = if (overlay.isVisible()) View.GONE else View.VISIBLE
            }
        }

        discoverEnvironments()
        buildEnvironmentCards()

        nativeCreate(assets)
        if (!applyDebugExtras()) {
            showStartScreen()
        }
    }

    private fun View.isVisible() = visibility == View.VISIBLE

    private fun dp(value: Int): Int =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            value.toFloat(),
            resources.displayMetrics
        ).toInt()

    private fun pageCount(): Int =
        ((environments.size + CARDS_PER_PAGE - 1) / CARDS_PER_PAGE).coerceAtLeast(1)

    private fun stepStartScreenPage(delta: Int) {
        if (environments.isEmpty()) return
        val pages = pageCount()
        currentPage = (currentPage + delta + pages) % pages
        nextDefaultIndex = (currentPage * CARDS_PER_PAGE).coerceIn(0, environments.lastIndex)
        handler.removeCallbacks(countdownTick)
        nativeSetEnvironment(environments[nextDefaultIndex].assetDir, false)
        countdownRemaining = COUNTDOWN_SECONDS
        updateVisibleCards()
        updateCountdownUi()
        handler.postDelayed(countdownTick, 1000L)
    }

    private fun discoverEnvironments() {
        environments.clear()
        val dirs = assets.list("environments") ?: emptyArray()
        for (dir in dirs) {
            val assetDir = "environments/$dir"
            val manifestPath = "$assetDir/manifest.json"
            val jsonStr = try {
                assets.open(manifestPath).bufferedReader().use { it.readText() }
            } catch (_: Exception) {
                continue
            }
            val obj = try {
                JSONObject(jsonStr)
            } catch (_: Exception) {
                continue
            }
            environments.add(
                EnvironmentEntry(
                    id = obj.optString("id", dir),
                    order = obj.optInt("order", Int.MAX_VALUE),
                    title = obj.optString("title", dir),
                    shortTitle = obj.optString("shortTitle", obj.optString("title", dir)),
                    subtitle = obj.optString("subtitle", ""),
                    attribution = obj.optString("attribution", getString(R.string.attribution)),
                    assetDir = assetDir,
                    previewAssetPath = "$assetDir/preview.png"
                )
            )
        }
        environments.sortWith(compareBy<EnvironmentEntry> { it.order }.thenBy { it.id })
    }

    private fun buildEnvironmentCards() {
        cardsContainer.removeAllViews()
        cardViews.clear()
        cardBadgeViews.clear()

        for ((index, env) in environments.withIndex()) {
            val card = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                background = getDrawable(R.drawable.card_background)
                setPadding(dp(12), dp(12), dp(12), dp(12))
                isClickable = true
                isFocusable = true
                setOnClickListener { launchEnvironment(index) }
            }
            val slotInPage = index % CARDS_PER_PAGE
            val cardParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f).apply {
                marginStart = if (slotInPage == 0) 0 else dp(8)
                marginEnd = if (slotInPage == CARDS_PER_PAGE - 1 || index == environments.lastIndex) 0 else dp(8)
            }

            val previewContainer = FrameLayout(this).apply {
                clipToOutline = true
                outlineProvider = object : ViewOutlineProvider() {
                    override fun getOutline(view: View, outline: Outline) {
                        outline.setRoundRect(0, 0, view.width, view.height, dp(10).toFloat())
                    }
                }
                setBackgroundColor(Color.parseColor("#1E2732"))
            }
            val previewParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f
            )

            val imageView = ImageView(this).apply {
                scaleType = ImageView.ScaleType.CENTER_CROP
                contentDescription = env.title
                try {
                    assets.open(env.previewAssetPath).use { stream ->
                        setImageBitmap(BitmapFactory.decodeStream(stream))
                    }
                } catch (_: Exception) {
                    // Preview image is optional if not yet captured.
                }
            }
            previewContainer.addView(
                imageView,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
            )

            val badge = TextView(this).apply {
                background = getDrawable(R.drawable.overlay_background)
                setPadding(dp(10), dp(4), dp(10), dp(4))
                setTextColor(Color.parseColor("#81D4FA"))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
                setTypeface(typeface, Typeface.BOLD)
            }
            val badgeParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.END
            ).apply {
                setMargins(dp(8), dp(8), dp(8), dp(8))
            }
            previewContainer.addView(badge, badgeParams)
            card.addView(previewContainer, previewParams)

            val titleView = TextView(this).apply {
                text = env.title
                setTextColor(Color.WHITE)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
                setTypeface(typeface, Typeface.BOLD)
                maxLines = 1
                ellipsize = TextUtils.TruncateAt.END
                setPadding(0, dp(10), 0, 0)
            }
            card.addView(titleView)

            if (env.subtitle.isNotEmpty()) {
                val subtitleView = TextView(this).apply {
                    text = env.subtitle
                    setTextColor(Color.parseColor("#B0BEC5"))
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
                    maxLines = 1
                    ellipsize = TextUtils.TruncateAt.END
                    setPadding(0, dp(2), 0, 0)
                }
                card.addView(subtitleView)
            }

            val attrView = TextView(this).apply {
                text = env.attribution
                setTextColor(Color.parseColor("#78909C"))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 10f)
                maxLines = 1
                ellipsize = TextUtils.TruncateAt.END
                setPadding(0, dp(4), 0, 0)
            }
            card.addView(attrView)

            cardsContainer.addView(card, cardParams)
            cardViews.add(card)
            cardBadgeViews.add(badge)
        }
        updateVisibleCards()
    }

    private fun updateVisibleCards() {
        val pages = pageCount()
        val showChevrons = pages > 1
        prevEnvButton.visibility = if (showChevrons) View.VISIBLE else View.GONE
        nextEnvButton.visibility = if (showChevrons) View.VISIBLE else View.GONE
        for (i in cardViews.indices) {
            cardViews[i].visibility = if (i / CARDS_PER_PAGE == currentPage) View.VISIBLE else View.GONE
        }
        if (pages > 1) {
            val firstIdx = currentPage * CARDS_PER_PAGE
            val lastIdx = (firstIdx + CARDS_PER_PAGE - 1).coerceAtMost(environments.lastIndex)
            pageSubtitleText.text = getString(
                R.string.start_screen_subtitle_paged,
                getString(R.string.start_screen_subtitle), firstIdx + 1, lastIdx + 1, environments.size
            )
        } else {
            pageSubtitleText.text = getString(R.string.start_screen_subtitle)
        }
    }

    private fun showStartScreen() {
        inFlyby = false
        handler.removeCallbacks(countdownTick)
        startOverlay.visibility = View.VISIBLE
        overlay.visibility = View.GONE
        if (environments.isEmpty()) return

        nextDefaultIndex = nextDefaultIndex.coerceIn(0, environments.lastIndex)
        currentPage = nextDefaultIndex / CARDS_PER_PAGE
        updateVisibleCards()
        val targetEnv = environments[nextDefaultIndex]
        // Pre-load the default countdown environment in the background while on the Start screen.
        nativeSetEnvironment(targetEnv.assetDir, false)

        countdownRemaining = COUNTDOWN_SECONDS
        updateCountdownUi()
        handler.postDelayed(countdownTick, 1000L)
    }

    private fun updateCountdownUi() {
        if (environments.isEmpty()) return
        val targetEnv = environments[nextDefaultIndex]
        countdownText.text = getString(R.string.countdown_banner, targetEnv.title, countdownRemaining)
        for (i in environments.indices) {
            val selected = (i == nextDefaultIndex)
            cardViews[i].background = getDrawable(
                if (selected) R.drawable.card_selected_background else R.drawable.card_background
            )
            cardBadgeViews[i].text = if (selected) {
                getString(R.string.badge_auto_start, countdownRemaining)
            } else {
                getString(R.string.badge_tap_to_start)
            }
            cardBadgeViews[i].setTextColor(
                Color.parseColor(if (selected) "#81D4FA" else "#CFD8DC")
            )
        }
    }

    private fun launchEnvironment(index: Int) {
        if (environments.isEmpty()) return
        handler.removeCallbacks(countdownTick)
        activeEnvIndex = index.coerceIn(0, environments.lastIndex)
        val env = environments[activeEnvIndex]
        inFlyby = true
        envTitleText.text = env.title
        attributionText.text = env.attribution
        startOverlay.visibility = View.GONE
        overlay.visibility = if (hudEnabled) View.VISIBLE else View.GONE
        nativeSetEnvironment(env.assetDir, true)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        if (!applyDebugExtras()) {
            showStartScreen()
        }
    }

    private fun applyDebugExtras(): Boolean {
        val extras = intent?.extras ?: return false
        hudEnabled = extras.getBoolean("hud", true)
        if (extras.getBoolean("startScreen", false)) {
            val reqEnv = extras.getString("env")
            if (reqEnv != null) {
                val idx = environments.indexOfFirst {
                    it.id.equals(reqEnv, ignoreCase = true) || it.assetDir.endsWith("/$reqEnv")
                }
                if (idx >= 0) nextDefaultIndex = idx
            }
            showStartScreen()
            return true
        }

        val hasDebugLaunch = extras.containsKey("env") ||
            extras.containsKey("startTime") ||
            extras.containsKey("freeze") ||
            extras.containsKey("camera")
        if (!hasDebugLaunch) return false

        val reqEnv = extras.getString("env")
        val envIdx = if (reqEnv != null) {
            environments.indexOfFirst {
                it.id.equals(reqEnv, ignoreCase = true) || it.assetDir.endsWith("/$reqEnv")
            }.coerceAtLeast(0)
        } else {
            0
        }
        launchEnvironment(envIdx)

        val startTime = extras.getFloat("startTime", 0f)
        val freeze = extras.getBoolean("freeze", false)
        val camera = extras.getString("camera")
            ?.split(",")
            ?.mapNotNull { it.trim().toFloatOrNull() }
            ?.takeIf { it.size == 6 }
            ?.toFloatArray()
        nativeSetDebugOptions(startTime, freeze, camera)
        return true
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
        if (!inFlyby && environments.isNotEmpty()) {
            handler.removeCallbacks(countdownTick)
            handler.postDelayed(countdownTick, 1000L)
        }
    }

    override fun onPause() {
        super.onPause()
        nativeSetPaused(true)
        handler.removeCallbacks(statusUpdater)
        handler.removeCallbacks(countdownTick)
    }

    override fun onDestroy() {
        handler.removeCallbacks(statusUpdater)
        handler.removeCallbacks(countdownTick)
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
    private external fun nativeIsFlybyCompleted(): Boolean
    private external fun nativeSetEnvironment(assetDir: String, startFlyby: Boolean)
    private external fun nativeSetDebugOptions(startTime: Float, freeze: Boolean, camera: FloatArray?)

    companion object {
        private const val STATUS_INTERVAL_MS = 250L
        private const val COUNTDOWN_SECONDS = 10
        private const val CARDS_PER_PAGE = 2

        init {
            System.loadLibrary("bistro")
        }
    }
}

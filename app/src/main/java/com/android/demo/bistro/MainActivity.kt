package com.android.demo.bistro

import android.annotation.SuppressLint
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
import android.widget.SeekBar
import android.widget.TextView
import java.util.Locale
import kotlin.math.abs
import kotlin.math.roundToInt
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
 *  - `preset` (string "low"|"medium"|"high"|"ultra"): apply a graphics quality preset.
 *  - `graphicsPanel` (boolean): open the Graphics Settings panel on launch.
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

    private data class GraphicsConfig(
        var renderScale: Float = 1.0f,
        var shadowSize: Int = 8192,
        var minClusterPixels: Float = 0.20f,
        var bloomScale: Float = 1.0f,
        var aoIntensityScale: Float = 1.0f,
        var ssrEnabled: Boolean = true,
        var contactShadowsEnabled: Boolean = true,
        var volumetricFogEnabled: Boolean = false,
        var aoEnabled: Boolean = true,
        var shadowPcfEnabled: Boolean = true,
        var taaEnabled: Boolean = false,
        var occlusionCulling: Boolean = true
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
    private lateinit var graphicsPanelContainer: View
    private lateinit var graphicsPanelContent: LinearLayout

    private val handler = Handler(Looper.getMainLooper())
    private val environments = mutableListOf<EnvironmentEntry>()
    private val cardViews = mutableListOf<View>()
    private val cardBadgeViews = mutableListOf<TextView>()

    private val gfx = GraphicsConfig()
    private var cameraFrozen = false
    private var suppressSeekBarCallbacks = false
    private var freezeButtonView: TextView? = null
    private val presetButtonViews = mutableListOf<Pair<String, TextView>>()
    private val shadowSizeButtonViews = mutableListOf<Pair<Int, TextView>>()
    private val toggleChipViews = mutableListOf< Triple<TextView, String, () -> Boolean> >()
    private var scaleLabelView: TextView? = null
    private var scaleSeekBar: SeekBar? = null
    private var cullLabelView: TextView? = null
    private var cullSeekBar: SeekBar? = null
    private var aoLabelView: TextView? = null
    private var aoSeekBar: SeekBar? = null
    private var bloomLabelView: TextView? = null
    private var bloomSeekBar: SeekBar? = null

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
            if (inFlyby || environments.isEmpty() || graphicsPanelContainer.isVisible()) return
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
        graphicsPanelContainer = findViewById(R.id.graphics_panel_container)
        graphicsPanelContent = findViewById(R.id.graphics_panel_content)

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

        findViewById<View>(R.id.graphics_button).setOnClickListener {
            toggleGraphicsPanel()
        }
        findViewById<View>(R.id.start_graphics_button).setOnClickListener {
            toggleGraphicsPanel()
        }

        onBackInvokedDispatcher.registerOnBackInvokedCallback(
            android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT
        ) {
            if (graphicsPanelContainer.isVisible()) {
                setGraphicsPanelVisible(false)
            } else if (inFlyby) {
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
            if (graphicsPanelContainer.isVisible()) {
                setGraphicsPanelVisible(false)
            } else if (inFlyby) {
                overlay.visibility = if (overlay.isVisible()) View.GONE else View.VISIBLE
            }
        }

        discoverEnvironments()
        buildEnvironmentCards()
        buildGraphicsPanel()

        nativeCreate(assets)
        pushGraphicsSettings()
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

    private fun toggleGraphicsPanel() {
        setGraphicsPanelVisible(!graphicsPanelContainer.isVisible())
    }

    private fun setGraphicsPanelVisible(visible: Boolean) {
        graphicsPanelContainer.visibility = if (visible) View.VISIBLE else View.GONE
        if (visible) {
            syncGraphicsPanelUi()
            if (!inFlyby) {
                handler.removeCallbacks(countdownTick)
            }
        } else if (!inFlyby && environments.isNotEmpty()) {
            handler.removeCallbacks(countdownTick)
            handler.postDelayed(countdownTick, 1000L)
        }
    }

    private fun applyPreset(presetName: String) {
        when (presetName.lowercase(Locale.ROOT)) {
            "low" -> {
                gfx.renderScale = 0.60f
                gfx.shadowSize = 2048
                gfx.minClusterPixels = 1.50f
                gfx.bloomScale = 1.0f
                gfx.aoIntensityScale = 0.0f
                gfx.ssrEnabled = false
                gfx.contactShadowsEnabled = false
                gfx.volumetricFogEnabled = false
                gfx.aoEnabled = false
                gfx.shadowPcfEnabled = false
                gfx.taaEnabled = false
                gfx.occlusionCulling = true
            }
            "medium" -> {
                gfx.renderScale = 0.75f
                gfx.shadowSize = 4096
                gfx.minClusterPixels = 0.60f
                gfx.bloomScale = 1.0f
                gfx.aoIntensityScale = 1.0f
                gfx.ssrEnabled = false
                gfx.contactShadowsEnabled = true
                gfx.volumetricFogEnabled = false
                gfx.aoEnabled = true
                gfx.shadowPcfEnabled = true
                gfx.taaEnabled = false
                gfx.occlusionCulling = true
            }
            "high" -> {
                gfx.renderScale = 1.0f
                gfx.shadowSize = 8192
                gfx.minClusterPixels = 0.20f
                gfx.bloomScale = 1.0f
                gfx.aoIntensityScale = 1.0f
                gfx.ssrEnabled = true
                gfx.contactShadowsEnabled = true
                gfx.volumetricFogEnabled = false
                gfx.aoEnabled = true
                gfx.shadowPcfEnabled = true
                gfx.taaEnabled = false
                gfx.occlusionCulling = true
            }
            else -> {
                gfx.renderScale = 1.0f
                gfx.shadowSize = 8192
                gfx.minClusterPixels = 0.20f
                gfx.bloomScale = 1.0f
                gfx.aoIntensityScale = 1.0f
                gfx.ssrEnabled = true
                gfx.contactShadowsEnabled = true
                gfx.volumetricFogEnabled = true
                gfx.aoEnabled = true
                gfx.shadowPcfEnabled = true
                gfx.taaEnabled = true
                gfx.occlusionCulling = true
            }
        }
        pushGraphicsSettings()
        syncGraphicsPanelUi()
    }

    private fun currentPresetName(): String {
        if (abs(gfx.renderScale - 1.0f) < 0.02f && gfx.shadowSize == 8192 &&
            abs(gfx.minClusterPixels - 0.20f) < 0.05f && gfx.ssrEnabled &&
            gfx.contactShadowsEnabled && gfx.volumetricFogEnabled && gfx.aoEnabled &&
            gfx.shadowPcfEnabled && gfx.taaEnabled && gfx.occlusionCulling
        ) return "ultra"
        if (abs(gfx.renderScale - 1.0f) < 0.02f && gfx.shadowSize == 8192 &&
            abs(gfx.minClusterPixels - 0.20f) < 0.05f && gfx.ssrEnabled &&
            gfx.contactShadowsEnabled && !gfx.volumetricFogEnabled && gfx.aoEnabled &&
            gfx.shadowPcfEnabled && !gfx.taaEnabled && gfx.occlusionCulling
        ) return "high"
        if (abs(gfx.renderScale - 0.75f) < 0.02f && gfx.shadowSize == 4096 &&
            abs(gfx.minClusterPixels - 0.60f) < 0.05f && !gfx.ssrEnabled &&
            gfx.contactShadowsEnabled && !gfx.volumetricFogEnabled && gfx.aoEnabled &&
            gfx.shadowPcfEnabled && !gfx.taaEnabled && gfx.occlusionCulling
        ) return "medium"
        if (abs(gfx.renderScale - 0.60f) < 0.02f && gfx.shadowSize == 2048 &&
            abs(gfx.minClusterPixels - 1.50f) < 0.05f && !gfx.ssrEnabled &&
            !gfx.contactShadowsEnabled && !gfx.volumetricFogEnabled && !gfx.aoEnabled &&
            !gfx.shadowPcfEnabled && !gfx.taaEnabled && gfx.occlusionCulling
        ) return "low"
        return ""
    }

    private fun pushGraphicsSettings() {
        nativeSetGraphicsSettings(
            gfx.renderScale,
            gfx.shadowSize,
            gfx.minClusterPixels,
            gfx.bloomScale,
            gfx.aoIntensityScale,
            gfx.ssrEnabled,
            gfx.contactShadowsEnabled,
            gfx.volumetricFogEnabled,
            gfx.aoEnabled,
            gfx.shadowPcfEnabled,
            gfx.taaEnabled,
            gfx.occlusionCulling
        )
    }

    @SuppressLint("SetTextI18n")
    private fun buildGraphicsPanel() {
        graphicsPanelContent.removeAllViews()
        presetButtonViews.clear()
        shadowSizeButtonViews.clear()
        toggleChipViews.clear()

        // 1. Header row: Title + Freeze Camera + Close button
        val headerRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val title = TextView(this).apply {
            text = "Graphics & Performance"
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
            setTypeface(typeface, Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        headerRow.addView(title)

        val freezeBtn = TextView(this).apply {
            background = getDrawable(R.drawable.card_background)
            setPadding(dp(8), dp(4), dp(8), dp(4))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            setTypeface(typeface, Typeface.BOLD)
            setOnClickListener {
                cameraFrozen = !cameraFrozen
                nativeSetCameraFrozen(cameraFrozen)
                syncGraphicsPanelUi()
            }
        }
        val freezeParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { marginEnd = dp(6) }
        headerRow.addView(freezeBtn, freezeParams)
        freezeButtonView = freezeBtn

        val closeBtn = TextView(this).apply {
            text = "✕"
            background = getDrawable(R.drawable.card_background)
            setPadding(dp(9), dp(3), dp(9), dp(3))
            setTextColor(Color.parseColor("#CFD8DC"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setTypeface(typeface, Typeface.BOLD)
            setOnClickListener { setGraphicsPanelVisible(false) }
        }
        headerRow.addView(closeBtn)
        graphicsPanelContent.addView(headerRow)

        fun addSectionLabel(labelText: String): TextView {
            val tv = TextView(this).apply {
                text = labelText
                setTextColor(Color.parseColor("#90A4AE"))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
                setTypeface(typeface, Typeface.BOLD)
                setPadding(0, dp(8), 0, dp(3))
            }
            graphicsPanelContent.addView(tv)
            return tv
        }

        // 2. Quality Presets row
        addSectionLabel("Quality Preset")
        val presetRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        val presets = listOf("low" to "Low", "medium" to "Medium", "high" to "High", "ultra" to "Ultra")
        for ((idx, pair) in presets.withIndex()) {
            val (key, label) = pair
            val btn = TextView(this).apply {
                text = label
                gravity = Gravity.CENTER
                setPadding(0, dp(5), 0, dp(5))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
                setTypeface(typeface, Typeface.BOLD)
                setOnClickListener { applyPreset(key) }
            }
            val lp = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginStart = if (idx == 0) 0 else dp(4)
            }
            presetRow.addView(btn, lp)
            presetButtonViews.add(key to btn)
        }
        graphicsPanelContent.addView(presetRow)

        // 3. Render Resolution Scale slider (50% .. 100%, step 5%)
        scaleLabelView = addSectionLabel("Render Scale: 100%")
        scaleSeekBar = SeekBar(this).apply {
            max = 10 // 0 -> 50%, 10 -> 100%
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    val pct = 50 + progress * 5
                    scaleLabelView?.text = String.format(Locale.US, "Render Scale: %d%%", pct)
                    if (!suppressSeekBarCallbacks) {
                        gfx.renderScale = pct / 100f
                        pushGraphicsSettings()
                        syncPresetHighlight()
                    }
                }
                override fun onStartTrackingTouch(seekBar: SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
            })
        }
        graphicsPanelContent.addView(scaleSeekBar)

        // 4. Shadow Map Resolution segmented buttons (2048, 4096, 8192)
        addSectionLabel("Directional Shadow Map")
        val shadowRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        val shadowSizes = listOf(2048 to "2K (2048)", 4096 to "4K (4096)", 8192 to "8K (8192)")
        for ((idx, pair) in shadowSizes.withIndex()) {
            val (sz, label) = pair
            val btn = TextView(this).apply {
                text = label
                gravity = Gravity.CENTER
                setPadding(0, dp(5), 0, dp(5))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
                setTypeface(typeface, Typeface.BOLD)
                setOnClickListener {
                    gfx.shadowSize = sz
                    pushGraphicsSettings()
                    syncGraphicsPanelUi()
                }
            }
            val lp = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginStart = if (idx == 0) 0 else dp(4)
            }
            shadowRow.addView(btn, lp)
            shadowSizeButtonViews.add(sz to btn)
        }
        graphicsPanelContent.addView(shadowRow)

        // 5. Feature Toggles (2-column compact chip grid)
        addSectionLabel("Rendering Features")
        data class ToggleSpec(val label: String, val get: () -> Boolean, val toggle: () -> Unit)
        val toggleSpecs = listOf(
            ToggleSpec("SSR Reflections", { gfx.ssrEnabled }, { gfx.ssrEnabled = !gfx.ssrEnabled }),
            ToggleSpec("Contact Shadows", { gfx.contactShadowsEnabled }, { gfx.contactShadowsEnabled = !gfx.contactShadowsEnabled }),
            ToggleSpec("Volumetric Rays", { gfx.volumetricFogEnabled }, { gfx.volumetricFogEnabled = !gfx.volumetricFogEnabled }),
            ToggleSpec("Screen-Space AO", { gfx.aoEnabled }, { gfx.aoEnabled = !gfx.aoEnabled }),
            ToggleSpec("8-Tap Shadow PCF", { gfx.shadowPcfEnabled }, { gfx.shadowPcfEnabled = !gfx.shadowPcfEnabled }),
            ToggleSpec("Temporal AA", { gfx.taaEnabled }, { gfx.taaEnabled = !gfx.taaEnabled }),
            ToggleSpec("Hi-Z Occlusion", { gfx.occlusionCulling }, { gfx.occlusionCulling = !gfx.occlusionCulling })
        )
        var rowLayout: LinearLayout? = null
        for ((idx, spec) in toggleSpecs.withIndex()) {
            if (idx % 2 == 0) {
                rowLayout = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    setPadding(0, dp(2), 0, dp(2))
                }
                graphicsPanelContent.addView(rowLayout)
            }
            val chip = TextView(this).apply {
                gravity = Gravity.CENTER
                setPadding(dp(6), dp(5), dp(6), dp(5))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
                setTypeface(typeface, Typeface.BOLD)
                setOnClickListener {
                    spec.toggle()
                    pushGraphicsSettings()
                    syncGraphicsPanelUi()
                }
            }
            val lp = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginStart = if (idx % 2 == 0) 0 else dp(4)
            }
            rowLayout?.addView(chip, lp)
            toggleChipViews.add(Triple(chip, spec.label, spec.get))
        }
        if (toggleSpecs.size % 2 == 1) {
            rowLayout?.addView(
                View(this),
                LinearLayout.LayoutParams(0, 1, 1f).apply { marginStart = dp(4) }
            )
        }

        // 6. Small-Cluster Culling Threshold slider (0.1 px .. 3.0 px)
        cullLabelView = addSectionLabel("Small-Cluster Culling: 0.2 px")
        cullSeekBar = SeekBar(this).apply {
            max = 29 // 0 -> 0.1 px, 29 -> 3.0 px
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    val px = (progress + 1) * 0.10f
                    cullLabelView?.text = String.format(Locale.US, "Small-Cluster Culling: %.1f px", px)
                    if (!suppressSeekBarCallbacks) {
                        gfx.minClusterPixels = px
                        pushGraphicsSettings()
                        syncPresetHighlight()
                    }
                }
                override fun onStartTrackingTouch(seekBar: SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
            })
        }
        graphicsPanelContent.addView(cullSeekBar)

        // 7. SSAO Intensity Scale slider (0% .. 200%, step 10%)
        aoLabelView = addSectionLabel("AO Intensity: 100%")
        aoSeekBar = SeekBar(this).apply {
            max = 20 // 0 -> 0%, 20 -> 200%
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    val pct = progress * 10
                    aoLabelView?.text = String.format(Locale.US, "AO Intensity: %d%%", pct)
                    if (!suppressSeekBarCallbacks) {
                        gfx.aoIntensityScale = pct / 100f
                        pushGraphicsSettings()
                        syncPresetHighlight()
                    }
                }
                override fun onStartTrackingTouch(seekBar: SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
            })
        }
        graphicsPanelContent.addView(aoSeekBar)

        // 8. Bloom Intensity Scale slider (0% .. 200%, step 10%)
        bloomLabelView = addSectionLabel("Bloom Intensity: 100%")
        bloomSeekBar = SeekBar(this).apply {
            max = 20 // 0 -> 0%, 20 -> 200%
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    val pct = progress * 10
                    bloomLabelView?.text = String.format(Locale.US, "Bloom Intensity: %d%%", pct)
                    if (!suppressSeekBarCallbacks) {
                        gfx.bloomScale = pct / 100f
                        pushGraphicsSettings()
                        syncPresetHighlight()
                    }
                }
                override fun onStartTrackingTouch(seekBar: SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
            })
        }
        graphicsPanelContent.addView(bloomSeekBar)

        syncGraphicsPanelUi()
    }

    private fun styleSelectableButton(tv: TextView, selected: Boolean) {
        tv.background = getDrawable(
            if (selected) R.drawable.card_selected_background else R.drawable.card_background
        )
        tv.setTextColor(Color.parseColor(if (selected) "#81D4FA" else "#B0BEC5"))
    }

    private fun syncPresetHighlight() {
        val activePreset = currentPresetName()
        for ((key, btn) in presetButtonViews) {
            styleSelectableButton(btn, key == activePreset)
        }
    }

    @SuppressLint("SetTextI18n")
    private fun syncGraphicsPanelUi() {
        freezeButtonView?.let { btn ->
            btn.text = if (cameraFrozen) "▶ Resume Cam" else "⏸ Freeze Cam"
            styleSelectableButton(btn, cameraFrozen)
        }
        syncPresetHighlight()
        for ((sz, btn) in shadowSizeButtonViews) {
            styleSelectableButton(btn, gfx.shadowSize == sz)
        }
        for ((chip, label, getter) in toggleChipViews) {
            val on = getter()
            chip.text = if (on) "✓ $label" else label
            styleSelectableButton(chip, on)
        }
        suppressSeekBarCallbacks = true
        val scaleProgress = (((gfx.renderScale * 100f).roundToInt() - 50) / 5).coerceIn(0, 10)
        scaleSeekBar?.progress = scaleProgress
        scaleLabelView?.text = String.format(Locale.US, "Render Scale: %d%%", (gfx.renderScale * 100f).roundToInt())

        val cullProgress = ((gfx.minClusterPixels * 10f).roundToInt() - 1).coerceIn(0, 29)
        cullSeekBar?.progress = cullProgress
        cullLabelView?.text = String.format(Locale.US, "Small-Cluster Culling: %.1f px", gfx.minClusterPixels)

        val aoProgress = ((gfx.aoIntensityScale * 10f).roundToInt()).coerceIn(0, 20)
        aoSeekBar?.progress = aoProgress
        aoLabelView?.text = String.format(Locale.US, "AO Intensity: %d%%", (gfx.aoIntensityScale * 100f).roundToInt())

        val bloomProgress = ((gfx.bloomScale * 10f).roundToInt()).coerceIn(0, 20)
        bloomSeekBar?.progress = bloomProgress
        bloomLabelView?.text = String.format(Locale.US, "Bloom Intensity: %d%%", (gfx.bloomScale * 100f).roundToInt())
        suppressSeekBarCallbacks = false
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
        cameraFrozen = false
        nativeSetCameraFrozen(false)
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
        if (!graphicsPanelContainer.isVisible()) {
            handler.postDelayed(countdownTick, 1000L)
        }
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
        cameraFrozen = false
        envTitleText.text = env.title
        attributionText.text = env.attribution
        startOverlay.visibility = View.GONE
        overlay.visibility = if (hudEnabled) View.VISIBLE else View.GONE
        syncGraphicsPanelUi()
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
        extras.getString("preset")?.let { applyPreset(it) }
        if (extras.containsKey("graphicsPanel")) {
            setGraphicsPanelVisible(extras.getBoolean("graphicsPanel", false))
        }
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
        cameraFrozen = freeze
        syncGraphicsPanelUi()
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
        if (!inFlyby && environments.isNotEmpty() && !graphicsPanelContainer.isVisible()) {
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
    private external fun nativeSetGraphicsSettings(
        renderScale: Float,
        shadowSize: Int,
        minClusterPixels: Float,
        bloomScale: Float,
        aoIntensityScale: Float,
        ssrEnabled: Boolean,
        contactShadowsEnabled: Boolean,
        volumetricFogEnabled: Boolean,
        aoEnabled: Boolean,
        shadowPcfEnabled: Boolean,
        taaEnabled: Boolean,
        occlusionCulling: Boolean
    )
    private external fun nativeSetCameraFrozen(frozen: Boolean)

    companion object {
        private const val STATUS_INTERVAL_MS = 250L
        private const val COUNTDOWN_SECONDS = 10
        private const val CARDS_PER_PAGE = 2

        init {
            System.loadLibrary("bistro")
        }
    }
}

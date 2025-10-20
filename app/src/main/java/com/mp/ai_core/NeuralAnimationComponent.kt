package com.mp.ai_core

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.AnimationVector1D
import androidx.compose.animation.core.Easing
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.apply
import kotlin.collections.chunked
import kotlin.collections.forEach
import kotlin.collections.forEachIndexed
import kotlin.collections.isNotEmpty
import kotlin.collections.lastIndex
import kotlin.collections.map
import kotlin.collections.takeLast
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.ranges.coerceIn

/**
 * Configuration for the Neural Animation
 */
data class NeuralAnimationConfig(
    val name: String = "Default",
    val primaryColor: Color = Color(0xFF00F5FF),
    val secondaryColor: Color = Color(0xFFFF006E),
    val accentColor: Color = Color(0xFF8338EC),
    val backgroundColor: Color = Color(0xFF0A0E27),
    val starColor: Color = Color.White,
    val starGlowColor: Color = Color(0xFF00F5FF),
    val baseRotationDuration: Int = 6000,
    val basePulseDuration: Int = 2000,
    val baseOrbitDuration: Int = 4000,
    val initialLayers: Int = 3,
    val initialNeuronCount: Int = 24,
    val starCount: Int = 170
)

/**
 * Predefined Neural Animation Themes
 */
object NeuralThemes {
    val ArcticDawn = NeuralAnimationConfig(
        name = "Arctic Dawn",
        primaryColor = Color(0xFF00BCD4),
        secondaryColor = Color(0xFF00ACC1),
        accentColor = Color(0xFF0F3438),
        backgroundColor = Color(0xFFEFF8FA),
        starColor = Color(0xFF006064),
        starGlowColor = Color(0xFF00BCD4)
    )

    val ArcticTeal = NeuralAnimationConfig(
        name = "Arctic Teal",
        primaryColor = Color(0xFF00E5CC),
        secondaryColor = Color(0xFF00BFA5),
        accentColor = Color(0xFF64FFDA),
        backgroundColor = Color(0xFF051015),
        starColor = Color(0xFFB2DFDB),
        starGlowColor = Color(0xFF00E5CC)
    )

    val CrimsonBlood = NeuralAnimationConfig(
        name = "Crimson Blood",
        primaryColor = Color(0xFFFF1744),
        secondaryColor = Color(0xFFF50057),
        accentColor = Color(0xFFFF4081),
        backgroundColor = Color(0xFF1A0308),
        starColor = Color(0xFFFFCDD2),
        starGlowColor = Color(0xFFFF1744)
    )

    val CoralDawn = NeuralAnimationConfig(
        name = "Coral Dawn",
        primaryColor = Color(0xFFFF5252),
        secondaryColor = Color(0xFFFF1744),
        accentColor = Color(0xFFD32F2F),
        backgroundColor = Color(0xFFFFF5F5),
        starColor = Color(0xFF6D1B1B),
        starGlowColor = Color(0xFFFF5252)
    )

    val MidnightMoss = NeuralAnimationConfig(
        name = "Midnight Moss",
        primaryColor = Color(0xFF689F38),
        secondaryColor = Color(0xFF558B2F),
        accentColor = Color(0xFF8BC34A),
        backgroundColor = Color(0xFF0D1A0A),
        starColor = Color(0xFFDCEDC8),
        starGlowColor = Color(0xFF689F38)
    )

    val SageGarden = NeuralAnimationConfig(
        name = "Sage Garden",
        primaryColor = Color(0xFF689F38),
        secondaryColor = Color(0xFF558B2F),
        accentColor = Color(0xFF33691E),
        backgroundColor = Color(0xFFF9FDF7),
        starColor = Color(0xFF33691E),
        starGlowColor = Color(0xFF689F38)
    )

    /**
     * List of all available themes
     */
    val allThemes = listOf(
        ArcticDawn,
        ArcticTeal,
        CrimsonBlood,
        CoralDawn,
        MidnightMoss,
        SageGarden
    )
}

/**
 * Represents a single neuron in the network
 */
data class Neuron(
    val id: Int,
    val angle: Float,
    val radius: Float,
    val orbitSpeed: Float,
    val pulsePhase: Float,
    val baseSize: Float,
    val layer: Int,
    val alpha: Animatable<Float, AnimationVector1D> = Animatable(0f)
)

/**
 * State management for the Neural Network Animation
 */
class NeuralNetworkState {
    private var _spikeIntensity = mutableFloatStateOf(0f)
    val spikeIntensity: State<Float> = _spikeIntensity

    private var _rotationSpeed = mutableFloatStateOf(1f)
    val rotationSpeed: State<Float> = _rotationSpeed

    private var _pulseSpeed = mutableFloatStateOf(1f)
    val pulseSpeed: State<Float> = _pulseSpeed

    private var _orbitSpeed = mutableFloatStateOf(1f)
    val orbitSpeed: State<Float> = _orbitSpeed

    private var _targetFps = mutableIntStateOf(60)
    val targetFps: State<Int> = _targetFps

    var _currentLayers = mutableIntStateOf(3)
    val currentLayers: State<Int> = _currentLayers

    var _currentNeuronCount = mutableIntStateOf(24)
    val currentNeuronCount: State<Int> = _currentNeuronCount

    private var _isAnimating = mutableStateOf(false)
    val isAnimating: State<Boolean> = _isAnimating

    fun spike(intensity: Float) {
        _spikeIntensity.floatValue = intensity.coerceIn(0f, 1f)
    }

    fun resetSpike() {
        _spikeIntensity.floatValue = 0f
    }

    fun setRotationSpeed(speed: Float) {
        _rotationSpeed.floatValue = speed.coerceIn(0.1f, 10f)
    }

    fun setPulseSpeed(speed: Float) {
        _pulseSpeed.floatValue = speed.coerceIn(0.1f, 10f)
    }

    fun setOrbitSpeed(speed: Float) {
        _orbitSpeed.floatValue = speed.coerceIn(0.1f, 10f)
    }

    fun setAllSpeeds(speed: Float) {
        val clampedSpeed = speed.coerceIn(0.1f, 10f)
        _rotationSpeed.floatValue = clampedSpeed
        _pulseSpeed.floatValue = clampedSpeed
        _orbitSpeed.floatValue = clampedSpeed
    }

    fun setTargetFps(fps: Int) {
        _targetFps.intValue = fps.coerceIn(15, 120)
    }

    suspend fun setLayers(layers: Int, durationMillis: Int = 800) {
        if (_isAnimating.value) return
        _isAnimating.value = true
        _currentLayers.intValue = layers.coerceIn(1, 10)
        delay(durationMillis.toLong())
        _isAnimating.value = false
    }

    suspend fun setNeuronCount(count: Int, durationMillis: Int = 800) {
        if (_isAnimating.value) return
        _isAnimating.value = true
        _currentNeuronCount.intValue = count.coerceIn(6, 100)
        delay(durationMillis.toLong())
        _isAnimating.value = false
    }

    fun getCurrentLayers(): Int = _currentLayers.intValue
    fun getCurrentNeuronCount(): Int = _currentNeuronCount.intValue
}

/**
 * Remember Neural Network State
 */
@Composable
fun rememberNeuralNetworkState(
    initialLayers: Int = 3,
    initialNeuronCount: Int = 24
): NeuralNetworkState {
    return remember {
        NeuralNetworkState().apply {
            _currentLayers.intValue = initialLayers
            _currentNeuronCount.intValue = initialNeuronCount
        }
    }
}

/**
 * Main Neural Animation Composable
 */
@Composable
fun FuturisticNeuralAnimation(
    modifier: Modifier = Modifier,
    config: NeuralAnimationConfig = NeuralAnimationConfig(),
    state: NeuralNetworkState = rememberNeuralNetworkState(
        initialLayers = config.initialLayers,
        initialNeuronCount = config.initialNeuronCount
    )
) {
    val neurons = rememberDynamicNeurons(state)

    // --- Animate colors smoothly ---
    val animatedPrimary = animateColorAsState(config.primaryColor, tween(1000, easing = FastOutSlowInEasing))
    val animatedSecondary = animateColorAsState(config.secondaryColor, tween(1000, easing = FastOutSlowInEasing))
    val animatedAccent = animateColorAsState(config.accentColor, tween(1000, easing = FastOutSlowInEasing))
    val animatedStarGlow = animateColorAsState(config.starGlowColor, tween(1000, easing = FastOutSlowInEasing))
    val animatedBackground = animateColorAsState(config.backgroundColor, tween(1000, easing = FastOutSlowInEasing))

    val stars = remember(config.starCount) {
        List(config.starCount) {
            Triple(Random.nextFloat(), Random.nextFloat(), 1f + Random.nextFloat() * 2.5f)
        }
    }

    val infiniteTransition = rememberInfiniteTransition(label = "neural")
    val sineEasing = Easing { fraction -> 0.5f + 0.5f * sin((fraction * 2 * PI - PI / 2).toFloat()) }

    val rotationSpeedMultiplier = state.rotationSpeed.value
    val pulseSpeedMultiplier = state.pulseSpeed.value
    val orbitSpeedMultiplier = state.orbitSpeed.value
    val fps = state.targetFps.value

    val actualRotationDuration = (config.baseRotationDuration / rotationSpeedMultiplier).toInt()
    val actualPulseDuration = (config.basePulseDuration / pulseSpeedMultiplier).toInt()
    val actualOrbitDuration = (config.baseOrbitDuration / orbitSpeedMultiplier).toInt()

    val frameDelay = 1000L / fps

    val time by infiniteTransition.animateFloat(0f, 360f, infiniteRepeatable(tween(actualRotationDuration, easing = sineEasing), RepeatMode.Restart), "time")
    val pulse by infiniteTransition.animateFloat(0f, 360f, infiniteRepeatable(tween(actualPulseDuration, easing = LinearEasing), RepeatMode.Restart), "pulse")
    val orbit by infiniteTransition.animateFloat(0f, 360f, infiniteRepeatable(tween(actualOrbitDuration, easing = LinearEasing), RepeatMode.Restart), "orbit")

    val spikeValue = state.spikeIntensity.value

    LaunchedEffect(spikeValue) {
        if (spikeValue > 0f) {
            delay(frameDelay)
            state.spike(spikeValue * 0.85f)
        }
    }

    Canvas(modifier = modifier.background(animatedBackground.value)) {
        val center = Offset(size.width / 2, size.height / 2)
        val maxRadius = size.minDimension / 2 * 0.85f

        val positions = neurons.map { neuron ->
            val animAngle = (neuron.angle + time * neuron.orbitSpeed) * PI / 180
            val r = (neuron.radius / 240f) * maxRadius
            val spikeBoost = if (spikeValue > 0) r * 0.3f * spikeValue else 0f
            Triple(Offset(center.x + (r + spikeBoost) * cos(animAngle).toFloat(), center.y + (r + spikeBoost) * sin(animAngle).toFloat()), neuron, neuron.alpha.value)
        }

        // === STARS ===
        stars.forEachIndexed { index, (xRatio, yRatio, starSize) ->
            val x = xRatio * size.width
            val y = yRatio * size.height
            val twinkle = sin((pulse + index * 24f) * PI / 180).toFloat()
            val alpha = 0.3f + twinkle * 0.4f

            // Small star
            drawCircle(
                color = animatedStarGlow.value.copy(alpha = alpha),
                center = Offset(x, y),
                radius = starSize
            )

            // Glow effect every 3rd star
            if (index % 3 == 0) {
                drawCircle(
                    color = animatedStarGlow.value.copy(alpha = alpha * 0.4f),
                    center = Offset(x, y),
                    radius = starSize * 2f
                )
            }
        }

        // === NEBULA CLOUDS ===
        repeat(5) { cloud ->
            val cloudX = size.width * (0.2f + cloud * 0.15f)
            val cloudY = size.height * (0.3f + (cloud % 3) * 0.2f)
            val cloudSize = size.minDimension * (0.2f + cloud * 0.05f)
            drawCircle(
                brush = Brush.radialGradient(listOf(animatedPrimary.value.copy(alpha = 0.08f), animatedSecondary.value.copy(alpha = 0.05f), Color.Transparent), Offset(cloudX, cloudY), cloudSize),
                center = Offset(cloudX, cloudY), radius = cloudSize
            )
        }

        // === SPIKE WAVES ===
        if (spikeValue > 0.1f) {
            repeat(3) { wave ->
                val waveRadius = maxRadius * spikeValue * (1f + wave * 0.3f)
                drawCircle(
                    brush = Brush.radialGradient(listOf(animatedPrimary.value.copy(alpha = spikeValue * 0.4f), Color.Transparent), center, waveRadius),
                    center = center, radius = waveRadius, style = Stroke(width = 2f)
                )
            }
        }

        // === ORBITING PARTICLES ===
        positions.forEach { (pos, neuron, alpha) ->
            if (alpha > 0.1f) {
                val orbitPhase = (orbit + neuron.id * 40f) % 360f
                val orbitRadius = 18f + sin((pulse + neuron.id * 20f) * PI / 180).toFloat() * 5f
                repeat(3) { orbitIndex ->
                    val orbitAngle = (orbitPhase + orbitIndex * 120f) * PI / 180
                    val orbitPos = Offset(pos.x + orbitRadius * cos(orbitAngle).toFloat(), pos.y + orbitRadius * sin(orbitAngle).toFloat())
                    drawCircle(brush = Brush.radialGradient(listOf(animatedAccent.value.copy(alpha = 0.8f * alpha), Color.Transparent), orbitPos, 4f), center = orbitPos, radius = 2f)
                }
            }
        }

        // === CONNECTIONS ===
        positions.forEachIndexed { i, (pos, neuron, alpha) ->
            if (alpha > 0.1f) positions.forEachIndexed { j, (otherPos, _, otherAlpha) ->
                if (i < j && otherAlpha > 0.1f) {
                    val distance = sqrt((pos.x - otherPos.x).pow(2) + (pos.y - otherPos.y).pow(2))
                    val connectionThreshold = maxRadius * 0.45f
                    if (distance < connectionThreshold) {
                        val strength = 1f - (distance / connectionThreshold)
                        val combinedAlpha = (alpha + otherAlpha) / 2f
                        val connectionAlpha = strength * 0.6f * (1f + spikeValue * 2f) * combinedAlpha
                        val flowPhase = ((time + neuron.id * 30f) % 360f) / 360f
                        drawLine(
                            brush = Brush.linearGradient(listOf(animatedPrimary.value.copy(alpha = connectionAlpha * 0.3f), animatedSecondary.value.copy(alpha = connectionAlpha * 0.8f), animatedSecondary.value.copy(alpha = connectionAlpha * 0.6f), animatedPrimary.value.copy(alpha = connectionAlpha * 0.3f)), pos, otherPos),
                            start = pos, end = otherPos, strokeWidth = (1f + strength * 2f) * (1f + spikeValue), cap = StrokeCap.Round
                        )
                        if (strength > 0.7f) {
                            val particlePos = Offset(pos.x + (otherPos.x - pos.x) * flowPhase, pos.y + (otherPos.y - pos.y) * flowPhase)
                            drawCircle(animatedAccent.value.copy(alpha = 0.9f * combinedAlpha), 2f * (1f + spikeValue), particlePos)
                        }
                    }
                }
            }
        }

        // === NEURONS ===
        positions.forEach { (pos, neuron, alpha) ->
            if (alpha > 0.01f) {
                val pulseVal = sin((pulse + neuron.pulsePhase) * PI / 180).toFloat()
                val size = neuron.baseSize * (1f + pulseVal * 0.3f) * (1f + spikeValue * 0.8f)
                val glowSize = size * (3f + spikeValue * 3f)

                // Outer glow
                drawCircle(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            animatedPrimary.value.copy(alpha = (0.5f + spikeValue * 0.5f) * alpha),
                            animatedSecondary.value.copy(alpha = 0.2f * alpha),
                            Color.Transparent
                        ),
                        center = pos,
                        radius = glowSize
                    ),
                    center = pos,
                    radius = glowSize
                )

                // Ring
                drawCircle(
                    color = animatedAccent.value.copy(alpha = (0.7f + pulseVal * 0.3f) * alpha),
                    center = pos,
                    radius = size * 1.3f
                )

                // Core
                val coreColor = if (spikeValue > 0.3f) animatedAccent.value else animatedPrimary.value
                drawCircle(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            Color.White.copy(alpha = alpha),
                            coreColor.copy(alpha = alpha),
                            animatedSecondary.value.copy(alpha = alpha)
                        ),
                        center = pos,
                        radius = size
                    ),
                    center = pos,
                    radius = size
                )

                // Spark
                if (spikeValue > 0.5f) {
                    drawCircle(
                        color = Color.White.copy(alpha = spikeValue * alpha),
                        center = pos,
                        radius = size * 0.4f
                    )
                }
            }
        }

        // === CENTRAL CORE ===
        val coreSize = 12f * (1f + spikeValue * 1.5f)
        drawCircle(brush = Brush.radialGradient(listOf(animatedPrimary.value.copy(alpha = 0.7f + spikeValue * 0.3f), animatedAccent.value.copy(alpha = 0.3f), Color.Transparent), center, coreSize * 4f), center = center, radius = coreSize * 4f)
        drawCircle(brush = Brush.radialGradient(listOf(Color.White, animatedPrimary.value, animatedSecondary.value), center, coreSize), center = center, radius = coreSize)
    }
}

/**
 * Remember and manage dynamic neurons
 */
@Composable
private fun rememberDynamicNeurons(
    state: NeuralNetworkState
): SnapshotStateList<Neuron> {
    val neurons = remember { mutableStateListOf<Neuron>() }
    val currentLayers = state.getCurrentLayers()
    val currentNeuronCount = state.getCurrentNeuronCount()

    LaunchedEffect(currentLayers, currentNeuronCount) {
        val targetSize = currentNeuronCount

        // Remove excess neurons
        if (neurons.size > targetSize) {
            val neuronsToRemove = neurons.size - targetSize
            val toRemove = neurons.takeLast(neuronsToRemove)

            toRemove.forEach { neuron ->
                launch {
                    neuron.alpha.animateTo(
                        0f,
                        animationSpec = tween(150, easing = FastOutSlowInEasing)
                    )
                }
            }

            delay(170)

            repeat(neuronsToRemove) {
                if (neurons.isNotEmpty()) {
                    neurons.removeAt(neurons.lastIndex)
                }
            }
        }

        // Add new neurons
        if (neurons.size < targetSize) {
            val neuronsToAdd = targetSize - neurons.size
            val newNeurons = mutableListOf<Neuron>()

            repeat(neuronsToAdd) {
                val id = neurons.size + it
                val layer = (id * currentLayers) / targetSize
                val neuronsInLayer = kotlin.comparisons.maxOf(1, targetSize / currentLayers)
                val posInLayer = id % neuronsInLayer

                newNeurons.add(
                    Neuron(
                        id = id,
                        angle = (posInLayer * 360f / neuronsInLayer) + (layer * 15f),
                        radius = 80f + (layer * 60f),
                        orbitSpeed = 0.3f + (Random.nextFloat() * 0.4f) * (if (layer % 2 == 0) 1f else -1f),
                        pulsePhase = Random.nextFloat() * 360f,
                        baseSize = 3f + Random.nextFloat() * 2f,
                        layer = layer
                    )
                )
            }

            neurons.addAll(newNeurons)

            val batchSize = 8
            newNeurons.chunked(batchSize).forEach { batch ->
                batch.forEach { neuron ->
                    launch {
                        neuron.alpha.animateTo(
                            1f,
                            animationSpec = tween(200, easing = FastOutSlowInEasing)
                        )
                    }
                }
                delay(30)
            }
        }
    }

    return neurons
}
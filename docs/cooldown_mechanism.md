# Cooldown Mechanism

## The Concept
The cooldown mechanism acts as a circuit breaker for the wake word trigger pipeline.

## How It Works
When you speak the wake word, the inference engine evaluates the incoming audio in overlapping micro-slices. Without a cooldown, saying the wake word once could cause the model to register multiple separate "hits" across those overlapping slices in the span of half a second. 

## Why We Use It
The cooldown locks the gate immediately after the first valid strike. It forces the daemon to temporarily ignore new triggers for a configured number of seconds (e.g., 2-3 seconds). This ensures that OpenClaw receives exactly one clean execution command instead of a chaotic stutter-fire of redundant triggers.
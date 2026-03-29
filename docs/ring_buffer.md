# Ring-Buffered Audio Capture Loop

## The Concept
A ring buffer (or circular buffer) is a continuous, self-eating memory loop. Instead of recording an endless stream of audio that would eventually consume all available RAM, the daemon allocates a fixed, circular block of memory.

## How It Works
As the microphone pulls in fresh audio, the system writes it into this circular buffer. Once it reaches the end of the allocated block, it loops back around to the beginning, constantly overwriting the oldest, useless noise with the present moment.

## Why We Use It
This architecture guarantees that the daemon only ever holds the exact, small window of time needed to catch the wake word. It maintains absolute memory efficiency, allowing the process to listen forever without memory leaks or bloat.
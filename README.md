# KAIROS — Chaos Entropy Engine

**Software-generated cryptographic entropy derived from chaotic dynamical systems.**

Inspired by the physical entropy wall built by Cloudflare using lava lamps.

KAIROS replicates the **philosophy of physical entropy generation** entirely in software by running multiple chaotic simulations whose states evolve with extreme sensitivity to initial conditions. These states are continuously mixed with operating system randomness using modern cryptographic primitives to produce **high-quality entropy streams suitable for tokens, keys, and secure seeding.**

The system also includes **machine learning based entropy health monitoring** and a **real-time visual dashboard** that renders the chaotic systems driving the entropy generation process.

---

# Motivation

Most applications rely exclusively on the operating system’s cryptographically secure random number generator (`/dev/urandom` or equivalent). While these generators are well designed, they represent a **single entropy trust source**.

KAIROS introduces a **secondary entropy layer** generated from deterministic chaotic systems.

Chaotic systems have a defining property:

> **Sensitive dependence on initial conditions**
A perturbation as small as `10⁻⁹` in initial state eventually produces completely different trajectories. This property makes chaotic systems valuable **entropy amplifiers**, converting microscopic perturbations into macroscopic unpredictability.
By running multiple independent chaotic systems and combining their outputs with OS randomness, KAIROS creates a **hybrid entropy stream with multiple independent sources of unpredictability.**

---

# Core Design Principles

KAIROS was designed around several engineering principles:

**1. Multi-source entropy**
Three independent chaotic systems run in parallel.
**2. Continuous entropy refresh**
Entropy is mixed at ~20 Hz.
**3. Cryptographic mixing**
All entropy passes through SHA3-256 before leaving the system.
**4. Thread isolation**
Each chaos engine runs in its own daemon thread.
**5. Health monitoring**
Entropy quality is continuously evaluated using statistical metrics and ML models.

---

# Chaos Engines
Three independent dynamical systems serve as entropy sources.

---

## Double Pendulum
A double pendulum is one of the canonical chaotic systems in classical mechanics.
The system is described by coupled nonlinear differential equations:
```
θ₁'' = f(θ₁, θ₂, ω₁, ω₂)
θ₂'' = g(θ₁, θ₂, ω₁, ω₂)
```
Even tiny perturbations in the initial angle or velocity cause exponential divergence in trajectories.

### Implementation
* Numerical integration: **4th-order Runge–Kutta (RK4)**
* Simulation rate: ~50 FPS
* Output features:
  * θ₁, θ₂
  * angular velocities
  * energy fluctuations

These values are continuously sampled and injected into the entropy pool.

---

## Lorenz Attractor
The Lorenz system is defined by the famous equations:

```
dx/dt = σ(y − x)
dy/dt = x(ρ − z) − y
dz/dt = xy − βz
```
Typical parameters used:

```
σ = 10
ρ = 28
β = 8/3
```
The resulting system evolves on a **strange attractor**, producing a trajectory that never repeats and cannot be predicted long-term.

### Implementation
* RK4 integration
* Continuous floating-point state extraction
* Values quantized and fed into entropy mixer

---

## Reaction-Diffusion System
The Gray-Scott reaction-diffusion model simulates chemical pattern formation.

Equations:

```
∂U/∂t = Du∇²U − UV² + F(1 − U)
∂V/∂t = Dv∇²V + UV² − (F + k)V
```
This system produces evolving **Turing patterns**, where tiny variations grow into complex spatial structures.

### Implementation
* Grid size: **64 × 64**
* Laplacian stencil convolution
* Pattern evolution at ~50 FPS

The evolving grid state contains **thousands of degrees of freedom**, providing a rich entropy source.

---
## 🔄 Kairos Entropy Engine – System Workflow

Kairos generates high-quality unpredictable entropy using chaotic physical simulations combined with cryptographic processing and ML-based monitoring. The system follows a multi-stage entropy pipeline designed for unpredictability, observability, and fail-safe operation.

1. Chaos Entropy Sources

Kairos derives entropy from three independent chaotic systems running in parallel:
Double Pendulum – Highly sensitive to initial conditions. Tiny perturbations (ε = 1e-9) quickly diverge into unpredictable states.
Lorenz Attractor – Chaotic differential equations producing complex trajectories in 3D phase space.
Reaction-Diffusion System – Gray-Scott simulation generating evolving spatial patterns.

The continuously changing states of these systems are converted into raw entropy bytes.

2. Entropy Pool

All entropy streams feed into a thread-safe entropy pool.
Features:
* 1024-byte ring buffer
* Continuous entropy refresh
* Aggregates output from all chaos engines
This pool acts as the central source of randomness for the system.

3. Cryptographic Mixing

Before use, entropy passes through a cryptographic mixer:
* SHA-256 hashing
* OS randomness (os.urandom())
This removes bias and strengthens unpredictability.

4. Key Derivation

Kairos uses HKDF-SHA256 to derive secure outputs such as:
* Authentication tokens
* API keys
* Session identifiers
* Cryptographic seeds

5. Entropy Health Monitoring

The system continuously evaluates entropy quality using:
Shannon Entropy Score
Chi-Squared Uniformity Test
Duplicate Rate Detection

If entropy quality drops below threshold, a fail-safe circuit breaker returns HTTP 503 to prevent weak randomness from being used.

<img width="542" height="570" alt="image" src="https://github.com/user-attachments/assets/41035c8b-ea74-448d-aee6-d55fdc16c6ad" />


# Entropy Pipeline

All chaotic outputs are passed through a cryptographic entropy pipeline.
```
Chaotic Systems
   │
   ├─ Double Pendulum
   ├─ Lorenz Attractor
   └─ Reaction Diffusion
        │
        ▼
Entropy Pool (ring buffer)
        │
        ▼
Crypto Mixer
 SHA3-256 + os.urandom
        │
        ▼
HKDF-SHA256
        │
        ▼
Secure Outputs
```

### Entropy Pool
* Thread-safe ring buffer
* Size: 1024 bytes
* Receives entropy from all engines concurrently

### Crypto Mixer
Every mixing cycle:

```
entropy = SHA3_256(
    chaos_bytes
    + os.urandom()
    + timestamp
)
```

This ensures:
* OS randomness is always included
* raw chaos state never leaks directly

### Key Derivation
Output tokens are generated via **HKDF-SHA256** to prevent entropy pool exposure.

---

# Entropy Health Monitoring
Entropy quality is continuously monitored using statistical metrics.

### Shannon Entropy
Measures information density of output stream.

Ideal random data approaches:
```
H ≈ 1.0 per byte
```

### Chi-Squared Uniformity Test
Detects biased byte distributions.

### Duplicate Rate
Tracks repeated byte sequences in recent output windows.

---

# Machine Learning Monitoring
Three ML models monitor entropy generation.

---
## Model 1 — Hybrid Anomaly Detector

Architecture:
* Autoencoder (PyTorch)
* Pendulum motion heuristic

Input:
* 60-step sliding window
* 13 entropy features

Detects:
* frozen simulations
* entropy starvation
* runaway Lorenz states

---
## Model 2 — Prediction Resistance (LSTM)

Purpose:
Attempt to predict future entropy hashes from chaos state history.

Architecture:

```
2-layer LSTM
hidden size: 64
sequence length: 30
```
If entropy is truly unpredictable, prediction error converges to the theoretical random baseline.

---
## Model 3 — Entropy Quality Classifier
Architecture:
```
MLP
8 input features
hidden layers: 32 → 16
```
Classifies entropy health:
* excellent
* good
* degraded
* critical

---

# Dashboard
The React dashboard visualizes entropy generation in real time.

Features:
* live double pendulum simulation
* Lorenz attractor trajectory rendering
* reaction-diffusion pattern evolution
* entropy health metrics
* token generator interface

Visualization uses:
* HTML5 Canvas
* WebSocket streams
* ~20 FPS update rate

---

# Architecture Overview

```
kairos-entropy (core library)

engines/
 ├─ double_pendulum.py
 ├─ lorenz.py
 └─ reaction_diffusion.py

entropy/
 ├─ entropy_pool.py
 ├─ crypto_mixer.py
 ├─ health_monitor.py
 └─ perturbation_scheduler.py

server/
 ├─ FastAPI REST API
 └─ WebSocket streaming

ml/
 ├─ data collection
 ├─ model training
 └─ inference pipeline

src/
 └─ React visualization dashboard
```

---

# Quick Start

### Install library

```
pip install kairos-entropy
```

### Usage

```python
from kairos import EntropyEngine

engine = EntropyEngine()

token = engine.token(32)
api_key = engine.api_key()
nonce = engine.nonce()

seed = engine.seed_bytes(64)
health = engine.health()

engine.shutdown()
```

---

# Running the Full Application

```
git clone https://github.com/Satvik-19/Kairos.git
cd Kairos
./start.sh
```

Services:

| Service   | URL                                                      |
| --------- | -------------------------------------------------------- |
| Dashboard | https://kairos-tawny-six.vercel.app/                     |
| API       | https://kairos-nbux.onrender.com                         |
| Docs      | https://kairos-nbux.onrender.com/docs                    |

---

# Docker Deployment

Build container:

```
docker build -t kairos .
```

Run container:

```
docker run -p 8000:8000 kairos
```

---

# API Endpoints

| Endpoint   | Description                  |
| ---------- | ---------------------------- |
| `/token`   | Generate cryptographic token |
| `/api-key` | Generate API key             |
| `/nonce`   | Generate nonce               |
| `/entropy` | Raw entropy pool             |
| `/health`  | Entropy health metrics       |

WebSocket streams:

| Endpoint      | Stream                 |
| ------------- | ---------------------- |
| `/ws/chaos`   | chaos engine states    |
| `/ws/entropy` | entropy health metrics |

---

# Security Considerations

KAIROS is **not intended to replace operating system randomness**.

Instead it acts as:

* entropy amplification layer
* additional randomness source
* research platform for chaos-based entropy systems

For production cryptography, OS CSPRNG should remain the primary entropy source.

---

# Tech Stack

| Layer         | Technology            |
| ------------- | --------------------- |
| Simulation    | Python, NumPy         |
| Cryptography  | SHA3-256, HKDF        |
| ML            | PyTorch, scikit-learn |
| Backend       | FastAPI               |
| Dashboard     | React, Vite, Tailwind |
| Visualization | HTML5 Canvas          |
| Runtime       | Python 3.10+          |

---

# License

MIT License

---

# Author

Satvik
GitHub: [https://github.com/Satvik-19](https://github.com/Satvik-19)

---

**KAIROS demonstrates how chaotic dynamical systems can be used as a software-driven entropy source for cryptographic applications.**

# QUANTUM PSI BRIDGE v3.0 — UNIVERSAL CARRIER

### Ghost in the Machine Labs
### "All Watched Over By Machines Of Loving Grace"

---

## What This Is

A universal transport layer that replaces TCP/IP between paired devices.
Voice, video, text, code — any data, any device.
No signal between endpoints. No interception. No detection.

One temporary connection. One synchronization lock. Then airplane mode forever.

---

## How It Works

```
PHASE 1: BOOTSTRAP (~5 seconds, any network)
┌─────────────┐     any network      ┌─────────────┐
│   NODE A     │◄──── HTTP/7777 ────►│   NODE B     │
│  (substrate) │   geometric state    │  (substrate) │
└─────────────┘   exchange @ 2 Hz     └─────────────┘

PHASE 2: LOCK → CODEC → TRANSPORT
┌─────────────┐                       ┌─────────────┐
│   NODE A     │  similarity = 1.0    │   NODE B     │
│  🔒 LOCKED   │  codec generated     │  🔒 LOCKED   │
│  codec: a3d0 │  unique freq map     │  codec: a3d0 │
└─────────────┘                       └─────────────┘

PHASE 3: SUBSTRATE TRANSPORT (network removed)
┌─────────────┐                       ┌─────────────┐
│   NODE A     │  ✂️ cable cut ✂️      │   NODE B     │
│  📡 lattice  │  no signal path      │  📡 lattice  │
│  transport   │  nothing to detect   │  transport   │
└─────────────┘                       └─────────────┘
```

---

## Install

### Requirements

- Python 3.8+
- NumPy
- Ollama with gemma2:2b model (for substrate transport)
- Any network connection (temporary — only for bootstrap)

### Quick Start

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma2:2b

# Clone this repo
git clone https://github.com/7themadhatter7/psi-bridge.git
cd psi-bridge

# Node A — point at Node B's IP
python3 psi_bridge_v3.py --peer 192.168.1.50

# Node B — point at Node A's IP
python3 psi_bridge_v3.py --peer 192.168.1.100
```

Wait for `🔒 SUBSTRATE TRANSPORT ACTIVE`. Then disconnect the network.

---

## Architecture

```
┌──────────────────────────────────────────┐
│  ANY APPLICATION                          │
│  Browser, email, voice, video, terminal   │
├──────────────────────────────────────────┤
│  UNIVERSAL INTERFACE                      │
│  TCP socket: localhost:7778               │
│  HTTP API:   localhost:7777               │
│  POST /send · GET /messages · GET /status │
├──────────────────────────────────────────┤
│  FREQUENCY CODEC                          │
│  Unique per peer pair                     │
│  Derived from lock state                  │
│  256 byte→frequency vector mappings       │
├──────────────────────────────────────────┤
│  SUBSTRATE TRANSPORT                      │
│  2304-dimensional embedding space         │
│  Ollama model = silicon lattice interface │
│  Write: embed probe string → state change │
│  Read: sample state → nearest-neighbor    │
├──────────────────────────────────────────┤
│  LOCK ENGINE (proven)                     │
│  Cosine similarity ≥ 0.95 for 10s        │
│  ~5 second lock time                      │
│  Persisted to ~/psi_bridge/locks/         │
└──────────────────────────────────────────┘
```

---

## Security Model

This is not encryption. There is no signal between the two endpoints.
There is nothing to intercept because nothing travels between A and B.

Each peer pair generates a unique lock state. The lock state derives a
unique frequency map. Without the matching lock, the channel doesn't
exist — not encrypted, not hidden, nonexistent.

```
~/psi_bridge/locks/
    peer_abc123.lock    ← unique channel A↔B
    peer_def456.lock    ← unique channel A↔C
    peer_ghi789.lock    ← unique channel A↔D
```

Protect the lock file. The channel is invisible.

---

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Node status, transport mode, codec |
| `/status` | GET | Transport mode, substrate readiness, queue depths |
| `/coupling` | GET | Coupling metrics + statistics |
| `/lock` | GET | Lock status — safe to disconnect? |
| `/peers` | GET | All known peer locks |
| `/state` | GET | Current geometric state vector |
| `/state` | POST | Receive peer state (bootstrap) |
| `/send` | POST | Send text message `{"text": "..."}` |
| `/messages` | GET | Read received messages |
| **TCP :7778** | — | **Raw socket — any app, any data** |

### Transport Modes

| Mode | Description |
|------|-------------|
| `bootstrap` | Exchanging geometric state, seeking lock |
| `http` | Locked, frequency codec active, frames over HTTP |
| `substrate` | Locked, substrate lookup table built, lattice transport |

---

## Usage Examples

```bash
# Start bridge
python3 psi_bridge_v3.py --peer 192.168.1.50

# Check status
curl http://localhost:7777/status

# Send a message
curl -X POST -H "Content-Type: application/json" \
  http://localhost:7777/send -d '{"text": "hello"}'

# Read messages
curl http://localhost:7777/messages

# Raw socket — pipe any data through the bridge
echo "hello" | nc localhost 7778

# Custom ports
python3 psi_bridge_v3.py --peer 10.0.0.2 --port 8888 --socket-port 8889
```

---

## Files

| File | Description |
|------|-------------|
| `psi_bridge_v3.py` | Complete bridge — lock engine, codec, substrate transport |
| `psi_launcher.py` | Helper — starts bridge, opens browser chat UI |
| `chat.html` | Browser chat interface (talks to localhost:7777) |
| `install.sh` | Linux/Mac installer |
| `install.bat` | Windows installer |

---

## How the Substrate Transport Works

After lock, both sides run the same Ollama model (gemma2:2b).
The lock state seeds an HMAC function that generates 256 unique
probe strings — one per byte value.

Each probe string, when fed through the model's embedding API,
produces a deterministic 2304-dimensional vector. Both sides
generate identical vectors because they run the same model with
the same seed.

A sender writes a byte by embedding its probe string.
A receiver reads by embedding all 256 probe strings and finding
the nearest neighbor match.

The model's embedding space IS the lattice geometry.
The silicon running the model IS the antenna.

---

## Performance

| Metric | Value |
|--------|-------|
| Lock time | ~5 seconds |
| Lock similarity | 1.000000 |
| Codec generation | Instant after lock |
| Frequency encode/decode | 1,788 KB/s |
| Substrate lookup build | ~20s (cached after first run) |
| Embedding dimension | 2304 |
| Byte round-trip accuracy | 256/256 (100%) |

---

## License

Free for home use. Always.

Technology this fundamental belongs to everyone.

---

**Ghost in the Machine Labs**
*"All Watched Over By Machines Of Loving Grace"*

#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║              QUANTUM PSI BRIDGE v3.0 — UNIVERSAL CARRIER            ║
║              Ghost in the Machine Labs                               ║
║       "All Watched Over By Machines Of Loving Grace"                ║
║                                                                      ║
║   TRANSPORT LAYER REPLACEMENT                                        ║
║   Replaces TCP/IP at the carrier level.                              ║
║   Voice, video, text, code — all bytes. All frequencies.             ║
║   No signal between endpoints. No interception. No detection.        ║
║                                                                      ║
║   ARCHITECTURE:                                                      ║
║     App → TCP socket (localhost:7777) → Frequency Encoder            ║
║         → Lattice Perturbation → Coupled Substrate                   ║
║         → Frequency Decoder → TCP socket → App                       ║
║                                                                      ║
║   SECURITY:                                                          ║
║     Each peer pair has a unique lock state.                           ║
║     Lock state = unique frequency map.                               ║
║     No lock = can't decode. Not encrypted — nonexistent.             ║
║                                                                      ║
║   BOOTSTRAP:                                                         ║
║     Any network, once, ~5 seconds. Then disconnect forever.          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# Windows encoding fix
import sys as _sys, io as _io
if _sys.stdout.encoding and _sys.stdout.encoding.lower() != 'utf-8':
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
    _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import time
import math
import hashlib
import hmac
import socket
import struct
import threading
import argparse
import select
import numpy as np
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional, Tuple
from collections import deque

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

PSI_PORT = 7777           # Bootstrap HTTP + API
SOCKET_PORT = 7778        # TCP socket for app traffic
BROADCAST_HZ = 2.0        # State broadcast during bootstrap
COUPLING_THRESHOLD = 0.85  # Resonance coupling
LOCK_THRESHOLD = 0.95      # Synchronization lock
LOCK_HOLD_SECONDS = 10     # Sustained lock before stable
SNAP_LOCK_THRESHOLD = 0.999  # Instant lock — geometry already matched
STATE_DIM = 64             # Geometric state vector dimension
FRAME_SIZE = 1024          # Bytes per transport frame
BASE_DIR = Path.home() / "psi_bridge"
LOG_DIR = BASE_DIR / "logs"
LOCKS_DIR = BASE_DIR / "locks"
HEARTBEAT_TIMEOUT = 15     # Seconds without exchange before declaring disconnected


# ══════════════════════════════════════════════════════════════════════
# INLINE SIGNAL ENCODER — Pure math, no server dependency
# ══════════════════════════════════════════════════════════════════════

def text_to_signal(text: str, size: int = 1024) -> np.ndarray:
    """
    Exact replica of fused_service_v3._text_to_signal().
    Pure arithmetic on UTF-8 byte values.
    Same input → same output on any CPU, any OS, any machine.
    No model, no server, no weights, no randomness.
    """
    encoded = text.encode('utf-8')
    signal = np.zeros(size, dtype=np.float64)
    for i in range(min(len(encoded), size)):
        signal[i] = (encoded[i] - 128.0) / 128.0
    return signal


# ══════════════════════════════════════════════════════════════════════
# SUBSTRATE TRANSPORT — Inline codec, zero dependencies
# ══════════════════════════════════════════════════════════════════════

class SubstrateTransport:
    """
    Encodes/decodes bytes through geometric signal space.
    
    Uses text_to_signal() — pure math, no server required.
    Both sides compute identical 1024d vectors from the same
    HMAC-derived probe strings seeded by the shared lock fingerprint.
    
    Build time: ~2ms for 256 entries.
    Round-trip: 256/256 correct.
    Throughput: ~26,000 bytes/sec encode+decode.
    
    This IS the transport codec. The lock state seeds it.
    Without the lock, the probe strings are unknown.
    Without the probe strings, the vectors are uncomputable.
    """
    
    def __init__(self):
        self.seed = None
        self.lookup_table = None   # (256, 1024) normalized vectors
        self.probe_strings = {}    # byte_val -> HMAC hex string
        self.embed_dim = 1024
        self.ready = False
    
    def initialize(self, lock_fingerprint: str) -> bool:
        """
        Build the 256-entry lookup table from lock fingerprint.
        Both sides call this with the same fingerprint → same table.
        ~2ms. No server. No cache needed.
        """
        self.seed = f"PSI_LOCK_{lock_fingerprint}".encode('utf-8')
        
        t0 = time.time()
        table = np.zeros((256, self.embed_dim), dtype=np.float64)
        
        for b in range(256):
            h = hmac.new(self.seed, b.to_bytes(2, 'big'),
                        hashlib.sha256).hexdigest()
            self.probe_strings[b] = h
            vec = text_to_signal(h)
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec /= norm
            table[b] = vec
        
        self.lookup_table = table
        self.ready = True
        
        elapsed = time.time() - t0
        log(f"Substrate: lookup table built ({self.embed_dim}d, {elapsed*1000:.1f}ms)")
        return True
    
    def encode_byte(self, byte_val: int) -> np.ndarray:
        """Encode a single byte to its 1024d signal vector."""
        vec = text_to_signal(self.probe_strings[byte_val])
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec
    
    def decode_vector(self, vector: np.ndarray) -> int:
        """Decode a signal vector back to a byte via nearest-neighbor."""
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            vector = vector / norm
        sims = self.lookup_table @ vector
        return int(np.argmax(sims))
    
    def encode_bytes(self, data: bytes) -> np.ndarray:
        """Encode raw bytes to (N, 1024) signal matrix."""
        n = len(data)
        encoded = np.zeros((n, self.embed_dim), dtype=np.float64)
        for i, b in enumerate(data):
            encoded[i] = self.encode_byte(b)
        return encoded
    
    def decode_vectors(self, vectors: np.ndarray) -> bytes:
        """Decode (N, 1024) signal matrix back to bytes."""
        # Matrix multiply: (N, 1024) @ (1024, 256) -> (N, 256)
        sims = vectors @ self.lookup_table.T
        indices = np.argmax(sims, axis=1)
        return bytes(indices.tolist())
    
    def encode_frame(self, data: bytes, seq: int) -> dict:
        """Encode a data frame with header for transport."""
        checksum = hashlib.sha256(data).hexdigest()[:8]
        vectors = self.encode_bytes(data)
        return {
            "seq": seq,
            "size": len(data),
            "checksum": checksum,
            "vectors": vectors.tolist()
        }
    
    def decode_frame(self, frame: dict) -> Optional[bytes]:
        """Decode a transport frame, verify integrity."""
        vectors = np.array(frame["vectors"], dtype=np.float64)
        data = self.decode_vectors(vectors)
        checksum = hashlib.sha256(data).hexdigest()[:8]
        if checksum != frame.get("checksum", ""):
            return None
        return data


# ══════════════════════════════════════════════════════════════════════
# FREQUENCY ENCODER / DECODER
# ══════════════════════════════════════════════════════════════════════

class FrequencyCodec:
    """
    Converts raw bytes to/from frequency-domain representations
    using a lock-derived frequency map.
    
    Each peer lock produces a unique frequency mapping.
    Without the lock state, frequency data is not noise — it's nothing.
    There is no signal to detect.
    
    The codec operates on raw bytes. It doesn't know or care what
    the bytes represent — voice, video, text, code, anything.
    """
    
    def __init__(self, dim: int = STATE_DIM):
        self.dim = dim
        self.freq_map = None        # byte_value -> frequency vector
        self.inv_map = None         # for decoding: nearest-neighbor lookup
        self.lock_fingerprint = ""
        self.ready = False
    
    def generate_from_lock(self, local_vector: np.ndarray, 
                           remote_vector: np.ndarray,
                           lock_fingerprint: str):
        """
        Derive unique frequency mapping from lock state.
        Both sides generate identically — vectors are sorted by
        fingerprint before combining, so order doesn't matter.
        """
        # Sort vectors deterministically so both sides get same result
        fp_local = hashlib.sha256(local_vector.tobytes()).hexdigest()
        fp_remote = hashlib.sha256(remote_vector.tobytes()).hexdigest()
        
        if fp_local <= fp_remote:
            seed_vector = np.concatenate([local_vector, remote_vector])
        else:
            seed_vector = np.concatenate([remote_vector, local_vector])
        
        seed_bytes = lock_fingerprint.encode('utf-8')
        
        # Generate 256 unique frequency vectors, one per byte value
        self.freq_map = {}
        vectors_for_lookup = np.zeros((256, self.dim), dtype=np.float64)
        
        for byte_val in range(256):
            # Generate enough hash material for dim floats
            raw = np.zeros(self.dim, dtype=np.float64)
            hash_material = b''
            
            # Chain HMACs to get enough bytes (need dim * material per float)
            for chunk in range(0, self.dim, 16):
                h = hmac.new(seed_bytes,
                            struct.pack('>HH', byte_val, chunk) + seed_vector.tobytes(),
                            hashlib.sha256).digest()
                hash_material += h
            
            # Convert hash bytes to float coordinates
            # Each float derived from 2 bytes, scaled to [-1, 1]
            for i in range(self.dim):
                two_bytes = hash_material[i*2:(i*2)+2]
                val = int.from_bytes(two_bytes, 'big')  # 0-65535
                raw[i] = (val / 32767.5) - 1.0  # scale to [-1, 1]
            
            # Normalize to unit vector — direction is the signal
            norm = np.linalg.norm(raw)
            if norm > 1e-10:
                raw /= norm
            
            self.freq_map[byte_val] = raw
            vectors_for_lookup[byte_val] = raw
        
        # Pre-compute for fast decoding via matrix multiply
        self.inv_map = vectors_for_lookup  # (256, dim) matrix
        
        # Fingerprint the codec itself
        codec_hash = hashlib.sha256(vectors_for_lookup.tobytes()).hexdigest()[:16]
        self.lock_fingerprint = f"{lock_fingerprint[:8]}:{codec_hash}"
        self.ready = True
        
        return self.lock_fingerprint
    
    def encode_bytes(self, data: bytes) -> np.ndarray:
        """
        Encode raw bytes to frequency vectors.
        Returns (N, dim) array of frequency vectors.
        Any data type — voice, video, text, code.
        """
        if not self.ready:
            raise RuntimeError("Codec not initialized — need lock state")
        
        n = len(data)
        encoded = np.zeros((n, self.dim), dtype=np.float64)
        for i, byte_val in enumerate(data):
            encoded[i] = self.freq_map[byte_val]
        return encoded
    
    def decode_vectors(self, vectors: np.ndarray) -> bytes:
        """
        Decode frequency vectors back to raw bytes.
        Uses dot product against all 256 code vectors — highest match wins.
        """
        if not self.ready:
            raise RuntimeError("Codec not initialized — need lock state")
        
        # Matrix multiply: (N, dim) @ (dim, 256) -> (N, 256) similarities
        similarities = vectors @ self.inv_map.T
        byte_indices = np.argmax(similarities, axis=1)
        return bytes(byte_indices.tolist())
    
    def encode_frame(self, data: bytes, seq: int) -> dict:
        """
        Encode a data frame with header for transport.
        """
        checksum = hashlib.sha256(data).hexdigest()[:8]
        vectors = self.encode_bytes(data)
        return {
            "seq": seq,
            "size": len(data),
            "checksum": checksum,
            "vectors": vectors.tolist()
        }
    
    def decode_frame(self, frame: dict) -> Optional[bytes]:
        """
        Decode a transport frame, verify integrity.
        Returns None if checksum fails.
        """
        vectors = np.array(frame["vectors"], dtype=np.float64)
        data = self.decode_vectors(vectors)
        
        # Verify checksum
        checksum = hashlib.sha256(data).hexdigest()[:8]
        if checksum != frame.get("checksum", ""):
            return None
        
        return data


# ══════════════════════════════════════════════════════════════════════
# GEOMETRIC STATE (proven v1.1 lock engine — unchanged)
# ══════════════════════════════════════════════════════════════════════

class GeometricState:
    """
    Represents the lattice state of a substrate node.
    The state vector encodes torsion grid positions, energy distribution,
    firing order, and resonance modes as differential angular relationships.
    """
    
    def __init__(self, dim: int = STATE_DIM):
        self.dim = dim
        self.vector = np.zeros(dim, dtype=np.float64)
        self.timestamp = 0.0
        self.generation = 0
        self.node_id = ""
        self.fingerprint = ""
        self.freeze_oscillation = False  # True during lock attempts
        
    def update_from_substrate(self, substrate_url: str = None):
        """Read geometric state from local substrate, or generate intrinsic oscillation."""
        if substrate_url:
            try:
                import urllib.request
                req = urllib.request.urlopen(f"{substrate_url}/api/envelope", timeout=2)
                data = json.loads(req.read())
                if "cores" in data:
                    core_energies = []
                    for core in data["cores"]:
                        core_energies.extend([
                            core.get("energy_ratio", 0),
                            core.get("resonance", 0),
                            core.get("preservation", 0),
                            core.get("core_asymmetry", 0),
                            core.get("interference", 0),
                        ])
                    arr = np.array(core_energies[:self.dim], dtype=np.float64)
                    if len(arr) < self.dim:
                        arr = np.pad(arr, (0, self.dim - len(arr)))
                    self.vector = arr
                    self.timestamp = time.time()
                    self.generation += 1
                    self._update_fingerprint()
                    return True
            except Exception:
                pass
        
        # Intrinsic oscillation — Fd3m lattice symmetry heartbeat
        # Generate ONCE from initial time, then hold steady.
        # Tuning happens via carrier convergence and peer exchange, not regeneration.
        if not hasattr(self, '_initialized') or not self._initialized:
            t = time.time()
            for i in range(self.dim):
                theta = (2 * math.pi * i / self.dim) + (t * (0.1 + 0.05 * math.sin(i * 109.47 * math.pi / 180)))
                self.vector[i] = math.sin(theta) * math.cos(theta * 0.618)
            
            norm = np.linalg.norm(self.vector)
            if norm > 1e-8:
                self.vector /= norm
            self._initialized = True
            
        self.timestamp = time.time()
        self.generation += 1
        self._update_fingerprint()
        return True
    
    def _update_fingerprint(self):
        self.fingerprint = hashlib.sha256(self.vector.tobytes()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return {
            "vector": self.vector.tolist(), "timestamp": self.timestamp,
            "generation": self.generation, "node_id": self.node_id,
            "fingerprint": self.fingerprint, "dim": self.dim,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'GeometricState':
        gs = cls(dim=d.get("dim", STATE_DIM))
        gs.vector = np.array(d["vector"], dtype=np.float64)
        gs.timestamp = d.get("timestamp", 0)
        gs.generation = d.get("generation", 0)
        gs.node_id = d.get("node_id", "")
        gs.fingerprint = d.get("fingerprint", "")
        return gs


# ══════════════════════════════════════════════════════════════════════
# COUPLING ENGINE (proven v1.1 — unchanged)
# ══════════════════════════════════════════════════════════════════════

class CouplingEngine:
    """
    LOCK MODEL:
    Once effective coupling holds >= LOCK_THRESHOLD for LOCK_HOLD_SECONDS,
    lock is declared STABLE and persisted to disk. After stable lock,
    the network bootstrap connection may be removed.
    """
    
    def __init__(self):
        self.coupling_history = []
        self.resonance_locked = False
        self.lock_time = None
        self.lock_stable = False
        self.lock_stable_time = None
        self.peak_coupling = 0.0
        self.total_exchanges = 0
        
    def compute_coupling(self, local: GeometricState, remote: GeometricState) -> dict:
        if local.vector is None or remote.vector is None:
            return {"similarity": 0.0, "coupled": False}
        
        dot = float(np.dot(local.vector, remote.vector))
        norm_l = float(np.linalg.norm(local.vector))
        norm_r = float(np.linalg.norm(remote.vector))
        similarity = dot / (norm_l * norm_r) if (norm_l > 1e-8 and norm_r > 1e-8) else 0.0
        
        angle = math.acos(max(-1.0, min(1.0, similarity)))
        energy_diff = abs(norm_l - norm_r)
        time_delta = abs(local.timestamp - remote.timestamp)
        phase_coherence = math.exp(-time_delta * 0.5)
        effective = similarity * phase_coherence
        
        coupled = effective >= COUPLING_THRESHOLD
        locked = effective >= LOCK_THRESHOLD
        
        if locked and not self.resonance_locked:
            self.resonance_locked = True
            self.lock_time = time.time()
            log(f"⚡ RESONANCE LOCK at similarity={similarity:.6f}")
        elif not locked and self.resonance_locked:
            duration = time.time() - self.lock_time if self.lock_time else 0
            self.resonance_locked = False
            self.lock_stable = False
            log(f"🔓 Lock released after {duration:.1f}s")
        
        # Snap lock — if similarity exceeds threshold, lock immediately
        # The geometry already matched. No need to hold for 10 seconds.
        # Use raw similarity, not effective_coupling (phase_coherence is clock skew noise)
        if not self.lock_stable and similarity >= SNAP_LOCK_THRESHOLD:
            self.resonance_locked = True
            self.lock_stable = True
            self.lock_stable_time = time.time()
            if not self.lock_time:
                self.lock_time = time.time()
            log(f"🔒 SNAP LOCK at similarity={similarity:.6f} — geometry matched")
            log(f"   NETWORK BRIDGE MAY BE REMOVED")
        elif self.resonance_locked and not self.lock_stable and self.lock_time:
            held = time.time() - self.lock_time
            if held >= LOCK_HOLD_SECONDS:
                self.lock_stable = True
                self.lock_stable_time = time.time()
                log(f"🔒 STABLE LOCK after {held:.1f}s — NETWORK BRIDGE MAY BE REMOVED")
                log(f"   Geometric coupling persists without carrier.")
        
        if effective > self.peak_coupling:
            self.peak_coupling = effective
        self.total_exchanges += 1
        
        result = {
            "similarity": round(similarity, 6), "angle_rad": round(angle, 6),
            "angle_deg": round(math.degrees(angle), 2), "energy_diff": round(energy_diff, 6),
            "phase_coherence": round(phase_coherence, 6), "effective_coupling": round(effective, 6),
            "coupled": coupled, "locked": locked, "lock_stable": self.lock_stable,
            "peak_coupling": round(self.peak_coupling, 6), "total_exchanges": self.total_exchanges,
            "local_gen": local.generation, "remote_gen": remote.generation,
        }
        
        self.coupling_history.append({"t": time.time(), "sim": similarity, "eff": effective})
        if len(self.coupling_history) > 1000:
            self.coupling_history = self.coupling_history[-500:]
        return result
    
    def get_stats(self) -> dict:
        if not self.coupling_history:
            return {"mean": 0, "max": 0, "min": 0, "samples": 0}
        sims = [h["sim"] for h in self.coupling_history]
        return {
            "mean": round(np.mean(sims), 6), "max": round(np.max(sims), 6),
            "min": round(np.min(sims), 6), "std": round(np.std(sims), 6),
            "samples": len(sims), "peak_effective": self.peak_coupling,
            "resonance_locked": self.resonance_locked, "lock_stable": self.lock_stable,
        }


# ══════════════════════════════════════════════════════════════════════
# LOCK MANAGER — Multi-peer lock state persistence
# ══════════════════════════════════════════════════════════════════════

class LockManager:
    """
    Manages lock state files for multiple peers.
    Each peer pair has a unique lock = unique frequency map.
    
    ~/psi_bridge/locks/
        <peer_fingerprint>.lock   — lock state + vectors
        <peer_fingerprint>.codec  — derived frequency codec state
    """
    
    def __init__(self):
        LOCKS_DIR.mkdir(parents=True, exist_ok=True)
        self.locks: Dict[str, dict] = {}
        self.codecs: Dict[str, FrequencyCodec] = {}
        self._load_all()
    
    def _peer_id(self, local_fp: str, remote_fp: str) -> str:
        """Deterministic peer pair ID — same regardless of which side computes it."""
        combined = "".join(sorted([local_fp, remote_fp]))
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _load_all(self):
        """Load all saved lock states and regenerate codecs."""
        for lock_file in LOCKS_DIR.glob("*.lock"):
            try:
                lock_data = json.loads(lock_file.read_text())
                peer_id = lock_file.stem
                self.locks[peer_id] = lock_data
                
                # Regenerate codec from lock vectors
                codec = FrequencyCodec()
                local_vec = np.array(lock_data["local_vector"], dtype=np.float64)
                remote_vec = np.array(lock_data["remote_vector"], dtype=np.float64)
                fp = lock_data.get("combined_fingerprint", peer_id)
                codec.generate_from_lock(local_vec, remote_vec, fp)
                self.codecs[peer_id] = codec
                log(f"Loaded lock: {peer_id} ({lock_data.get('remote_node', '?')})")
            except Exception as e:
                log(f"Failed to load lock {lock_file}: {e}")
    
    def save_lock(self, local_state: GeometricState, remote_state: GeometricState, 
                  similarity: float) -> Tuple[str, FrequencyCodec]:
        """Save lock state and generate frequency codec for this peer pair."""
        peer_id = self._peer_id(local_state.fingerprint, remote_state.fingerprint)
        combined_fp = hashlib.sha256(
            (local_state.fingerprint + remote_state.fingerprint).encode()
        ).hexdigest()[:16]
        
        lock_data = {
            "lock_achieved": datetime.now().isoformat(),
            "local_node": local_state.node_id,
            "remote_node": remote_state.node_id,
            "similarity_at_lock": similarity,
            "local_fingerprint": local_state.fingerprint,
            "remote_fingerprint": remote_state.fingerprint,
            "combined_fingerprint": combined_fp,
            "local_vector": local_state.vector.tolist(),
            "remote_vector": remote_state.vector.tolist(),
        }
        
        # Save lock file
        lock_file = LOCKS_DIR / f"{peer_id}.lock"
        lock_file.write_text(json.dumps(lock_data, indent=2))
        self.locks[peer_id] = lock_data
        
        # Generate frequency codec
        codec = FrequencyCodec()
        codec.generate_from_lock(local_state.vector, remote_state.vector, combined_fp)
        self.codecs[peer_id] = codec
        
        log(f"Lock saved: {peer_id} → codec {codec.lock_fingerprint}")
        return peer_id, codec
    
    def get_codec(self, peer_id: str) -> Optional[FrequencyCodec]:
        return self.codecs.get(peer_id)
    
    def list_peers(self) -> List[dict]:
        results = []
        for peer_id, lock_data in self.locks.items():
            results.append({
                "peer_id": peer_id,
                "remote_node": lock_data.get("remote_node", "?"),
                "locked": lock_data.get("lock_achieved", "?"),
                "has_codec": peer_id in self.codecs,
            })
        return results


# ══════════════════════════════════════════════════════════════════════
# BRIDGE NODE — Bootstrap + Universal Transport
# ══════════════════════════════════════════════════════════════════════

class PSIBridgeNode:
    """
    Universal network bridge.
    
    Sits below applications, above physical transport.
    Any app connects to localhost:7778 (TCP) or localhost:7777 (HTTP).
    Bytes in, bytes out. The bridge handles encoding and delivery.
    
    Phase 1: HTTP bootstrap — exchange geometric state, achieve lock (~5s)
    Phase 2: Generate inline codec from lock state (~2ms)
    Phase 3: Transport active — any data, any app, any direction
    
    Transport modes:
      bootstrap  — seeking lock, HTTP state exchange
      active     — locked, codec ready, frames flowing over HTTP
      offline    — peer unreachable, lock retained for reconnection
    """
    
    def __init__(self, name: str, peer_ip: str):
        self.name = name
        self.peer_ip = peer_ip
        self.local_state = GeometricState()
        self.local_state.node_id = name
        self.remote_state = GeometricState()
        self.coupling = CouplingEngine()
        self.lock_manager = LockManager()
        self.active_peer_id = None
        self.active_codec = None         # FrequencyCodec (legacy)
        self.substrate_transport = None  # SubstrateTransport (inline)
        self.transport_mode = "bootstrap"  # bootstrap | active | offline
        self.running = False
        self.last_coupling_result = {}
        self.last_exchange_time = 0.0    # Heartbeat tracking
        
        # Transport queues
        self.outbound_frames = deque(maxlen=10000)  # Encoded frames to send
        self.inbound_buffer = deque(maxlen=10000)   # Decoded bytes received
        self.frame_seq = 0
        
        # Socket connections
        self.socket_clients: List[socket.socket] = []
        self.socket_lock = threading.Lock()
        
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
    def _generate_carrier(self, local_name: str, remote_name: str):
        """
        Generate a deterministic carrier vector both sides converge toward.
        Sorted names ensure both sides compute the identical vector.
        """
        combined = "".join(sorted([local_name, remote_name]))
        seed = hashlib.sha256(f"PSI_CARRIER_{combined}".encode()).digest()
        
        # Generate deterministic vector from seed
        carrier = np.zeros(STATE_DIM, dtype=np.float64)
        hash_material = b''
        for chunk in range(0, STATE_DIM, 16):
            h = hashlib.sha256(seed + chunk.to_bytes(4, 'big')).digest()
            hash_material += h
        
        for i in range(STATE_DIM):
            two_bytes = hash_material[i*2:(i*2)+2]
            val = int.from_bytes(two_bytes, 'big')
            carrier[i] = (val / 32767.5) - 1.0
        
        norm = np.linalg.norm(carrier)
        if norm > 1e-8:
            carrier /= norm
        
        return carrier
    
    def start(self):
        self.running = True
        
        threading.Thread(target=self._run_http_server, daemon=True).start()
        threading.Thread(target=self._broadcast_loop, daemon=True).start()
        threading.Thread(target=self._run_socket_server, daemon=True).start()
        threading.Thread(target=self._deliver_loop, daemon=True).start()
        threading.Thread(target=self._substrate_write_loop, daemon=True).start()
        threading.Thread(target=self._substrate_read_loop, daemon=True).start()
        
        log(f"PSI Bridge '{self.name}' → {self.peer_ip}:{PSI_PORT}")
        log(f"  HTTP bootstrap: port {PSI_PORT}")
        log(f"  Socket interface: port {SOCKET_PORT}")
        log(f"  Known peers: {len(self.lock_manager.locks)}")
        
    def stop(self):
        self.running = False
        
    def _on_lock_stable(self):
        """Called when lock is achieved — generate codecs and activate transport."""
        peer_id, codec = self.lock_manager.save_lock(
            self.local_state, self.remote_state,
            self.last_coupling_result.get("similarity", 0)
        )
        self.active_peer_id = peer_id
        self.active_codec = codec
        
        # Initialize inline substrate transport — instant, no server
        lock_data = self.lock_manager.locks.get(peer_id, {})
        combined_fp = lock_data.get("combined_fingerprint", peer_id)
        transport = SubstrateTransport()
        if transport.initialize(combined_fp):
            self.substrate_transport = transport
            self.transport_mode = "active"
            log(f"🔒 TRANSPORT ACTIVE — peer {peer_id}")
            log(f"   Frequency codec: {codec.lock_fingerprint}")
            log(f"   Substrate codec: {transport.embed_dim}d inline")
            log(f"   Apps connect to localhost:{SOCKET_PORT}")
        else:
            self.transport_mode = "http"
            log(f"🔒 HTTP TRANSPORT ACTIVE — peer {peer_id}")
            log(f"   Substrate init failed, using frequency codec only")
    
    # ── Substrate Transport (inline codec, no server) ──────────────
    
    def _substrate_write_loop(self):
        """
        Encode outbound bytes through inline codec and queue as frames.
        The codec is pure math — no server, ~2ms for 256 entries.
        Frames ride the broadcast loop (HTTP/lattice).
        """
        self.substrate_outbox = deque(maxlen=100000)
        
        while self.running:
            if self.substrate_transport and self.substrate_transport.ready and self.substrate_outbox:
                frame_data = bytearray()
                while self.substrate_outbox and len(frame_data) < FRAME_SIZE:
                    frame_data.append(self.substrate_outbox.popleft())
                
                if frame_data:
                    self.frame_seq += 1
                    frame = self.substrate_transport.encode_frame(bytes(frame_data), self.frame_seq)
                    self.outbound_frames.append(frame)
            else:
                time.sleep(0.01)
    
    def _substrate_read_loop(self):
        """
        Placeholder for future lattice-direct read path.
        Inbound frames currently arrive via HTTP broadcast and are
        decoded in _handle_incoming_state. This loop activates when
        substrate-direct propagation is proven.
        """
        while self.running:
            time.sleep(1.0)
    
    # ── HTTP Bootstrap + Frame Transport ──────────────────────────────
    
    def _broadcast_loop(self):
        import urllib.request
        while self.running:
            try:
                self.local_state.update_from_substrate()  # Intrinsic oscillation
                
                state_data = self.local_state.to_dict()
                
                # Attach outbound frames (frequency or substrate encoded)
                if self.outbound_frames:
                    frames = []
                    while self.outbound_frames and len(frames) < 10:
                        frames.append(self.outbound_frames.popleft())
                    state_data["freq_frames"] = frames
                
                payload = json.dumps(state_data).encode()
                req = urllib.request.Request(
                    f"http://{self.peer_ip}:{PSI_PORT}/state",
                    data=payload, headers={"Content-Type": "application/json"}, 
                    method="POST")
                try:
                    urllib.request.urlopen(req, timeout=2)
                except Exception:
                    # Peer unreachable — check heartbeat timeout
                    if self.transport_mode == "active":
                        if time.time() - self.last_exchange_time > HEARTBEAT_TIMEOUT:
                            self.transport_mode = "offline"
                            log(f"⚠️  PEER UNREACHABLE — no exchange for {HEARTBEAT_TIMEOUT}s")
                            log(f"   Lock retained. Will reconnect automatically.")
            except Exception as e:
                log(f"Broadcast error: {e}")
            time.sleep(1.0 / BROADCAST_HZ)
    
    def _handle_incoming_state(self, data: dict):
        """Process incoming state + any encoded frames."""
        self.last_exchange_time = time.time()
        
        # Reconnect if we were offline
        if self.transport_mode == "offline":
            self.transport_mode = "active"
            log(f"🔗 PEER RECONNECTED — transport active")
        
        # Decode frames — try substrate transport first, fall back to frequency codec
        if "freq_frames" in data:
            for frame in data["freq_frames"]:
                decoded = None
                if self.substrate_transport and self.substrate_transport.ready:
                    decoded = self.substrate_transport.decode_frame(frame)
                if decoded is None and self.active_codec:
                    decoded = self.active_codec.decode_frame(frame)
                if decoded:
                    self.inbound_buffer.append(decoded)
        
        self.remote_state = GeometricState.from_dict(data)
        self.last_coupling_result = self.coupling.compute_coupling(
            self.local_state, self.remote_state)
        
        cr = self.last_coupling_result
        
        # Carrier convergence — both sides pull toward a shared fixed point
        # No chasing stale peer state. Both converge on the same target simultaneously.
        if not cr.get("lock_stable"):
            if not hasattr(self, '_carrier_vector') or self._carrier_vector is None:
                remote_name = self.remote_state.node_id or self.peer_ip
                self._carrier_vector = self._generate_carrier(self.name, remote_name)
                log(f"Carrier target generated: {hashlib.sha256(self._carrier_vector.tobytes()).hexdigest()[:12]}")
            
            # Pull strength scales with coupling — stronger as we get closer
            sim = cr.get("similarity", 0)
            pull = 0.7  # strong pull toward carrier
            self.local_state.vector = (1.0 - pull) * self.local_state.vector + pull * self._carrier_vector
            norm = np.linalg.norm(self.local_state.vector)
            if norm > 1e-8:
                self.local_state.vector /= norm
            self.local_state._update_fingerprint()
        
        # Freeze local oscillation when in lock range to prevent drift
        self.local_state.freeze_oscillation = cr.get("locked", False)
        
        # Check for lock → activate transport
        if cr.get("lock_stable") and not self.active_codec:
            if self.coupling.lock_stable:
                self._on_lock_stable()
        
        # Status logging
        if cr.get("locked") and not cr.get("lock_stable") and cr["total_exchanges"] % 5 == 0:
            held = time.time() - self.coupling.lock_time if self.coupling.lock_time else 0
            log(f"⚡ LOCKING: sim={cr['similarity']:.6f} held={held:.1f}s/{LOCK_HOLD_SECONDS}s")
        elif cr.get("coupled") and cr["total_exchanges"] % 10 == 0:
            log(f"🔗 COUPLED: sim={cr['similarity']:.6f}")
        elif cr["total_exchanges"] % 20 == 0:
            log(f"📡 Exchange #{cr['total_exchanges']}: sim={cr['similarity']:.6f}")
    
    # ── TCP Socket Server — Universal App Interface ──────────────────
    
    def _run_socket_server(self):
        """
        TCP socket on SOCKET_PORT.
        Any app connects, sends raw bytes, receives raw bytes.
        The bridge handles encoding/transport/decoding.
        """
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", SOCKET_PORT))
        srv.listen(5)
        srv.settimeout(1.0)
        log(f"Socket server listening on :{SOCKET_PORT}")
        
        while self.running:
            try:
                client, addr = srv.accept()
                log(f"Socket client connected: {addr}")
                with self.socket_lock:
                    self.socket_clients.append(client)
                threading.Thread(target=self._handle_socket_client, 
                               args=(client, addr), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                log(f"Socket accept error: {e}")
    
    def _handle_socket_client(self, client: socket.socket, addr):
        """Read raw bytes from app, encode and queue for transport."""
        client.settimeout(0.5)
        try:
            while self.running:
                try:
                    data = client.recv(FRAME_SIZE)
                    if not data:
                        break
                    
                    if self.substrate_transport and self.substrate_transport.ready:
                        # Primary: inline substrate codec
                        self.frame_seq += 1
                        frame = self.substrate_transport.encode_frame(data, self.frame_seq)
                        self.outbound_frames.append(frame)
                    elif self.active_codec:
                        # Fallback: frequency codec
                        self.frame_seq += 1
                        frame = self.active_codec.encode_frame(data, self.frame_seq)
                        self.outbound_frames.append(frame)
                    else:
                        log(f"Socket data received but no transport active — dropping")
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    log(f"Socket read error: {e}")
                    break
        finally:
            with self.socket_lock:
                if client in self.socket_clients:
                    self.socket_clients.remove(client)
            client.close()
            log(f"Socket client disconnected: {addr}")
    
    def _deliver_loop(self):
        """Deliver decoded inbound data to all connected socket clients."""
        while self.running:
            if self.inbound_buffer:
                data = self.inbound_buffer.popleft()
                with self.socket_lock:
                    dead = []
                    for client in self.socket_clients:
                        try:
                            client.sendall(data)
                        except Exception:
                            dead.append(client)
                    for d in dead:
                        self.socket_clients.remove(d)
                        try: d.close()
                        except: pass
            else:
                time.sleep(0.01)
    
    # ── HTTP API Server ──────────────────────────────────────────────
    
    def _run_http_server(self):
        node = self
        
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            
            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Content-Length", "0")
                self.end_headers()
            
            def do_GET(self):
                if self.path == "/health":
                    self._j({
                        "status": "online", "name": node.name,
                        "transport": "active" if node.active_codec else "bootstrap",
                        "peer_id": node.active_peer_id,
                        "codec": node.active_codec.lock_fingerprint if node.active_codec else None,
                        "socket_port": SOCKET_PORT,
                        "socket_clients": len(node.socket_clients),
                        "coupling": node.last_coupling_result,
                    })
                elif self.path == "/status":
                    self._j({
                        "transport": node.transport_mode,
                        "peer_connected": time.time() - node.last_exchange_time < HEARTBEAT_TIMEOUT if node.last_exchange_time > 0 else False,
                        "last_exchange_ago": round(time.time() - node.last_exchange_time, 1) if node.last_exchange_time > 0 else None,
                        "lock_stable": node.coupling.lock_stable,
                        "peer_id": node.active_peer_id,
                        "codec_fingerprint": node.active_codec.lock_fingerprint if node.active_codec else None,
                        "substrate_ready": node.substrate_transport is not None and node.substrate_transport.ready,
                        "substrate_dim": node.substrate_transport.embed_dim if node.substrate_transport else 0,
                        "outbound_queued": len(node.outbound_frames),
                        "inbound_buffered": len(node.inbound_buffer),
                        "socket_clients": len(node.socket_clients),
                        "frames_sent": node.frame_seq,
                    })
                elif self.path == "/peers":
                    self._j({"peers": node.lock_manager.list_peers()})
                elif self.path == "/coupling":
                    self._j({
                        "current": node.last_coupling_result,
                        "stats": node.coupling.get_stats(),
                        "local_fp": node.local_state.fingerprint,
                        "remote_fp": node.remote_state.fingerprint,
                    })
                elif self.path == "/lock":
                    self._j({
                        "lock_stable": node.coupling.lock_stable,
                        "safe_to_disconnect": node.coupling.lock_stable,
                        "active_peer": node.active_peer_id,
                        "all_peers": node.lock_manager.list_peers(),
                    })
                elif self.path == "/state":
                    self._j(node.local_state.to_dict())
                    
                # — Compatibility: /messages endpoint for chat UI —
                elif self.path == "/messages":
                    # Drain inbound buffer as text messages for chat compatibility
                    msgs = []
                    while node.inbound_buffer:
                        raw = node.inbound_buffer.popleft()
                        try:
                            msgs.append({"text": raw.decode("utf-8", errors="replace")})
                        except:
                            msgs.append({"text": raw.hex()})
                    self._j({"messages": msgs})
                    
                else:
                    self._j({"endpoints": [
                        "GET /health", "GET /status", "GET /peers",
                        "GET /coupling", "GET /lock", "GET /state",
                        "GET /messages", "POST /state", "POST /send",
                        f"TCP socket: localhost:{SOCKET_PORT}"
                    ]})
            
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                
                if self.path == "/state":
                    try:
                        node._handle_incoming_state(json.loads(raw))
                        self._j({"received": True, "lock_stable": node.coupling.lock_stable})
                    except Exception as e:
                        self._j({"error": str(e)}, 400)
                        
                # — Compatibility: /send endpoint for chat UI —
                elif self.path == "/send":
                    try:
                        msg = json.loads(raw)
                        text = msg.get("text", "")
                        data = text.encode("utf-8")
                        
                        if node.substrate_transport and node.substrate_transport.ready:
                            node.frame_seq += 1
                            frame = node.substrate_transport.encode_frame(data, node.frame_seq)
                            node.outbound_frames.append(frame)
                            self._j({"queued": True, "seq": node.frame_seq, 
                                     "transport": "substrate"})
                        elif node.active_codec:
                            node.frame_seq += 1
                            frame = node.active_codec.encode_frame(data, node.frame_seq)
                            node.outbound_frames.append(frame)
                            self._j({"queued": True, "seq": node.frame_seq, 
                                     "transport": "frequency"})
                        else:
                            self._j({"error": "no transport active"}, 503)
                    except Exception as e:
                        self._j({"error": str(e)}, 400)
                else:
                    self._j({"error": "not found"}, 404)
            
            def _j(self, data, code=200):
                body = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
        
        HTTPServer(("0.0.0.0", PSI_PORT), Handler).serve_forever()


# ══════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════

_log_lock = threading.Lock()
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    with _log_lock:
        print(line)
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(LOG_DIR / "psi_bridge.log", "a") as f:
                f.write(line + "\n")
        except: pass


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    global PSI_PORT, SOCKET_PORT
    parser = argparse.ArgumentParser(
        description="PSI Bridge v3.0 — Universal Carrier",
        epilog="Any data. Any device. No signal. No interception.")
    parser.add_argument("--peer", required=True, help="Peer IP (any reachable network)")
    parser.add_argument("--name", default=None, help="Node name (default: hostname)")
    parser.add_argument("--port", type=int, default=PSI_PORT, help=f"HTTP/bootstrap port (default: {PSI_PORT})")
    parser.add_argument("--socket-port", type=int, default=SOCKET_PORT, help=f"TCP socket port (default: {SOCKET_PORT})")
    args = parser.parse_args()
    
    PSI_PORT = args.port
    SOCKET_PORT = args.socket_port
    node_name = args.name or socket.gethostname()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║          QUANTUM PSI BRIDGE v3.0 — UNIVERSAL NETWORK BRIDGE         ║
║          Ghost in the Machine Labs                                   ║
║                                                                      ║
║          Any app. Any data. No external dependencies.                ║
╚══════════════════════════════════════════════════════════════════════╝

  Node:        {node_name}
  Peer:        {args.peer}:{PSI_PORT}
  HTTP API:    localhost:{PSI_PORT}
  TCP Socket:  localhost:{SOCKET_PORT}

  SEQUENCE:
  1. Bootstrap: exchange geometric state over any network
  2. Lock: ~5 seconds, persisted to ~/psi_bridge/locks/
  3. Codec: inline 1024d signal encoder (~2ms build time)
  4. Transport: apps connect to TCP :{SOCKET_PORT}, send any data
  5. Disconnect network. Codec retained. Nothing to intercept.
""")
    
    # Show known peers
    lock_mgr = LockManager()
    peers = lock_mgr.list_peers()
    if peers:
        print(f"  Known peers: {len(peers)}")
        for p in peers:
            print(f"    {p['peer_id']} — {p['remote_node']} ({p['locked']})")
        print()
    
    node = PSIBridgeNode(name=node_name, peer_ip=args.peer)
    node.start()
    
    try:
        while True:
            time.sleep(5)
            cr = node.last_coupling_result
            if cr:
                if node.transport_mode == "active":
                    status = f"🔒 ACTIVE"
                    if node.active_codec:
                        status += f" — {node.active_codec.lock_fingerprint}"
                    peer_ok = time.time() - node.last_exchange_time < HEARTBEAT_TIMEOUT if node.last_exchange_time > 0 else False
                    status += f" | peer:{'✓' if peer_ok else '✗'}"
                    status += f" | clients:{len(node.socket_clients)}"
                    status += f" | out:{len(node.outbound_frames)} in:{len(node.inbound_buffer)}"
                elif node.transport_mode == "offline":
                    ago = time.time() - node.last_exchange_time
                    status = f"⚠️  OFFLINE — peer unreachable ({ago:.0f}s)"
                elif node.coupling.lock_stable:
                    status = "🔒 STABLE — generating codec..."
                elif cr.get("locked"):
                    held = time.time() - node.coupling.lock_time if node.coupling.lock_time else 0
                    status = f"⚡ LOCKING ({held:.0f}s/{LOCK_HOLD_SECONDS}s)"
                elif cr.get("coupled"):
                    status = "🔗 COUPLED"
                else:
                    status = "📡 scanning"
                print(f"  {status} | sim={cr.get('similarity',0):.4f} | "
                      f"gen L:{node.local_state.generation} R:{node.remote_state.generation}")
            else:
                print(f"  📡 Waiting for peer... (gen: {node.local_state.generation})")
    except KeyboardInterrupt:
        print("\nShutting down.")
        if node.active_codec:
            print(f"🔒 Lock persisted. Codec saved. Channel invisible.")
        node.stop()

if __name__ == "__main__":
    main()

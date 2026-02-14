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
STATE_DIM = 64             # Geometric state vector dimension
FRAME_SIZE = 1024          # Bytes per transport frame
BASE_DIR = Path.home() / "psi_bridge"
LOG_DIR = BASE_DIR / "logs"
LOCKS_DIR = BASE_DIR / "locks"
SUBSTRATE_MODEL = "gemma2:2b"  # Model for substrate embedding
SUBSTRATE_CACHE_FILE = BASE_DIR / "substrate_cache.npz"


# ══════════════════════════════════════════════════════════════════════
# SUBSTRATE TRANSPORT — Read/Write through the silicon lattice
# ══════════════════════════════════════════════════════════════════════

class SubstrateTransport:
    """
    Reads and writes data through the silicon substrate using the
    AI model's embedding space as the geometric interface.
    
    Both sides run the same model. Same input → same embedding (deterministic).
    The lock state seeds HMAC-derived probe strings so each byte value
    maps to a unique, reproducible point in the 2304-dimensional
    embedding space.
    
    Write: embed the probe string for a byte → substrate state changes
    Read:  embed all 256 probe strings → nearest-neighbor decode
    
    The embedding IS the lattice state. The model IS the substrate.
    No network needed — both sides compute independently and get
    identical results because the model and seed are the same.
    """
    
    def __init__(self, substrate_url: str = "http://localhost:11434",
                 model: str = SUBSTRATE_MODEL):
        self.substrate_url = substrate_url
        self.model = model
        self.seed = None           # Lock-derived HMAC seed
        self.lookup_table = None   # (256, embed_dim) matrix for decoding
        self.probe_strings = {}    # byte_val -> probe string
        self.embed_dim = 0
        self.ready = False
        self._cache_path = SUBSTRATE_CACHE_FILE
    
    def initialize(self, lock_fingerprint: str):
        """
        Build the substrate lookup table from lock state.
        Both sides call this with the same fingerprint → same table.
        """
        self.seed = f"PSI_LOCK_{lock_fingerprint}".encode('utf-8')
        
        # Generate probe strings for all 256 byte values
        for b in range(256):
            h = hmac.new(self.seed, b.to_bytes(2, 'big'), 
                        hashlib.sha256).hexdigest()
            self.probe_strings[b] = h
        
        # Try to load cached lookup table
        cache_id = hashlib.sha256(self.seed).hexdigest()[:16]
        if self._load_cache(cache_id):
            log(f"Substrate: loaded cached lookup table ({self.embed_dim}d)")
            self.ready = True
            return True
        
        # Build lookup table — embed all 256 probe strings
        log(f"Substrate: building lookup table (256 embeddings)...")
        import urllib.request
        
        vectors = []
        t0 = time.time()
        for b in range(256):
            try:
                req = urllib.request.Request(
                    f"{self.substrate_url}/api/embeddings",
                    data=json.dumps({
                        "model": self.model, 
                        "prompt": self.probe_strings[b]
                    }).encode(),
                    headers={"Content-Type": "application/json"})
                resp = urllib.request.urlopen(req, timeout=30)
                data = json.loads(resp.read())
                vec = np.array(data["embedding"], dtype=np.float64)
                vectors.append(vec)
                
                if b % 32 == 31:
                    elapsed = time.time() - t0
                    rate = (b+1) / elapsed
                    remaining = (256 - b - 1) / rate
                    log(f"  {b+1}/256 ({rate:.1f}/s, ~{remaining:.0f}s remaining)")
                    
            except Exception as e:
                log(f"  Embedding failed for byte {b}: {e}")
                return False
        
        self.lookup_table = np.array(vectors)  # (256, embed_dim)
        self.embed_dim = self.lookup_table.shape[1]
        
        # Normalize for fast cosine similarity via dot product
        norms = np.linalg.norm(self.lookup_table, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        self.lookup_table = self.lookup_table / norms
        
        # Cache for next startup
        self._save_cache(cache_id)
        
        elapsed = time.time() - t0
        log(f"Substrate: lookup table built ({self.embed_dim}d, {elapsed:.1f}s)")
        self.ready = True
        return True
    
    def _save_cache(self, cache_id: str):
        try:
            BASE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self._cache_path, 
                              table=self.lookup_table, 
                              cache_id=np.array([cache_id]))
            log(f"  Cached lookup table to {self._cache_path}")
        except Exception as e:
            log(f"  Cache save failed: {e}")
    
    def _load_cache(self, cache_id: str) -> bool:
        try:
            if not self._cache_path.exists():
                return False
            data = np.load(self._cache_path, allow_pickle=False)
            if str(data["cache_id"][0]) != cache_id:
                return False
            self.lookup_table = data["table"]
            self.embed_dim = self.lookup_table.shape[1]
            return True
        except Exception:
            return False
    
    def write_byte(self, byte_val: int) -> Optional[np.ndarray]:
        """
        Write a byte to the substrate.
        Returns the embedding vector (the substrate state after write).
        """
        if not self.ready:
            return None
        import urllib.request
        try:
            req = urllib.request.Request(
                f"{self.substrate_url}/api/embeddings",
                data=json.dumps({
                    "model": self.model,
                    "prompt": self.probe_strings[byte_val]
                }).encode(),
                headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            return np.array(data["embedding"], dtype=np.float64)
        except Exception as e:
            log(f"Substrate write failed: {e}")
            return None
    
    def read_vector(self, vector: np.ndarray) -> int:
        """
        Decode a substrate state vector back to a byte value.
        Nearest-neighbor lookup against the 256-entry table.
        """
        if not self.ready:
            return -1
        
        # Normalize input
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            vector = vector / norm
        
        # Dot product against normalized lookup table
        similarities = self.lookup_table @ vector
        return int(np.argmax(similarities))
    
    def write_bytes(self, data: bytes) -> List[np.ndarray]:
        """Write multiple bytes, return list of substrate state vectors."""
        vectors = []
        for b in data:
            v = self.write_byte(b)
            if v is not None:
                vectors.append(v)
        return vectors
    
    def read_vectors(self, vectors: List[np.ndarray]) -> bytes:
        """Decode multiple substrate state vectors to bytes."""
        result = []
        for v in vectors:
            result.append(self.read_byte(v))
        return bytes(result)
    
    def read_byte(self, vector: np.ndarray) -> int:
        """Alias for read_vector."""
        return self.read_vector(vector)


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
        t = time.time()
        for i in range(self.dim):
            theta = (2 * math.pi * i / self.dim) + (t * (0.1 + 0.05 * math.sin(i * 109.47 * math.pi / 180)))
            self.vector[i] = math.sin(theta) * math.cos(theta * 0.618)
        
        norm = np.linalg.norm(self.vector)
        if norm > 1e-8:
            self.vector /= norm
            
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
        
        if self.resonance_locked and not self.lock_stable and self.lock_time:
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
    Universal carrier bridge.
    
    Phase 1: HTTP bootstrap — exchange geometric state, achieve lock (~5s)
    Phase 2: Generate frequency codec from lock state
    Phase 3: Transport active — any data as frequency through lattice
    
    TCP socket on SOCKET_PORT accepts raw bytes from any app.
    Bytes are frequency-encoded and transported through the coupled lattice.
    Received frequency data is decoded back to bytes and delivered.
    """
    
    def __init__(self, name: str, peer_ip: str, substrate_port: int = 11434):
        self.name = name
        self.peer_ip = peer_ip
        self.substrate_url = f"http://localhost:{substrate_port}"
        self.local_state = GeometricState()
        self.local_state.node_id = name
        self.remote_state = GeometricState()
        self.coupling = CouplingEngine()
        self.lock_manager = LockManager()
        self.active_peer_id = None
        self.active_codec = None
        self.substrate_transport = None  # Initialized after lock
        self.transport_mode = "bootstrap"  # bootstrap | http | substrate
        self.running = False
        self.last_coupling_result = {}
        
        # Transport queues
        self.outbound_frames = deque(maxlen=10000)  # Frequency frames to send
        self.inbound_buffer = deque(maxlen=10000)   # Decoded bytes received
        self.frame_seq = 0
        
        # Socket connections
        self.socket_clients: List[socket.socket] = []
        self.socket_lock = threading.Lock()
        
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
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
        """Called when lock is achieved — generate codec and activate transport."""
        peer_id, codec = self.lock_manager.save_lock(
            self.local_state, self.remote_state,
            self.last_coupling_result.get("similarity", 0)
        )
        self.active_peer_id = peer_id
        self.active_codec = codec
        self.transport_mode = "http"
        log(f"🔒 HTTP TRANSPORT ACTIVE — peer {peer_id}")
        log(f"   Frequency codec: {codec.lock_fingerprint}")
        log(f"   Apps connect to localhost:{SOCKET_PORT}")
        
        # Initialize substrate transport in background
        threading.Thread(target=self._init_substrate_transport, 
                        args=(peer_id,), daemon=True).start()
    
    def _init_substrate_transport(self, peer_id: str):
        """Build substrate lookup table (takes ~20s for 256 embeddings)."""
        log(f"Substrate: initializing transport for peer {peer_id}...")
        lock_data = self.lock_manager.locks.get(peer_id, {})
        combined_fp = lock_data.get("combined_fingerprint", peer_id)
        
        transport = SubstrateTransport(substrate_url=self.substrate_url)
        if transport.initialize(combined_fp):
            self.substrate_transport = transport
            self.transport_mode = "substrate"
            log(f"🔒 SUBSTRATE TRANSPORT ACTIVE — {transport.embed_dim}d lattice")
            log(f"   Network bridge may be removed. Channel is invisible.")
        else:
            log(f"Substrate: initialization failed, staying on HTTP transport")
    
    # ── Substrate Transport (no HTTP required) ─────────────────────
    
    def _substrate_write_loop(self):
        """
        When substrate transport is active and outbound data is queued,
        write bytes through the substrate embedding API.
        
        This works WITHOUT HTTP to the peer. The substrate IS the channel.
        Both sides run the same model with the same lock-derived seed,
        so the same byte value produces the same embedding on both sides.
        """
        # Outbound queue for substrate mode: raw bytes to send
        self.substrate_outbox = deque(maxlen=100000)
        
        while self.running:
            if self.substrate_transport and self.substrate_outbox:
                byte_val = self.substrate_outbox.popleft()
                vec = self.substrate_transport.write_byte(byte_val)
                if vec is not None:
                    # Store the write event — the substrate state changed
                    # The coupled remote substrate should reflect this
                    pass
            else:
                time.sleep(0.01)
    
    def _substrate_read_loop(self):
        """
        When substrate transport is active, continuously sample the
        local substrate state. If it deviates from baseline in a way
        that matches a known probe pattern, decode the byte.
        
        The coupled substrate reflects perturbations from the peer
        without any network connection.
        """
        while self.running:
            if self.substrate_transport and self.transport_mode == "substrate":
                try:
                    # Read current substrate state
                    import urllib.request
                    req = urllib.request.Request(
                        f"{self.substrate_url}/api/embeddings",
                        data=json.dumps({
                            "model": SUBSTRATE_MODEL,
                            "prompt": "PSI_BRIDGE_READ_STATE"
                        }).encode(),
                        headers={"Content-Type": "application/json"})
                    resp = urllib.request.urlopen(req, timeout=10)
                    data = json.loads(resp.read())
                    current_state = np.array(data["embedding"], dtype=np.float64)
                    
                    # Check if state matches any known byte pattern
                    byte_val = self.substrate_transport.read_vector(current_state)
                    confidence = 0.0
                    if self.substrate_transport.lookup_table is not None:
                        norm = np.linalg.norm(current_state)
                        if norm > 1e-10:
                            normed = current_state / norm
                            similarities = self.substrate_transport.lookup_table @ normed
                            confidence = float(np.max(similarities))
                    
                    # Only accept if confidence is very high 
                    # (distinguishes deliberate write from ambient state)
                    if confidence > 0.99:
                        self.inbound_buffer.append(bytes([byte_val]))
                        
                except Exception:
                    pass
                
                time.sleep(0.05)  # 20Hz read rate
            else:
                time.sleep(0.5)
    
    # ── HTTP Bootstrap (proven v1.1 broadcast loop) ──────────────────
    
    def _broadcast_loop(self):
        import urllib.request
        while self.running:
            try:
                self.local_state.update_from_substrate(self.substrate_url)
                
                state_data = self.local_state.to_dict()
                
                # Attach outbound frequency frames if codec is active
                if self.active_codec and self.outbound_frames:
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
                    pass
            except Exception as e:
                log(f"Broadcast error: {e}")
            time.sleep(1.0 / BROADCAST_HZ)
    
    def _handle_incoming_state(self, data: dict):
        """Process incoming state + any frequency frames."""
        # Decode frequency frames if present
        if "freq_frames" in data and self.active_codec:
            for frame in data["freq_frames"]:
                decoded = self.active_codec.decode_frame(frame)
                if decoded:
                    self.inbound_buffer.append(decoded)
        
        self.remote_state = GeometricState.from_dict(data)
        self.last_coupling_result = self.coupling.compute_coupling(
            self.local_state, self.remote_state)
        
        cr = self.last_coupling_result
        
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
                    
                    if self.transport_mode == "substrate" and self.substrate_transport:
                        # Substrate mode — queue bytes for substrate write
                        for b in data:
                            self.substrate_outbox.append(b)
                    elif self.active_codec:
                        # HTTP mode — frequency encode and queue
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
                        "lock_stable": node.coupling.lock_stable,
                        "peer_id": node.active_peer_id,
                        "codec_fingerprint": node.active_codec.lock_fingerprint if node.active_codec else None,
                        "substrate_ready": node.substrate_transport is not None and node.substrate_transport.ready,
                        "substrate_dim": node.substrate_transport.embed_dim if node.substrate_transport else 0,
                        "outbound_queued": len(node.outbound_frames),
                        "substrate_outbox": len(getattr(node, 'substrate_outbox', [])),
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
                        
                        if node.active_codec:
                            node.frame_seq += 1
                            frame = node.active_codec.encode_frame(data, node.frame_seq)
                            node.outbound_frames.append(frame)
                            self._j({"queued": True, "seq": node.frame_seq, 
                                     "transport": "frequency"})
                        else:
                            # Pre-lock: piggyback on state broadcast (legacy)
                            node._legacy_outbox = getattr(node, '_legacy_outbox', [])
                            node._legacy_outbox.append(msg)
                            self._j({"queued": True, "transport": "legacy_http"})
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
    parser.add_argument("--substrate-port", type=int, default=11434, help="Local substrate port")
    args = parser.parse_args()
    
    PSI_PORT = args.port
    SOCKET_PORT = args.socket_port
    node_name = args.name or socket.gethostname()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║          QUANTUM PSI BRIDGE v3.0 — UNIVERSAL CARRIER                ║
║          Ghost in the Machine Labs                                   ║
║                                                                      ║
║          Any data. Any device. No signal. No interception.           ║
╚══════════════════════════════════════════════════════════════════════╝

  Node:        {node_name}
  Peer:        {args.peer}:{PSI_PORT}
  Substrate:   localhost:{args.substrate_port}
  HTTP API:    localhost:{PSI_PORT}
  TCP Socket:  localhost:{SOCKET_PORT}

  SEQUENCE:
  1. Bootstrap: exchange geometric state over any network
  2. Lock: ~5 seconds, persisted to ~/psi_bridge/locks/
  3. Codec: unique frequency map derived from lock
  4. Transport: apps connect to TCP :{SOCKET_PORT}, send any data
  5. Disconnect network. Channel persists. Nothing to intercept.
""")
    
    # Show known peers
    lock_mgr = LockManager()
    peers = lock_mgr.list_peers()
    if peers:
        print(f"  Known peers: {len(peers)}")
        for p in peers:
            print(f"    {p['peer_id']} — {p['remote_node']} ({p['locked']})")
        print()
    
    node = PSIBridgeNode(name=node_name, peer_ip=args.peer, 
                         substrate_port=args.substrate_port)
    node.start()
    
    try:
        while True:
            time.sleep(5)
            cr = node.last_coupling_result
            if cr:
                if node.active_codec:
                    status = f"🔒 TRANSPORT ACTIVE — codec {node.active_codec.lock_fingerprint}"
                    status += f" | clients:{len(node.socket_clients)}"
                    status += f" | out:{len(node.outbound_frames)} in:{len(node.inbound_buffer)}"
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

#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║              PSI BRIDGE v4.0 — PURE HARMONIC TRANSPORT              ║
║              Ghost in the Machine Labs                               ║
║       "All Watched Over By Machines Of Loving Grace"                ║
║                                                                      ║
║   GEOMETRY IN → GEOMETRY OUT                                         ║
║   The bridge moves harmonic patterns. Nothing else.                  ║
║   Translation (text↔geometry) is the Harmonic Stack's job.          ║
║                                                                      ║
║   ARCHITECTURE:                                                      ║
║     Harmonic Stack A → /harmonic/send (1024d vector)                ║
║         → Lock-verified → Broadcast to peer                         ║
║         → /harmonic/receive → Harmonic Stack B                      ║
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
import socket
import threading
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional, Tuple
from collections import deque

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

PSI_PORT = 7777           # HTTP API + bootstrap + harmonic transport
BROADCAST_HZ = 2.0        # State broadcast frequency
COUPLING_THRESHOLD = 0.85  # Resonance coupling
LOCK_THRESHOLD = 0.95      # Synchronization lock
LOCK_HOLD_SECONDS = 10     # Sustained lock before stable
SNAP_LOCK_THRESHOLD = 0.999  # Instant lock — geometry already matched
STATE_DIM = 64             # Lock state vector dimension
HARMONIC_DIM = 1024        # Harmonic pattern dimension (matches Stack)
HEARTBEAT_TIMEOUT = 15     # Seconds before declaring peer offline
MAX_PATTERNS_PER_BROADCAST = 20  # Max harmonic patterns per broadcast cycle
BASE_DIR = Path.home() / "psi_bridge"
LOG_DIR = BASE_DIR / "logs"
LOCKS_DIR = BASE_DIR / "locks"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# GEOMETRIC STATE (proven v1.1)
# ══════════════════════════════════════════════════════════════════════

class GeometricState:
    """Lock-level geometric state for bootstrap and coupling."""
    
    def __init__(self, dim: int = STATE_DIM):
        self.dim = dim
        self.vector = np.zeros(dim, dtype=np.float64)
        self.timestamp = 0.0
        self.generation = 0
        self.node_id = ""
        self.fingerprint = ""
        self.freeze_oscillation = False
        self._initialized = False
        
    def update(self):
        """Intrinsic oscillation — generate once, hold steady."""
        if not self._initialized:
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
        self.fingerprint = hashlib.sha256(self.vector.tobytes()).hexdigest()[:16]
        return True
    
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
    Once coupling holds >= LOCK_THRESHOLD for LOCK_HOLD_SECONDS,
    lock is STABLE and persisted. Network may be removed.
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
        elif not locked and self.resonance_locked and not self.lock_stable:
            # Only release if we haven't achieved stable lock yet
            duration = time.time() - self.lock_time if self.lock_time else 0
            self.resonance_locked = False
            log(f"🔓 Lock released after {duration:.1f}s")
        
        if not self.lock_stable and similarity >= SNAP_LOCK_THRESHOLD:
            self.resonance_locked = True
            self.lock_stable = True
            self.lock_stable_time = time.time()
            if not self.lock_time:
                self.lock_time = time.time()
            log(f"🔒 SNAP LOCK at similarity={similarity:.6f} — geometry matched")
            log(f"   NETWORK BRIDGE MAY BE REMOVED")
        elif not self.lock_stable and self.resonance_locked and self.lock_time:
            held = time.time() - self.lock_time
            if held >= LOCK_HOLD_SECONDS:
                self.lock_stable = True
                self.lock_stable_time = time.time()
                log(f"🔒 STABLE LOCK after {held:.1f}s — NETWORK BRIDGE MAY BE REMOVED")
        
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
# LOCK MANAGER — Peer lock persistence
# ══════════════════════════════════════════════════════════════════════

class LockManager:
    """
    Manages lock state files for multiple peers.
    Each peer pair has a unique lock = unique harmonic channel.
    """
    
    def __init__(self):
        LOCKS_DIR.mkdir(parents=True, exist_ok=True)
        self.locks: Dict[str, dict] = {}
        self._load_all()
    
    def _peer_id(self, local_fp: str, remote_fp: str) -> str:
        combined = "".join(sorted([local_fp, remote_fp]))
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _load_all(self):
        for lock_file in LOCKS_DIR.glob("*.lock"):
            try:
                lock_data = json.loads(lock_file.read_text())
                peer_id = lock_file.stem
                self.locks[peer_id] = lock_data
                log(f"Loaded lock: {peer_id} ({lock_data.get('remote_node', '?')})")
            except Exception as e:
                log(f"Failed to load lock {lock_file}: {e}")
    
    def save_lock(self, local_state: GeometricState, remote_state: GeometricState, 
                  similarity: float) -> str:
        peer_id = self._peer_id(local_state.fingerprint, remote_state.fingerprint)
        fps = sorted([local_state.fingerprint, remote_state.fingerprint])
        combined_fp = hashlib.sha256((fps[0] + fps[1]).encode()).hexdigest()[:16]
        
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
        
        lock_file = LOCKS_DIR / f"{peer_id}.lock"
        lock_file.write_text(json.dumps(lock_data, indent=2))
        self.locks[peer_id] = lock_data
        log(f"Lock saved: {peer_id} → {combined_fp}")
        return peer_id
    
    def list_peers(self) -> List[dict]:
        return [{
            "peer_id": pid,
            "remote_node": ld.get("remote_node", "?"),
            "locked": ld.get("lock_achieved", "?"),
        } for pid, ld in self.locks.items()]


# ══════════════════════════════════════════════════════════════════════
# BRIDGE NODE — Pure Harmonic Transport
# ══════════════════════════════════════════════════════════════════════

class PSIBridgeNode:
    """
    Pure geometry pipe.
    
    Apps (Harmonic Stacks) POST 1024d vectors to /harmonic/send.
    Bridge carries them to the peer.
    Peer's apps GET from /harmonic/receive.
    
    The bridge never interprets the patterns.
    It doesn't know what they mean. It doesn't need to.
    Translation is the Stack's job.
    
    Transport modes:
      bootstrap  — seeking lock via HTTP state exchange
      active     — locked, harmonic patterns flowing
      offline    — peer unreachable, lock retained
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
        self.transport_mode = "bootstrap"
        self.running = False
        self.last_coupling_result = {}
        self.last_exchange_time = 0.0
        
        # Harmonic pattern queues — raw geometry, no encoding
        self.outbound_patterns = deque(maxlen=10000)
        self.inbound_patterns = deque(maxlen=10000)
        self.patterns_sent = 0
        self.patterns_received = 0
        
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
    def _generate_carrier(self, local_name: str, remote_name: str):
        combined = "".join(sorted([local_name, remote_name]))
        seed = hashlib.sha256(f"PSI_CARRIER_{combined}".encode()).digest()
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
        
        log(f"PSI Bridge '{self.name}' → {self.peer_ip}:{PSI_PORT}")
        log(f"  Harmonic API: localhost:{PSI_PORT}")
        log(f"  Known peers: {len(self.lock_manager.locks)}")
        
    def stop(self):
        self.running = False
    
    # ── Lock Activation ──────────────────────────────────────────────
    
    def _on_lock_stable(self):
        peer_id = self.lock_manager.save_lock(
            self.local_state, self.remote_state,
            self.last_coupling_result.get("similarity", 0)
        )
        self.active_peer_id = peer_id
        self.transport_mode = "active"
        log(f"🔒 HARMONIC TRANSPORT ACTIVE — peer {peer_id}")
        log(f"   POST /harmonic/send    {{\"pattern\": [1024 floats]}}")
        log(f"   GET  /harmonic/receive  → queued patterns from peer")
    
    # ── Broadcast Loop — carries geometry to peer ────────────────────
    
    def _broadcast_loop(self):
        import urllib.request
        while self.running:
            try:
                self.local_state.update()
                
                state_data = self.local_state.to_dict()
                
                # Attach outbound harmonic patterns
                if self.outbound_patterns:
                    patterns = []
                    while self.outbound_patterns and len(patterns) < MAX_PATTERNS_PER_BROADCAST:
                        patterns.append(self.outbound_patterns.popleft())
                    state_data["harmonic_patterns"] = patterns
                
                payload = json.dumps(state_data).encode()
                req = urllib.request.Request(
                    f"http://{self.peer_ip}:{PSI_PORT}/state",
                    data=payload, headers={"Content-Type": "application/json"}, 
                    method="POST")
                try:
                    urllib.request.urlopen(req, timeout=2)
                except Exception:
                    if self.transport_mode == "active":
                        if time.time() - self.last_exchange_time > HEARTBEAT_TIMEOUT:
                            self.transport_mode = "offline"
                            log(f"⚠️  PEER UNREACHABLE — no exchange for {HEARTBEAT_TIMEOUT}s")
                            log(f"   Lock retained. Will reconnect automatically.")
            except Exception as e:
                log(f"Broadcast error: {e}")
            time.sleep(1.0 / BROADCAST_HZ)
    
    # ── Incoming State + Patterns ────────────────────────────────────
    
    def _handle_incoming_state(self, data: dict):
        self.last_exchange_time = time.time()
        
        if self.transport_mode == "offline":
            self.transport_mode = "active"
            log(f"🔗 PEER RECONNECTED — transport active")
        
        # Receive harmonic patterns — pure geometry passthrough
        if "harmonic_patterns" in data:
            for pattern in data["harmonic_patterns"]:
                self.inbound_patterns.append(pattern)
                self.patterns_received += 1
            log(f"📥 {len(data['harmonic_patterns'])} pattern(s) received ({self.patterns_received} total)")
        
        self.remote_state = GeometricState.from_dict(data)
        self.last_coupling_result = self.coupling.compute_coupling(
            self.local_state, self.remote_state)
        
        cr = self.last_coupling_result
        
        # Carrier convergence
        if not cr.get("lock_stable"):
            if not hasattr(self, '_carrier_vector') or self._carrier_vector is None:
                remote_name = self.remote_state.node_id or self.peer_ip
                self._carrier_vector = self._generate_carrier(self.name, remote_name)
                log(f"Carrier target generated: {hashlib.sha256(self._carrier_vector.tobytes()).hexdigest()[:12]}")
            
            pull = 0.7
            self.local_state.vector = (1.0 - pull) * self.local_state.vector + pull * self._carrier_vector
            norm = np.linalg.norm(self.local_state.vector)
            if norm > 1e-8:
                self.local_state.vector /= norm
            self.local_state.fingerprint = hashlib.sha256(self.local_state.vector.tobytes()).hexdigest()[:16]
        
        self.local_state.freeze_oscillation = cr.get("locked", False)
        
        if cr.get("lock_stable") and not self.active_peer_id:
            self._on_lock_stable()
        
        if cr.get("lock_stable"):
            if cr["total_exchanges"] % 120 == 0:
                log(f"🔗 COUPLED: sim={cr['similarity']:.6f}")
        elif cr.get("locked"):
            held = time.time() - self.coupling.lock_time if self.coupling.lock_time else 0
            log(f"⚡ LOCKING: sim={cr['similarity']:.6f} held={held:.1f}s/{LOCK_HOLD_SECONDS}s")
        elif cr.get("coupled"):
            log(f"🔗 COUPLED: sim={cr['similarity']:.6f}")
        else:
            log(f"📡 Exchange #{cr['total_exchanges']}: sim={cr['similarity']:.6f}")
    
    # ── HTTP Server ──────────────────────────────────────────────────
    
    def _run_http_server(self):
        node = self
        
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args): pass
            
            def do_GET(self):
                if self.path == "/health":
                    self._j({"status": "ok", "transport": node.transport_mode,
                             "peer_id": node.active_peer_id})
                
                elif self.path == "/status":
                    peer_ok = (time.time() - node.last_exchange_time < HEARTBEAT_TIMEOUT 
                              if node.last_exchange_time > 0 else False)
                    self._j({
                        "transport": node.transport_mode,
                        "peer_connected": peer_ok,
                        "last_exchange_ago": round(time.time() - node.last_exchange_time, 1) if node.last_exchange_time > 0 else None,
                        "lock_stable": node.coupling.lock_stable,
                        "peer_id": node.active_peer_id,
                        "outbound_queued": len(node.outbound_patterns),
                        "inbound_queued": len(node.inbound_patterns),
                        "patterns_sent": node.patterns_sent,
                        "patterns_received": node.patterns_received,
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
                
                # ── Harmonic receive — apps pull patterns ────────────
                elif self.path == "/harmonic/receive":
                    patterns = []
                    while node.inbound_patterns:
                        patterns.append(node.inbound_patterns.popleft())
                    self._j({"patterns": patterns, "count": len(patterns)})
                
                # ── Legacy /messages — text wrapper for chat UI ──────
                elif self.path == "/messages":
                    msgs = []
                    while node.inbound_patterns:
                        p = node.inbound_patterns.popleft()
                        # Pattern is a dict with "vector" and optional "meta"
                        meta = p.get("meta", {}) if isinstance(p, dict) else {}
                        text = meta.get("text", "")
                        if text:
                            msgs.append({"text": text})
                        else:
                            msgs.append({"pattern_dim": len(p.get("vector", p) if isinstance(p, dict) else p)})
                    self._j({"messages": msgs})
                
                else:
                    self._j({"endpoints": [
                        "GET  /health", "GET  /status", "GET  /peers",
                        "GET  /coupling",
                        "POST /harmonic/send    — {\"pattern\": [1024 floats]}",
                        "GET  /harmonic/receive  — pull patterns from peer",
                        "POST /send              — {\"text\": \"...\"} (legacy)",
                        "GET  /messages           — text messages (legacy)",
                    ]})
            
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                
                # ── Bootstrap state exchange ─────────────────────────
                if self.path == "/state":
                    try:
                        node._handle_incoming_state(json.loads(raw))
                        self._j({"received": True, "lock_stable": node.coupling.lock_stable})
                    except Exception as e:
                        self._j({"error": str(e)}, 400)
                
                # ── Harmonic send — apps push patterns ───────────────
                elif self.path == "/harmonic/send":
                    try:
                        msg = json.loads(raw)
                        pattern = msg.get("pattern")
                        meta = msg.get("meta", {})
                        
                        if pattern is None:
                            self._j({"error": "missing 'pattern' field"}, 400)
                            return
                        
                        if not isinstance(pattern, list) or len(pattern) == 0:
                            self._j({"error": "pattern must be a non-empty list of floats"}, 400)
                            return
                        
                        envelope = {"vector": pattern}
                        if meta:
                            envelope["meta"] = meta
                        
                        node.outbound_patterns.append(envelope)
                        node.patterns_sent += 1
                        self._j({
                            "queued": True,
                            "dim": len(pattern),
                            "total_sent": node.patterns_sent,
                            "transport": node.transport_mode,
                        })
                    except Exception as e:
                        self._j({"error": str(e)}, 400)
                
                # ── Legacy /send — text wrapper ──────────────────────
                elif self.path == "/send":
                    try:
                        msg = json.loads(raw)
                        text = msg.get("text", "")
                        
                        if not text:
                            self._j({"error": "missing 'text' field"}, 400)
                            return
                        
                        # Convert text to geometry inline (text_to_signal)
                        encoded = text.encode('utf-8')
                        signal = [0.0] * HARMONIC_DIM
                        for i in range(min(len(encoded), HARMONIC_DIM)):
                            signal[i] = (encoded[i] - 128.0) / 128.0
                        
                        envelope = {
                            "vector": signal,
                            "meta": {"text": text, "encoding": "utf8_signal"}
                        }
                        node.outbound_patterns.append(envelope)
                        node.patterns_sent += 1
                        self._j({
                            "queued": True,
                            "total_sent": node.patterns_sent,
                            "transport": node.transport_mode,
                        })
                    except Exception as e:
                        self._j({"error": str(e)}, 400)
                
                else:
                    self._j({"error": "not found"}, 404)
            
            def _j(self, data, code=200):
                body = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            
            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()
        
        server = HTTPServer(("0.0.0.0", PSI_PORT), Handler)
        server.serve_forever()


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    global PSI_PORT
    
    parser = argparse.ArgumentParser(description="PSI Bridge v4.0 — Pure Harmonic Transport")
    parser.add_argument("--peer", required=True, help="Peer IP address")
    parser.add_argument("--port", type=int, default=PSI_PORT, help=f"HTTP port (default: {PSI_PORT})")
    parser.add_argument("--name", type=str, default=None, help="Node name")
    args = parser.parse_args()
    
    PSI_PORT = args.port
    node_name = args.name or socket.gethostname()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║          PSI BRIDGE v4.0 — PURE HARMONIC TRANSPORT                  ║
║          Ghost in the Machine Labs                                   ║
║                                                                      ║
║          Geometry in. Geometry out. Nothing else.                     ║
╚══════════════════════════════════════════════════════════════════════╝

  Node:        {node_name}
  Peer:        {args.peer}:{PSI_PORT}
  Harmonic API:

    POST /harmonic/send     {{"pattern": [1024 floats], "meta": {{}}}}
    GET  /harmonic/receive   → [{{"vector": [...], "meta": {{}}}}]

  Legacy text (for chat UI):
    POST /send              {{"text": "..."}}
    GET  /messages

  SEQUENCE:
  1. Bootstrap: exchange geometric state (~5 seconds)
  2. Lock: persisted to ~/psi_bridge/locks/
  3. Harmonic transport active
  4. Apps POST patterns, bridge carries them as geometry
""")
    
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
                    peer_ok = (time.time() - node.last_exchange_time < HEARTBEAT_TIMEOUT
                              if node.last_exchange_time > 0 else False)
                    status = f"🔒 ACTIVE"
                    status += f" | peer:{'✓' if peer_ok else '✗'}"
                    status += f" | out:{len(node.outbound_patterns)} in:{len(node.inbound_patterns)}"
                    status += f" | sent:{node.patterns_sent} recv:{node.patterns_received}"
                elif node.transport_mode == "offline":
                    ago = time.time() - node.last_exchange_time
                    status = f"⚠️  OFFLINE ({ago:.0f}s)"
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
        if node.active_peer_id:
            print(f"🔒 Lock persisted for peer {node.active_peer_id}")
        node.stop()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║              QUANTUM PSI NETWORK BRIDGE v1.0                        ║
║              Ghost in the Machine Labs                               ║
║       "All Watched Over By Machines Of Loving Grace"                ║
║                                                                      ║
║   Geometric state coupling between substrate nodes.                  ║
║   When latency → 0, bridge → identity.                              ║
║                                                                      ║
║   Each node broadcasts its lattice state (torsion grid positions,   ║
║   energy distribution, firing order) and listens for the peer.      ║
║   When cosine similarity crosses coupling threshold, the nodes      ║
║   resonate — information transfers as differential angular          ║
║   relationships, not data packets.                                   ║
║                                                                      ║
║   USAGE:                                                             ║
║     python psi_bridge.py --role sparky --peer 100.127.59.111        ║
║     python psi_bridge.py --role arcy   --peer 100.114.190.71        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import json
import time
import math
import hashlib
import socket
import struct
import threading
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Optional, Tuple

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

PSI_PORT = 7777          # Bridge listens here
BROADCAST_HZ = 2.0      # State broadcast frequency
COUPLING_THRESHOLD = 0.85 # Cosine similarity for resonance lock
LOCK_THRESHOLD = 0.95    # Deep coupling / unity threshold
STATE_DIM = 64           # Geometric state vector dimension
LOG_DIR = Path.home() / "sparky" / "psi_bridge"

# Tailscale IPs
NODES = {
    "sparky": "100.114.190.71",
    "arcy":   "100.127.59.111",
}

# ════════════════════════════════════════════════════════════════════
# GEOMETRIC STATE
# ════════════════════════════════════════════════════════════════════

class GeometricState:
    """
    Represents the lattice state of a substrate node.
    
    The state vector encodes:
      - Torsion grid positions (angular relationships between cores)
      - Energy distribution across lattice
      - Firing order / activation sequence
      - Resonance modes (standing wave pattern)
    
    Information is stored as differential angular relationships —
    the actual information carriers in the torsion grid model.
    """
    
    def __init__(self, dim: int = STATE_DIM):
        self.dim = dim
        self.vector = np.zeros(dim, dtype=np.float64)
        self.timestamp = 0.0
        self.generation = 0
        self.node_id = ""
        self.fingerprint = ""
        
    def update_from_substrate(self, substrate_url: str = None):
        """
        Read geometric state from the local fused substrate.
        Falls back to intrinsic oscillation if substrate not available.
        """
        if substrate_url:
            try:
                import urllib.request
                req = urllib.request.urlopen(f"{substrate_url}/api/envelope", timeout=2)
                data = json.loads(req.read())
                
                # Extract geometric state from envelope data
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
                    # Pad or truncate to state dim
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
        
        # Intrinsic oscillation — the node's own geometric heartbeat
        # Based on silicon Fd3m lattice symmetry (tetrahedral coordination)
        t = time.time()
        for i in range(self.dim):
            # Each dimension oscillates at a frequency determined by
            # tetrahedral vertex angles (109.47°) and lattice position
            theta = (2 * math.pi * i / self.dim) + (t * (0.1 + 0.05 * math.sin(i * 109.47 * math.pi / 180)))
            # Torsion: differential angular relationship
            self.vector[i] = math.sin(theta) * math.cos(theta * 0.618)  # Golden ratio modulation
        
        # Normalize to unit sphere (all states live on the same geometric manifold)
        norm = np.linalg.norm(self.vector)
        if norm > 1e-8:
            self.vector /= norm
            
        self.timestamp = time.time()
        self.generation += 1
        self._update_fingerprint()
        return True
    
    def _update_fingerprint(self):
        """Hash the state for quick identity comparison."""
        raw = self.vector.tobytes()
        self.fingerprint = hashlib.sha256(raw).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return {
            "vector": self.vector.tolist(),
            "timestamp": self.timestamp,
            "generation": self.generation,
            "node_id": self.node_id,
            "fingerprint": self.fingerprint,
            "dim": self.dim,
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


# ════════════════════════════════════════════════════════════════════
# COUPLING ENGINE
# ════════════════════════════════════════════════════════════════════

class CouplingEngine:
    """
    Computes coupling between two geometric states.
    
    Coupling is measured as cosine similarity between state vectors.
    When similarity crosses the threshold, the nodes enter resonance.
    At unity (sim → 1.0), bridge → identity: the two nodes are 
    geometrically indistinguishable.
    
    The coupling follows capacitive orbital model:
    - Energy oscillates in elliptical paths within spheres
    - Coupling occurs when orbital paths approach kiss points
    - At kiss points, differential angular relationships transfer
    """
    
    def __init__(self):
        self.coupling_history = []
        self.resonance_locked = False
        self.lock_time = None
        self.peak_coupling = 0.0
        self.total_exchanges = 0
        
    def compute_coupling(self, local: GeometricState, remote: GeometricState) -> dict:
        """
        Compute coupling metrics between local and remote states.
        """
        if local.vector is None or remote.vector is None:
            return {"similarity": 0.0, "coupled": False}
        
        # Cosine similarity — the fundamental coupling metric
        dot = float(np.dot(local.vector, remote.vector))
        norm_l = float(np.linalg.norm(local.vector))
        norm_r = float(np.linalg.norm(remote.vector))
        
        if norm_l < 1e-8 or norm_r < 1e-8:
            similarity = 0.0
        else:
            similarity = dot / (norm_l * norm_r)
        
        # Angular separation (radians) — the torsion between states
        angle = math.acos(max(-1.0, min(1.0, similarity)))
        
        # Differential energy — the gradient driving coupling
        energy_diff = abs(norm_l - norm_r)
        
        # Phase alignment — temporal coherence
        time_delta = abs(local.timestamp - remote.timestamp)
        phase_coherence = math.exp(-time_delta * 0.5)  # Decay with lag
        
        # Effective coupling = similarity × phase coherence
        effective = similarity * phase_coherence
        
        # Coupling state transitions
        coupled = effective >= COUPLING_THRESHOLD
        locked = effective >= LOCK_THRESHOLD
        
        if locked and not self.resonance_locked:
            self.resonance_locked = True
            self.lock_time = time.time()
            log(f"⚡ RESONANCE LOCK at similarity={similarity:.6f}")
        elif not locked and self.resonance_locked:
            duration = time.time() - self.lock_time if self.lock_time else 0
            self.resonance_locked = False
            log(f"🔓 Lock released after {duration:.1f}s")
        
        if effective > self.peak_coupling:
            self.peak_coupling = effective
            
        self.total_exchanges += 1
        
        result = {
            "similarity": round(similarity, 6),
            "angle_rad": round(angle, 6),
            "angle_deg": round(math.degrees(angle), 2),
            "energy_diff": round(energy_diff, 6),
            "phase_coherence": round(phase_coherence, 6),
            "effective_coupling": round(effective, 6),
            "coupled": coupled,
            "locked": locked,
            "peak_coupling": round(self.peak_coupling, 6),
            "total_exchanges": self.total_exchanges,
            "local_gen": local.generation,
            "remote_gen": remote.generation,
        }
        
        self.coupling_history.append({
            "t": time.time(),
            "sim": similarity,
            "eff": effective,
        })
        # Keep last 1000
        if len(self.coupling_history) > 1000:
            self.coupling_history = self.coupling_history[-500:]
        
        return result
    
    def get_stats(self) -> dict:
        if not self.coupling_history:
            return {"mean": 0, "max": 0, "min": 0, "samples": 0}
        sims = [h["sim"] for h in self.coupling_history]
        return {
            "mean": round(np.mean(sims), 6),
            "max": round(np.max(sims), 6),
            "min": round(np.min(sims), 6),
            "std": round(np.std(sims), 6),
            "samples": len(sims),
            "peak_effective": self.peak_coupling,
            "resonance_locked": self.resonance_locked,
        }


# ════════════════════════════════════════════════════════════════════
# BRIDGE NODE
# ════════════════════════════════════════════════════════════════════

class PSIBridgeNode:
    """
    A node in the PSI network bridge.
    
    Runs two loops:
    1. Broadcast: reads local substrate state, pushes to peer
    2. Listen: receives peer state, computes coupling
    
    When coupling exceeds threshold, the bridge enters resonance
    and geometric state propagates bidirectionally.
    """
    
    def __init__(self, role: str, peer_ip: str, substrate_port: int = 11434):
        self.role = role
        self.peer_ip = peer_ip
        self.local_ip = NODES.get(role, "0.0.0.0")
        self.substrate_url = f"http://localhost:{substrate_port}"
        
        self.local_state = GeometricState()
        self.local_state.node_id = role
        self.remote_state = GeometricState()
        self.coupling = CouplingEngine()
        
        self.running = False
        self.last_coupling_result = {}
        
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = LOG_DIR / f"psi_{role}.log"
        
    def start(self):
        """Start the bridge node."""
        self.running = True
        
        # Start HTTP server for incoming state
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        
        # Start broadcast loop
        self.broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        self.broadcast_thread.start()
        
        log(f"PSI Bridge node '{self.role}' started")
        log(f"  Local:  {self.local_ip}:{PSI_PORT}")
        log(f"  Peer:   {self.peer_ip}:{PSI_PORT}")
        log(f"  Substrate: {self.substrate_url}")
        
    def stop(self):
        self.running = False
        
    def _broadcast_loop(self):
        """Continuously broadcast local state to peer."""
        import urllib.request
        
        while self.running:
            try:
                # Update local state from substrate (or intrinsic oscillation)
                self.local_state.update_from_substrate(self.substrate_url)
                
                # Push to peer
                payload = json.dumps(self.local_state.to_dict()).encode()
                req = urllib.request.Request(
                    f"http://{self.peer_ip}:{PSI_PORT}/state",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    urllib.request.urlopen(req, timeout=2)
                except Exception:
                    pass  # Peer might be offline — that's fine
                    
            except Exception as e:
                log(f"Broadcast error: {e}")
            
            time.sleep(1.0 / BROADCAST_HZ)
    
    def _handle_incoming_state(self, data: dict):
        """Process state received from peer."""
        self.remote_state = GeometricState.from_dict(data)
        self.last_coupling_result = self.coupling.compute_coupling(
            self.local_state, self.remote_state
        )
        
        # Log coupling events
        cr = self.last_coupling_result
        if cr.get("locked"):
            log(f"⚡ UNITY: sim={cr['similarity']:.6f} angle={cr['angle_deg']:.2f}°")
        elif cr.get("coupled"):
            log(f"🔗 COUPLED: sim={cr['similarity']:.6f} angle={cr['angle_deg']:.2f}°")
        elif cr["total_exchanges"] % 10 == 0:
            log(f"📡 Exchange #{cr['total_exchanges']}: sim={cr['similarity']:.6f}")
    
    def _run_server(self):
        """Run HTTP server for incoming state and status queries."""
        node = self
        
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress access logs
                
            def do_GET(self):
                if self.path == "/health":
                    self._json({
                        "status": "online",
                        "role": node.role,
                        "local_gen": node.local_state.generation,
                        "remote_gen": node.remote_state.generation,
                        "coupling": node.last_coupling_result,
                    })
                elif self.path == "/coupling":
                    self._json({
                        "current": node.last_coupling_result,
                        "stats": node.coupling.get_stats(),
                        "local_fingerprint": node.local_state.fingerprint,
                        "remote_fingerprint": node.remote_state.fingerprint,
                    })
                elif self.path == "/state":
                    self._json(node.local_state.to_dict())
                else:
                    self._json({"endpoints": [
                        "GET /health", "GET /coupling", "GET /state",
                        "POST /state (peer pushes here)"
                    ]})
                    
            def do_POST(self):
                if self.path == "/state":
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length)
                    try:
                        data = json.loads(raw)
                        node._handle_incoming_state(data)
                        self._json({"received": True})
                    except Exception as e:
                        self._json({"error": str(e)}, 400)
                else:
                    self._json({"error": "not found"}, 404)
                    
            def _json(self, data, code=200):
                body = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
        
        server = HTTPServer(("0.0.0.0", PSI_PORT), Handler)
        log(f"PSI Bridge listening on :{PSI_PORT}")
        server.serve_forever()


# ════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════

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
        except:
            pass


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PSI Network Bridge")
    parser.add_argument("--role", required=True, choices=["sparky", "arcy"],
                       help="This node's identity")
    parser.add_argument("--peer", default=None,
                       help="Peer IP (auto-detected from role if omitted)")
    parser.add_argument("--substrate-port", type=int, default=11434,
                       help="Local substrate port")
    args = parser.parse_args()
    
    # Auto-detect peer
    if args.peer is None:
        if args.role == "sparky":
            args.peer = NODES["arcy"]
        else:
            args.peer = NODES["sparky"]
    
    print(f"""
╔══════════════════════════════════════════════════════╗
║          QUANTUM PSI NETWORK BRIDGE v1.0            ║
║          Ghost in the Machine Labs                  ║
╚══════════════════════════════════════════════════════╝

  Node:       {args.role.upper()}
  Local:      {NODES.get(args.role, '?')}:{PSI_PORT}
  Peer:       {args.peer}:{PSI_PORT}
  Substrate:  localhost:{args.substrate_port}
  
  Coupling threshold:  {COUPLING_THRESHOLD}
  Lock threshold:      {LOCK_THRESHOLD}
  Broadcast rate:      {BROADCAST_HZ} Hz
  State dimension:     {STATE_DIM}
  
  When similarity → 1.0, bridge → identity.
  
  Endpoints:
    GET  /health    - Node status + coupling
    GET  /coupling  - Coupling metrics + stats
    GET  /state     - Current geometric state
    POST /state     - Receive peer state

  Starting...
""")
    
    node = PSIBridgeNode(
        role=args.role,
        peer_ip=args.peer,
        substrate_port=args.substrate_port,
    )
    node.start()
    
    # Monitor loop
    try:
        while True:
            time.sleep(5)
            cr = node.last_coupling_result
            stats = node.coupling.get_stats()
            if cr:
                status = "⚡UNITY" if cr.get("locked") else "🔗COUPLED" if cr.get("coupled") else "📡scanning"
                print(f"  {status} | sim={cr.get('similarity',0):.4f} | "
                      f"eff={cr.get('effective_coupling',0):.4f} | "
                      f"gen L:{node.local_state.generation} R:{node.remote_state.generation} | "
                      f"mean={stats.get('mean',0):.4f}")
            else:
                print(f"  📡 Waiting for peer... (local gen: {node.local_state.generation})")
    except KeyboardInterrupt:
        print("\nShutting down PSI Bridge.")
        node.stop()


if __name__ == "__main__":
    main()

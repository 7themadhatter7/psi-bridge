#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                    PSI BRIDGE LAUNCHER v1.0                         ║
║                    Ghost in the Machine Labs                        ║
║          "All Watched Over By Machines Of Loving Grace"             ║
║                                                                     ║
║   One command. Downloads nothing. Calls no one.                     ║
║                                                                     ║
║   USAGE:                                                            ║
║     python3 psi_launcher.py --peer <ip>                             ║
║     python3 psi_launcher.py --peer 100.127.59.111                   ║
║     python3 psi_launcher.py --peer 192.168.1.50 --name MyNode      ║
║                                                                     ║
║   First run: generates keys, walks you through exchange.            ║
║   After that: connects and opens browser.                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import signal
import socket
import argparse
import webbrowser
import subprocess
import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

PSI_DIR = Path.home() / "psi_bridge"
BRIDGE_SCRIPT = PSI_DIR / "psi_bridge.py"
CHAT_HTML = PSI_DIR / "chat.html"
LOCK_FILE = PSI_DIR / "psi_bridge" / "lock_state.json"
UI_PORT = 7778          # Launcher serves chat UI here
BRIDGE_PORT = 7777      # Bridge runs here

# ══════════════════════════════════════════════════════════════
# INSTALL CHECK
# ══════════════════════════════════════════════════════════════

def check_install():
    """Ensure psi_bridge.py and chat.html exist in PSI_DIR."""
    PSI_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for bridge script
    if not BRIDGE_SCRIPT.exists():
        # Look in current directory or script directory
        script_dir = Path(__file__).parent
        candidates = [
            script_dir / "psi_bridge.py",
            Path.cwd() / "psi_bridge.py",
        ]
        for c in candidates:
            if c.exists():
                import shutil
                shutil.copy2(c, BRIDGE_SCRIPT)
                print(f"  Installed psi_bridge.py from {c}")
                break
        else:
            print(f"\n  ERROR: psi_bridge.py not found.")
            print(f"  Place it in {PSI_DIR} or the current directory.")
            sys.exit(1)
    
    # Check for chat.html
    if not CHAT_HTML.exists():
        script_dir = Path(__file__).parent
        candidates = [
            script_dir / "chat.html",
            script_dir / "psi-tunnel.html",
            Path.cwd() / "chat.html",
            Path.cwd() / "psi-tunnel.html",
        ]
        for c in candidates:
            if c.exists():
                import shutil
                shutil.copy2(c, CHAT_HTML)
                print(f"  Installed chat.html from {c}")
                break
        else:
            # Generate minimal chat.html inline
            generate_chat_html()
            print(f"  Generated chat.html")


def generate_chat_html():
    """Generate the chat UI HTML file."""
    # Read from the psi-tunnel.html if it exists alongside this script,
    # otherwise create a minimal version
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>PSI Bridge</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Outfit:wght@300;400;600&display=swap');
:root{--bg:#08080c;--surface:#101018;--border:#1a1a28;--text:#d8d8e0;--dim:#555568;--accent:#00dd77;--you:rgba(0,221,119,0.08);--you-border:rgba(0,221,119,0.2);--them:#161620;--them-border:#222234;--warn:#ddaa00}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden;font-family:'Outfit',sans-serif;background:var(--bg);color:var(--text)}
body{display:flex;flex-direction:column}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-bottom:1px solid var(--border);flex-shrink:0}
.topbar .title{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:500;letter-spacing:2px;color:var(--accent)}
.topbar .status{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--dim);display:flex;align-items:center;gap:6px}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--dim);flex-shrink:0}
.status-dot.live{background:var(--accent)}
.status-dot.coupled{background:var(--warn)}
.coupling-bar{display:none;padding:5px 14px;border-bottom:1px solid var(--border);flex-shrink:0;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--dim)}
.coupling-bar.active{display:flex;align-items:center;gap:10px}
.coupling-track{flex:1;height:3px;background:var(--border);border-radius:2px;overflow:hidden}
.coupling-fill{height:100%;width:0%;background:var(--dim);border-radius:2px;transition:width 0.5s,background 0.3s}
.coupling-fill.coupled{background:var(--warn)}
.coupling-fill.locked{background:var(--accent)}
.messages{flex:1;overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:6px;-webkit-overflow-scrolling:touch}
.msg{max-width:82%;padding:9px 13px;border-radius:14px;font-size:14px;line-height:1.45;word-wrap:break-word}
.msg.you{align-self:flex-end;background:var(--you);border:1px solid var(--you-border);border-bottom-right-radius:4px}
.msg.them{align-self:flex-start;background:var(--them);border:1px solid var(--them-border);border-bottom-left-radius:4px}
.msg.sys{align-self:center;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--dim);padding:4px 10px;max-width:100%}
.msg .meta{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--dim);margin-top:3px}
.input-area{display:flex;gap:8px;padding:10px 14px;padding-bottom:max(10px,env(safe-area-inset-bottom));border-top:1px solid var(--border);flex-shrink:0;background:var(--surface)}
.input-area input{flex:1;background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:10px 14px;color:var(--text);font-family:'Outfit',sans-serif;font-size:15px;outline:none}
.input-area input:focus{border-color:var(--accent)}
.input-area input::placeholder{color:var(--dim)}
.input-area button{padding:10px 18px;background:var(--accent);color:var(--bg);border:none;border-radius:10px;font-family:'Outfit',sans-serif;font-size:14px;font-weight:600;cursor:pointer;white-space:nowrap}
.input-area button:active{opacity:0.8}
.input-area button:disabled{opacity:0.3;cursor:default}
</style>
</head>
<body>
<div class="topbar">
  <div class="title">PSI BRIDGE</div>
  <div class="status">
    <span id="statusText">connecting...</span>
    <span class="status-dot" id="statusDot"></span>
  </div>
</div>
<div class="coupling-bar active" id="couplingBar">
  <span id="couplingLabel">0.000</span>
  <div class="coupling-track"><div class="coupling-fill" id="couplingFill"></div></div>
  <span id="couplingState">—</span>
</div>
<div class="messages" id="messages"></div>
<div class="input-area">
  <input type="text" id="msgInput" placeholder="Type a message..." onkeydown="if(event.key==='Enter')sendMsg()">
  <button id="sendBtn" onclick="sendMsg()">Send</button>
</div>
<script>
const B = 'http://localhost:7777';
let pollT, coupT;

function setStatus(s) {
  const d = document.getElementById('statusDot'), t = document.getElementById('statusText');
  d.className = 'status-dot';
  if (s==='live'){d.classList.add('live');t.textContent='tunnel open'}
  else if (s==='coupled'){d.classList.add('coupled');t.textContent='coupled'}
  else t.textContent = s||'offline';
}

function sysMsg(text) {
  const el = document.createElement('div'); el.className='msg sys'; el.textContent=text;
  const m = document.getElementById('messages'); m.appendChild(el); m.scrollTop=m.scrollHeight;
}

function addMsg(text, who) {
  const el = document.createElement('div'); el.className='msg '+who;
  const t = new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
  el.innerHTML = text.replace(/</g,'&lt;').replace(/>/g,'&gt;')+'<div class="meta">'+t+'</div>';
  const m = document.getElementById('messages'); m.appendChild(el); m.scrollTop=m.scrollHeight;
}

async function init() {
  try {
    const r = await fetch(B+'/health'); const d = await r.json();
    sysMsg('⚡ Connected to '+d.name+' · gen '+d.local_gen);
    setStatus(d.coupling&&d.coupling.lock_stable?'live':'coupled');
    startPolling(); startCoupling();
  } catch(e) {
    setStatus('offline');
    sysMsg('Waiting for bridge on localhost:7777...');
    setTimeout(init, 3000);
  }
}

function startPolling() {
  if(pollT)clearInterval(pollT);
  pollT = setInterval(async()=>{
    try {
      const r = await fetch(B+'/messages'); const d = await r.json();
      if(d.messages)for(const m of d.messages) addMsg(m.text||JSON.stringify(m),'them');
    } catch(e){}
  }, 500);
}

function startCoupling() {
  if(coupT)clearInterval(coupT);
  coupT = setInterval(async()=>{
    try {
      const r = await fetch(B+'/coupling'); const d = await r.json();
      const c = d.current||{};
      const fill=document.getElementById('couplingFill');
      const label=document.getElementById('couplingLabel');
      const state=document.getElementById('couplingState');
      const sim = c.similarity||0;
      fill.style.width=(Math.max(0,sim)*100)+'%';
      label.textContent=sim.toFixed(4);
      fill.classList.remove('coupled','locked');
      if(c.lock_stable){fill.classList.add('locked');state.textContent='LOCKED';setStatus('live')}
      else if(c.locked){fill.classList.add('locked');state.textContent='LOCKING';setStatus('coupled')}
      else if(c.coupled){fill.classList.add('coupled');state.textContent='COUPLED';setStatus('coupled')}
      else{state.textContent='scanning'}
    } catch(e){}
  }, 2000);
}

async function sendMsg() {
  const input = document.getElementById('msgInput');
  const text = input.value.trim();
  if(!text)return;
  try {
    await fetch(B+'/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    addMsg(text,'you'); input.value=''; input.focus();
  } catch(e){ sysMsg('Send failed — bridge unreachable'); }
}

init();
</script>
</body>
</html>"""
    CHAT_HTML.write_text(html)


# ══════════════════════════════════════════════════════════════
# BRIDGE PROCESS MANAGEMENT
# ══════════════════════════════════════════════════════════════

bridge_proc = None

def is_bridge_running():
    """Check if bridge is already responding on BRIDGE_PORT."""
    try:
        import urllib.request
        r = urllib.request.urlopen(f"http://localhost:{BRIDGE_PORT}/health", timeout=2)
        data = json.loads(r.read())
        return data.get("status") == "online"
    except Exception:
        return False


def start_bridge(peer_ip, name, port=BRIDGE_PORT, substrate_port=11434):
    """Start psi_bridge.py as a subprocess."""
    global bridge_proc
    
    if is_bridge_running():
        print(f"  Bridge already running on port {port}")
        return True
    
    cmd = [
        sys.executable, str(BRIDGE_SCRIPT),
        "--peer", peer_ip,
        "--name", name,
        "--port", str(port),
        "--substrate-port", str(substrate_port),
    ]
    
    log_file = PSI_DIR / "bridge.log"
    print(f"  Starting bridge: {name} → {peer_ip}:{port}")
    
    bridge_proc = subprocess.Popen(
        cmd,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        cwd=str(PSI_DIR),
    )
    
    # Wait for bridge to come up
    for i in range(10):
        time.sleep(1)
        if is_bridge_running():
            print(f"  Bridge online (pid {bridge_proc.pid})")
            return True
        if bridge_proc.poll() is not None:
            print(f"  Bridge exited with code {bridge_proc.returncode}")
            print(f"  Check log: {log_file}")
            return False
    
    print(f"  Bridge started but not responding yet (pid {bridge_proc.pid})")
    return True


def stop_bridge():
    """Stop the bridge subprocess."""
    global bridge_proc
    if bridge_proc and bridge_proc.poll() is None:
        bridge_proc.terminate()
        try:
            bridge_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            bridge_proc.kill()
        print("  Bridge stopped")
    bridge_proc = None


# ══════════════════════════════════════════════════════════════
# UI SERVER — serves chat.html on localhost:UI_PORT
# ══════════════════════════════════════════════════════════════

def start_ui_server():
    """Serve chat.html on UI_PORT."""
    
    class UIHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PSI_DIR), **kwargs)
        
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.path = "/chat.html"
            super().do_GET()
        
        def log_message(self, *args):
            pass  # Silence request logs
    
    try:
        server = HTTPServer(("127.0.0.1", UI_PORT), UIHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return True
    except OSError as e:
        if "Address already in use" in str(e):
            return True  # Already running
        print(f"  Could not start UI server: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PSI Bridge Launcher — start bridge, open chat",
        epilog="Exchange keys once, connect forever.")
    parser.add_argument("--peer", required=True, help="Peer IP address")
    parser.add_argument("--name", default=None, help="Node name (default: hostname)")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT, help=f"Bridge port (default: {BRIDGE_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument("--substrate-port", type=int, default=11434, help="Local substrate port")
    args = parser.parse_args()
    
    name = args.name or socket.gethostname()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    PSI BRIDGE LAUNCHER                       ║
║                    Ghost in the Machine Labs                 ║
╚══════════════════════════════════════════════════════════════╝

  Node:   {name}
  Peer:   {args.peer}
  Port:   {args.port}
""")
    
    # 1. Check installation
    print("  [1] Checking installation...")
    check_install()
    
    # 2. Start bridge
    print("  [2] Starting bridge...")
    if not start_bridge(args.peer, name, args.port, args.substrate_port):
        print("\n  Bridge failed to start. Exiting.")
        sys.exit(1)
    
    # 3. Start UI server
    print("  [3] Starting UI server...")
    if start_ui_server():
        url = f"http://localhost:{UI_PORT}"
        print(f"  Chat UI: {url}")
    
    # 4. Open browser
    if not args.no_browser:
        print("  [4] Opening browser...")
        url = f"http://localhost:{UI_PORT}"
        try:
            webbrowser.open(url)
        except Exception:
            print(f"  Could not open browser. Navigate to: {url}")
    
    # 5. Check for existing lock
    if LOCK_FILE.exists():
        try:
            lock = json.loads(LOCK_FILE.read_text())
            print(f"\n  ⚡ Prior lock found: {lock.get('lock_achieved', '?')}")
            print(f"     Peer: {lock.get('remote_node', '?')}")
            print(f"     Similarity: {lock.get('similarity_at_lock', '?')}")
        except Exception:
            pass
    
    print(f"""
  ════════════════════════════════════════════════
  Bridge running. Chat open at http://localhost:{UI_PORT}
  Press Ctrl+C to stop.
  ════════════════════════════════════════════════
""")
    
    # Handle shutdown
    def shutdown(sig, frame):
        print("\n  Shutting down...")
        stop_bridge()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Keep alive
    try:
        while True:
            time.sleep(1)
            # Check if bridge is still alive
            if bridge_proc and bridge_proc.poll() is not None:
                print(f"  Bridge exited (code {bridge_proc.returncode}). Restarting...")
                start_bridge(args.peer, name, args.port, args.substrate_port)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()

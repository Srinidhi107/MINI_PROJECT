import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.predict import load_trained_model, predict_c_code

print("[INFO] Loading Graph Transformer model and vocabulary for Prototype Server...")
MODEL, VOCAB, DEVICE, THRESHOLD = load_trained_model()
print(f"[INFO] Model successfully loaded (Threshold: {THRESHOLD:.2f}) and ready for live inference!")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Devign: Graph Transformer Vulnerability Detection</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #090d16;
      --card-bg: rgba(18, 24, 38, 0.85);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --danger: #ef4444;
      --success: #10b981;
      --accent: #38bdf8;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --code-bg: #0d1117;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-dark);
      color: var(--text-main);
      font-family: 'Plus Jakarta Sans', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      background-image: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
                        radial-gradient(circle at 85% 85%, rgba(56, 189, 248, 0.08) 0%, transparent 40%);
    }
    header {
      border-bottom: 1px solid var(--card-border);
      padding: 1.25rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(9, 13, 22, 0.8);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .brand { display: flex; align-items: center; gap: 0.75rem; }
    .brand-badge {
      background: linear-gradient(135deg, var(--primary), var(--accent));
      padding: 0.4rem 0.75rem;
      border-radius: 8px;
      font-weight: 700;
      font-size: 0.9rem;
      letter-spacing: 0.5px;
    }
    .brand-title { font-size: 1.2rem; font-weight: 700; }
    .nav-tabs { display: flex; gap: 0.5rem; }
    .nav-tab {
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      padding: 0.5rem 1rem;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .nav-tab.active {
      background: rgba(99, 102, 241, 0.15);
      border-color: rgba(99, 102, 241, 0.3);
      color: var(--accent);
    }
    .container {
      max-width: 1300px;
      margin: 2rem auto;
      padding: 0 1.5rem;
      width: 100%;
      flex: 1;
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; animation: fadeIn 0.3s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

    .grid-2 {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 1.5rem;
    }
    @media (max-width: 960px) { .grid-2 { grid-template-columns: 1fr; } }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.5rem;
      backdrop-filter: blur(16px);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .card-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .presets {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-bottom: 1rem;
    }
    .preset-btn {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .preset-btn:hover {
      background: rgba(99, 102, 241, 0.2);
      color: var(--text-main);
      border-color: var(--primary);
    }
    textarea {
      width: 100%;
      height: 360px;
      background: var(--code-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      color: #e6edf3;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.9rem;
      padding: 1rem;
      resize: vertical;
      line-height: 1.5;
      outline: none;
    }
    textarea:focus { border-color: var(--primary); box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25); }
    .btn-submit {
      width: 100%;
      margin-top: 1rem;
      background: linear-gradient(135deg, var(--primary), var(--primary-hover));
      color: white;
      border: none;
      padding: 0.9rem;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      transition: transform 0.1s, filter 0.2s;
    }
    .btn-submit:hover { filter: brightness(1.1); transform: translateY(-1px); }
    .btn-submit:active { transform: translateY(1px); }
    .btn-submit:disabled { opacity: 0.6; cursor: not-allowed; }

    /* Result Panel */
    .result-box {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      height: 100%;
      justify-content: center;
      text-align: center;
    }
    .status-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      padding: 0.75rem 1.5rem;
      border-radius: 9999px;
      font-size: 1.25rem;
      font-weight: 700;
      margin: 0 auto;
    }
    .status-badge.safe {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-badge.vuln {
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .meter-container {
      background: rgba(0, 0, 0, 0.3);
      padding: 1.25rem;
      border-radius: 12px;
      border: 1px solid var(--card-border);
    }
    .meter-label { display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 0.5rem; }
    .meter-bar {
      height: 12px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 6px;
      overflow: hidden;
      position: relative;
    }
    .meter-fill {
      height: 100%;
      transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
      border-radius: 6px;
    }
    .stat-pills {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }
    .stat-pill {
      background: rgba(0, 0, 0, 0.2);
      border: 1px solid var(--card-border);
      padding: 0.75rem;
      border-radius: 10px;
      text-align: left;
    }
    .stat-pill-label { font-size: 0.8rem; color: var(--text-muted); }
    .stat-pill-val { font-size: 1.15rem; font-weight: 700; color: var(--accent); margin-top: 0.25rem; }

    /* Overview Dashboard */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }
    .metric-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      padding: 1.25rem;
      border-radius: 14px;
      text-align: center;
    }
    .metric-num { font-size: 2rem; font-weight: 800; color: var(--accent); margin: 0.25rem 0; }
    .metric-title { font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }

    .plots-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
      gap: 1.5rem;
    }
    .plot-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1rem;
      text-align: center;
    }
    .plot-card img {
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      margin-top: 0.75rem;
    }

    .loader {
      border: 3px solid rgba(255,255,255,0.1);
      border-top: 3px solid var(--accent);
      border-radius: 50%;
      width: 24px;
      height: 24px;
      animation: spin 0.8s linear infinite;
      display: inline-block;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-badge">Devign AI</div>
      <div class="brand-title">Graph Transformer Vulnerability Detection</div>
    </div>
    <div class="nav-tabs">
      <button class="nav-tab active" onclick="switchTab('scanner')">Live Scanner</button>
      <button class="nav-tab" onclick="switchTab('dashboard')">Project Review Dashboard</button>
    </div>
  </header>

  <div class="container">
    <!-- TAB 1: LIVE SCANNER -->
    <div id="tab-scanner" class="tab-content active">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>C/C++ Source Code Function</span>
            <span style="font-size: 0.8rem; color: var(--text-muted); font-weight: normal;">Joern CPG + Graph Transformer</span>
          </div>

          <div class="presets">
            <span style="font-size: 0.8rem; color: var(--text-muted); align-self: center;">Test Presets:</span>
            <button class="preset-btn" onclick="loadPreset('buffer_overflow')">Buffer Overflow</button>
            <button class="preset-btn" onclick="loadPreset('safe_func')">Safe Null Check</button>
            <button class="preset-btn" onclick="loadPreset('use_after_free')">Use-After-Free</button>
            <button class="preset-btn" onclick="loadPreset('int_overflow')">Integer Overflow</button>
          </div>

          <textarea id="codeInput" placeholder="Paste your C/C++ function code here..."></textarea>

          <button id="scanBtn" class="btn-submit" onclick="analyzeCode()">
            <span id="btnText">Analyze & Detect Vulnerability</span>
          </button>
        </div>

        <div class="card">
          <div class="card-title">Vulnerability Analysis Result</div>
          <div id="resultContent" class="result-box">
            <div style="color: var(--text-muted); font-size: 0.95rem;">
              👈 Paste or select a C/C++ function on the left and click <strong>"Analyze & Detect Vulnerability"</strong>.
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: REVIEW DASHBOARD -->
    <div id="tab-dashboard" class="tab-content">
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-title">Total Processed Graphs</div>
          <div class="metric-num">5,979</div>
          <div style="font-size: 0.8rem; color: var(--text-muted);">Devign Benchmark</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">Test Recall Rate</div>
          <div class="metric-num" style="color: #34d399;">90.0%</div>
          <div style="font-size: 0.8rem; color: var(--text-muted);">Caught 135/150 Vulns</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">Vulnerable Class F1</div>
          <div class="metric-num" style="color: var(--accent);">0.646</div>
          <div style="font-size: 0.8rem; color: var(--text-muted);">Balanced Precision & Recall</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">Graph Representation</div>
          <div class="metric-num">AST+CFG+PDG</div>
          <div style="font-size: 0.8rem; color: var(--text-muted);">Joern Code Property Graphs</div>
        </div>
      </div>

      <div class="plots-grid">
        <div class="plot-card">
          <div style="font-weight: 600;">Training & Validation Loss Curve</div>
          <img src="/results/plots/training_loss.png" alt="Loss Curve" onerror="this.src='/results/plots/training_metrics.png'">
        </div>
        <div class="plot-card">
          <div style="font-weight: 600;">Test Confusion Matrix</div>
          <img src="/results/plots/test_confusion_matrix.png" alt="Confusion Matrix">
        </div>
      </div>
    </div>
  </div>

  <script>
    const PRESETS = {
      buffer_overflow: `int process_user_input(char *src_str, int len) {
    char local_buffer[64];
    if (len > 0) {
        // VULNERABLE: Unbounded strcpy causes buffer overflow
        strcpy(local_buffer, src_str);
        return 0;
    }
    return -1;
}`,
      safe_func: `int safe_copy_buffer(const char *src, char *dest, size_t dest_size) {
    if (!src || !dest || dest_size == 0) {
        return -1;
    }
    // SAFE: Explicit bounds checking with snprintf
    snprintf(dest, dest_size, "%s", src);
    return 0;
}`,
      use_after_free: `void handle_connection(struct Client *client) {
    if (client->has_error) {
        free(client);
    }
    // VULNERABLE: Accessing memory after potential free()
    client->status = STATUS_CLOSED;
}`,
      int_overflow: `void *allocate_table(int num_entries, int entry_size) {
    // VULNERABLE: Integer multiplication can overflow leading to undersized allocation
    int total_bytes = num_entries * entry_size;
    void *ptr = malloc(total_bytes);
    return ptr;
}`
    };

    function loadPreset(key) {
      document.getElementById('codeInput').value = PRESETS[key] || '';
    }
    // Load default
    loadPreset('buffer_overflow');

    function switchTab(tab) {
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      if (tab === 'scanner') {
        document.querySelector('.nav-tab:nth-child(1)').classList.add('active');
        document.getElementById('tab-scanner').classList.add('active');
      } else {
        document.querySelector('.nav-tab:nth-child(2)').classList.add('active');
        document.getElementById('tab-dashboard').classList.add('active');
      }
    }

    async function analyzeCode() {
      const code = document.getElementById('codeInput').value.trim();
      if (!code) {
        alert("Please paste C/C++ source code to analyze!");
        return;
      }

      const scanBtn = document.getElementById('scanBtn');
      const resultContent = document.getElementById('resultContent');
      
      scanBtn.disabled = true;
      scanBtn.innerHTML = '<div class="loader"></div><span>Generating Joern CPG & Inferencing...</span>';

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: code })
        });

        const data = await response.json();

        if (data.status === 'error') {
          resultContent.innerHTML = `
            <div style="color: var(--danger); font-size: 1.1rem; font-weight: 600;">❌ Analysis Error</div>
            <div style="color: var(--text-muted); font-size: 0.9rem;">\${data.message}</div>
          `;
          return;
        }

        const isVuln = data.prediction_label === 1;
        const badgeClass = isVuln ? 'vuln' : 'safe';
        const badgeIcon = isVuln ? '🚨 VULNERABLE FUNCTION' : '🛡️ SAFE FUNCTION';
        const fillGrad = isVuln 
          ? 'linear-gradient(90deg, #f59e0b, #ef4444)' 
          : 'linear-gradient(90deg, #38bdf8, #10b981)';

        resultContent.innerHTML = `
          <div class="status-badge \${badgeClass}">
            <span>\${badgeIcon}</span>
          </div>

          <div class="meter-container">
            <div class="meter-label">
              <span>Vulnerability Probability</span>
              <span style="color: \${isVuln ? '#f87171' : '#34d399'};">\${data.vulnerability_probability}%</span>
            </div>
            <div class="meter-bar">
              <div class="meter-fill" style="width: \${data.vulnerability_probability}%; background: \${fillGrad};"></div>
            </div>
          </div>

          <div class="stat-pills">
            <div class="stat-pill">
              <div class="stat-pill-label">CPG Nodes</div>
              <div class="stat-pill-val">\${data.graph_stats.nodes_count}</div>
            </div>
            <div class="stat-pill">
              <div class="stat-pill-label">CPG Edges</div>
              <div class="stat-pill-val">\${data.graph_stats.edges_count}</div>
            </div>
            <div class="stat-pill">
              <div class="stat-pill-label">Graph Transformer Layers</div>
              <div class="stat-pill-val" style="color: var(--primary);">4 Attention Layers</div>
            </div>
            <div class="stat-pill">
              <div class="stat-pill-label">Edge Feature Types</div>
              <div class="stat-pill-val" style="color: var(--primary);">AST + CFG + PDG</div>
            </div>
          </div>
        `;

      } catch (err) {
        resultContent.innerHTML = `
          <div style="color: var(--danger);">Connection Error: \${err.message}</div>
        `;
      } finally {
        scanBtn.disabled = false;
        scanBtn.innerHTML = '<span>Analyze & Detect Vulnerability</span>';
      }
    }
  </script>
</body>
</html>
"""

class PrototypeServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        
        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return
            
        elif path.startswith("/results/plots/"):
            rel_path = path.lstrip("/")
            local_path = os.path.join(PROJECT_ROOT, rel_path)
            if os.path.exists(local_path) and local_path.endswith(".png"):
                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.end_headers()
                with open(local_path, "rb") as f:
                    self.wfile.write(f.read())
                return
                
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode("utf-8"))
                code = payload.get("code", "")
                
                # Perform inference
                result = predict_c_code(code, MODEL, VOCAB, DEVICE, threshold=THRESHOLD, verbose=False)
                
                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return
            
        self.send_response(404)
        self.end_headers()

def run_server(port: int = 5000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, PrototypeServerHandler)
    print("\n" + "=" * 70)
    print("🚀 DEVIGN GRAPH TRANSFORMER PROTOTYPE SERVER")
    print("=" * 70)
    print(f"👉 Local Demo URL: http://localhost:{port}")
    print(f"👉 Press Ctrl+C in terminal to stop server")
    print("=" * 70 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    port = 5000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)

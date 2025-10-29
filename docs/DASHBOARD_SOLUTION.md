================================================================================
VEGA DASHBOARD - MEMORY LIMITATION ISSUE
================================================================================
Date: October 29, 2025

PROBLEM IDENTIFIED:
-------------------
The Vega system has heavy dependencies (torch, transformers, etc.) that cause
memory allocation errors (std::bad_alloc) when trying to start programmatically.

The error occurs because:
  1. src/vega/core/app.py imports the full Vega AI system
  2. This triggers loading of PyTorch, transformers, and other ML libraries
  3. System runs out of available memory during initialization
  4. Error: "terminate called after throwing an instance of 'std::bad_alloc'"

ROOT CAUSE:
-----------
Your system (125GB RAM, but possibly high usage) cannot allocate enough 
contiguous memory for torch initialization when launched via subprocess.

This is a limitation of:
  - Python's memory management
  - Torch/CUDA initialization requirements
  - Current system memory usage

RECOMMENDED SOLUTION:
---------------------
Instead of using the dashboard buttons to start/stop Vega, use these approaches:

OPTION 1: Manual Start/Stop (Recommended)
------------------------------------------
Start Vega manually in a dedicated terminal:

  # Start Vega server
  cd /home/ncacord/Vega2.0
  source .venv/bin/activate
  python main.py server --host 127.0.0.1 --port 8000

  # Or in background with nohup
  nohup python main.py server > vega_server.log 2>&1 &

The dashboard will automatically detect when Vega is running and update status.

OPTION 2: Systemd Service (Best for Production)
------------------------------------------------
Create a systemd service for Vega (similar to the dashboard):

  1. Create /etc/systemd/system/vega-server.service:

[Unit]
Description=Vega AI Server
After=network.target

[Service]
Type=simple
User=ncacord
WorkingDirectory=/home/ncacord/Vega2.0
Environment="PATH=/home/ncacord/Vega2.0/.venv/bin"
ExecStart=/home/ncacord/Vega2.0/.venv/bin/python main.py server --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target

  2. Install and start:
     sudo systemctl daemon-reload
     sudo systemctl enable vega-server
     sudo systemctl start vega-server

  3. Manage:
     sudo systemctl status vega-server
     sudo systemctl stop vega-server
     sudo systemctl restart vega-server

OPTION 3: Screen/Tmux Session
------------------------------
Run Vega in a persistent terminal session:

  # Using screen
  screen -S vega
  cd /home/ncacord/Vega2.0 && source .venv/bin/activate
  python main.py server
  # Press Ctrl+A then D to detach
  # Reattach with: screen -r vega

  # Using tmux
  tmux new -s vega
  cd /home/ncacord/Vega2.0 && source .venv/bin/activate
  python main.py server
  # Press Ctrl+B then D to detach
  # Reattach with: tmux attach -t vega

DASHBOARD FUNCTIONALITY:
-------------------------
The dashboard will still work for monitoring:
  ✅ View system status (CPU, Memory, GPU)
  ✅ View conversation history
  ✅ Monitor AI thoughts
  ✅ See recent interactions
  ✅ Real-time metrics updates

But control buttons won't work due to memory limitations:
  ❌ "Start System" - causes memory allocation error
  ❌ "Stop System" - will work if Vega is running
  ⚠️  "Force Chat" - requires Vega to be running first

WHY THIS HAPPENS:
-----------------
When you click "Start System":
  1. Dashboard spawns subprocess: python -m uvicorn vega.core.app:app
  2. Uvicorn imports app.py
  3. app.py imports torch, transformers, langchain, etc.
  4. Torch tries to initialize CUDA and allocate memory
  5. System cannot allocate required memory → std::bad_alloc
  6. Process crashes before server starts

When you start manually in terminal:
  1. You have full shell environment
  2. Memory is allocated in parent process context
  3. Better memory management and allocation patterns
  4. Works successfully

MEMORY OPTIMIZATION TIPS:
--------------------------
To reduce memory usage and help Vega start:

  1. Close unnecessary applications before starting Vega
  2. Check current memory: free -h
  3. Clear cache: sudo sync && sudo sysctl vm.drop_caches=3
  4. Monitor processes: htop or top
  5. Consider increasing swap space if needed

VERIFICATION:
-------------
After starting Vega manually, verify the dashboard can monitor it:

  1. Start Vega in terminal (see Option 1 above)
  2. Open dashboard: http://192.168.1.147:8080
  3. Click "Refresh" button
  4. Status should show "System Running"
  5. CPU/Memory/GPU metrics should update

FUTURE IMPROVEMENTS:
--------------------
Possible solutions to enable dashboard control:

  1. Lazy loading: Modify app.py to import torch only when needed
  2. Memory limits: Set PYTORCH_CUDA_ALLOC_CONF environment variables
  3. Separate workers: Run ML models in separate process pool
  4. Container limits: Use Docker with memory constraints
  5. Code refactoring: Split heavy imports into separate modules

================================================================================
SUMMARY: Use manual start or systemd service for Vega, dashboard for monitoring
================================================================================

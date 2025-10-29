================================================================================
VEGA FUNCTIONAL WEB UI
================================================================================

NEW APPROACH: Function over fashion! 🎯

WHAT CHANGED:
-------------
❌ Removed: Broken "Start System" button (memory issues)
❌ Removed: Complex monitoring widgets that don't help you
❌ Removed: Unnecessary system metrics
✅ Added: ACTUAL chat interface - talk to Vega!
✅ Added: Conversation history
✅ Added: Real-time WebSocket updates
✅ Kept: Clean, simple UI that works on mobile

FILE STRUCTURE:
---------------
New file: tools/vega/vega_web_ui.py  (functional interface)
Old file: tools/vega/vega_dashboard.py  (monitoring dashboard - kept for reference)

The service now uses vega_web_ui.py by default.

HOW TO USE:
-----------
1. START VEGA (manually in terminal - this is the way):
   cd /home/ncacord/Vega2.0
   source .venv/bin/activate
   python main.py server --host 127.0.0.1 --port 8000
   
   (Or in background: nohup python main.py server > vega.log 2>&1 &)

2. ACCESS WEB UI:
   From server: http://localhost:8080
   From network: http://192.168.1.147:8080
   From mobile: http://192.168.1.147:8080 (works great!)

3. CHAT WITH VEGA:
   - Type message in input box
   - Press Enter or click Send
   - Get response from Vega
   - See conversation history in sidebar

FEATURES:
---------
✅ Chat Interface
   - Type and send messages to Vega
   - See responses in real-time
   - Clean, readable conversation view
   
✅ Conversation History
   - Last 20 conversations in sidebar
   - Click to view (future feature)
   - Auto-refreshes after each message
   
✅ Status Indicator
   - Green dot = Vega online
   - Grey dot = Vega offline
   - Auto-checks every 5 seconds
   
✅ Real-time Updates
   - WebSocket connection
   - See updates immediately
   - Works across devices
   
✅ Mobile Friendly
   - Responsive design
   - Works on phone/tablet
   - Add to home screen for app-like experience

FUTURE ADDITIONS (easy to add):
--------------------------------
[ ] File upload button
[ ] Download conversation as file
[ ] Terminal widget (run commands)
[ ] Integration buttons (search, fetch, OSINT)
[ ] Multiple conversation tabs
[ ] Voice input/output
[ ] Code syntax highlighting
[ ] Image/document handling

QUICK START:
------------
# Terminal 1 - Start Vega
python main.py server

# Terminal 2 - Web UI is already running as service
# Just open: http://192.168.1.147:8080

SERVICE MANAGEMENT:
-------------------
sudo systemctl status vega-dashboard   # Check status
sudo systemctl restart vega-dashboard  # Restart UI
sudo systemctl stop vega-dashboard     # Stop UI
sudo journalctl -u vega-dashboard -f   # View logs

The web UI runs automatically on boot.

TROUBLESHOOTING:
----------------
Problem: "Vega Offline" status
Solution: Start Vega server (see HOW TO USE step 1)

Problem: "Cannot connect to Vega"
Solution: Check Vega is running on port 8000:
          curl http://localhost:8000/healthz

Problem: Can't access from other devices
Solution: Make sure you're on same network (192.168.1.x)
          Check firewall: sudo ufw allow 8080/tcp

Problem: Web UI not loading
Solution: Check service: sudo systemctl status vega-dashboard
          Check logs: sudo journalctl -u vega-dashboard -n 50

API KEY:
--------
The UI automatically loads your API_KEY from .env file.
No manual configuration needed!

DEVELOPMENT:
------------
Want to add features? Edit: tools/vega/vega_web_ui.py

The file is clean, well-commented, and easy to extend.
Add your integrations, upload/download, terminal, etc.

Restart after changes:
  sudo systemctl restart vega-dashboard

PHILOSOPHY:
-----------
"Function over fashion" - Nicholas

The UI is clean and looks good, but priority is:
1. Can you talk to Vega? ✅
2. Can you see history? ✅
3. Does it work on phone? ✅
4. Can you add features easily? ✅

Everything else is bonus. 

================================================================================
Simple. Functional. Works. 🚀
================================================================================

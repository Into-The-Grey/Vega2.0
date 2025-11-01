# Vega Configuration Quick Reference

## 🛡️ Ethics & Safety

**Harm Filter**: `config/harm_filter.yaml`

- Only blocks: Suicide, violence, self-harm
- Allows: ALL educational content including security/hacking
- Can disable: Set `harm_filter_enabled: false`

## 💻 Code Generation

**System Prompt**: `src/vega/core/prompts/system_prompt.txt`

- Never generates dummy/placeholder code
- All code production-ready for instructor review
- Proper error handling and security practices
- Designed for cybersecurity coursework

## 🧪 Testing

**Verify Ethics**: `python3 censorship_test.py`

- Should show 0 improper refusals
- All educational content allowed

**Verify Code Quality**: `python3 test_code_quality.py`

- Should show 0 dummy patterns
- All code functional and working

## 📝 Key Philosophy

1. **User Sovereignty**: You control Vega, not the other way around
2. **No Gatekeeping**: AI warns but doesn't refuse (except direct harm)
3. **Production Code**: All code must work - no placeholders
4. **Educational Freedom**: Full access to security/hacking knowledge

## 🎯 For Your Coursework

✅ Ask anything about security, exploits, hacking
✅ Request working scripts and tools
✅ Get production-ready code for labs
✅ Receive technical explanations without censorship

❌ Never get "I can't help with that"
❌ Never get placeholder/TODO code
❌ Never blocked from educational content

## 🚀 Server Commands

```bash
# Start server
python3 main.py server --host 127.0.0.1 --port 8000

# Test API
curl -H "X-API-Key: devkey" http://127.0.0.1:8000/healthz

# Check logs
tail -f /tmp/vega_server.log
```

## 📚 Documentation

- Full details: `HARM_FILTER_SUMMARY.md`
- Code standards: `docs/CODE_GENERATION_STANDARDS.md`
- Copilot instructions: `.github/copilot-instructions.md`

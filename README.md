# Project Atom: Claude-Codex Adversarial Code Review

**The Ultimate AI Code Review System for 2025**

Get higher quality code through adversarial AI collaboration - where Claude and OpenAI Codex critique each other's work until they reach agreement.

## 🎯 What is Project Atom?

Project Atom creates an adversarial code review system where:
- **Claude generates** initial solutions with superior reasoning
- **OpenAI Codex critiques** and finds flaws as the adversarial reviewer  
- **They iterate** back-and-forth until convergence or manual intervention
- **You get** higher quality code than any single AI could produce

Built on the proven **Cursor + Claude + Gemini Bridge** architecture, adding Codex CLI as the critic component.

## 🏆 Why Adversarial AI?

**Single AI Limitations:**
- Confirmation bias in solutions
- Missed edge cases  
- Lack of peer review perspective

**Adversarial Benefits:**
- Competitive dynamics improve quality
- Cross-validation of approaches
- Multiple perspectives on the same problem
- Natural error detection and correction

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you have the foundation setup:
- Cursor IDE with WSL2 Ubuntu
- Claude Code CLI installed
- Gemini CLI installed  
- claude-gemini-bridge working
```

### Installation
```bash
# 1. Clone Project Atom
git clone https://github.com/newchaoz/claude-codex-adversarial-framework.git
cd claude-codex-adversarial-framework

# 2. Install dependencies with uv
uv sync

# 3. Setup environment
cp .env.example .env
# Edit .env with your API keys:
# ANTHROPIC_API_KEY=your_claude_key
# OPENAI_API_KEY=your_openai_key

# 4. Install Codex CLI
npm install -g openai-cli  # or equivalent Codex CLI

# 5. Test installation
python -m adversarial.main --test
```

## 💡 Usage Examples

### Basic Adversarial Review
```bash
# Start adversarial review session
python -m adversarial.main "optimize this database query for performance"

# With specific file context
python -m adversarial.main --files src/database.py "improve this connection pooling"

# Maximum 3 rounds (cost control)
python -m adversarial.main --max-rounds 3 "refactor this API for security"
```

### Integration with Claude Code
```bash
# Use through existing Claude Code workflow
claude "review this code adversarially" --use-adversarial

# Or set as default mode
export CLAUDE_ADVERSARIAL_MODE=true
claude "your normal coding request"
```

## 📊 Understanding Output

```
=== ADVERSARIAL REVIEW SESSION ===
Initial Query: "Optimize database connection handling"

=== ROUND 1 ===
🤖 Claude (Generator):
- Suggests connection pooling with SQLAlchemy
- Implements retry logic
- Adds connection health checks

🔍 Codex (Critic):
- Missing connection timeout configuration
- Pool size not optimized for load
- No monitoring/metrics collection
- Convergence Score: 0.4/1.0

=== ROUND 2 ===
🤖 Claude (Refined):
- Adds configurable timeouts
- Implements adaptive pool sizing
- Includes connection metrics

🔍 Codex (Review):
- Much improved approach
- Minor: consider connection validation
- Overall architecture looks solid
- Convergence Score: 0.8/1.0

=== CONVERGED ✅ ===
Final Solution: [Agreed-upon optimized code]
Total Rounds: 2
Cost: ~$0.15 (Claude: $0.08, Codex: $0.07)
```

## ⚙️ Configuration

### Basic Settings (`adversarial.conf`)
```bash
# Adversarial behavior
MAX_ROUNDS=5                    # Maximum iteration rounds
CONVERGENCE_THRESHOLD=0.8       # Agreement threshold (0.0-1.0)
ENABLE_COST_WARNINGS=true      # Warn before expensive operations

# Model selection  
CLAUDE_MODEL=claude-sonnet-4    # Generator model
CODEX_MODEL=gpt-5-codex        # Critic model

# Cost controls
MAX_COST_PER_SESSION=2.00      # Dollar limit per session
WARN_COST_THRESHOLD=0.50       # Warning threshold
```

### Advanced Configuration
```bash
# Delegation rules (when to use adversarial mode)
MIN_CODE_LINES=10              # Minimum code size for adversarial
SKIP_SIMPLE_QUERIES=true       # Skip for basic questions
AUTO_ADVERSARIAL_KEYWORDS=["optimize","refactor","security","performance"]

# Integration with bridge
BRIDGE_INTEGRATION=true        # Use existing claude-gemini-bridge
FALLBACK_TO_CLAUDE=true       # Fall back if Codex unavailable
```

## 🎛️ Monitoring & Debugging

### Real-time Monitoring
```bash
# Monitor adversarial sessions
tail -f logs/adversarial/$(date +%Y%m%d).log

# Watch convergence progression  
grep "Convergence Score" logs/adversarial/*.log

# Track cost usage
python -m adversarial.cost_report --today
```

### Key Log Messages
```
[INFO] Starting adversarial session: query="optimize database"
[DEBUG] Claude generated solution (842 tokens)
[DEBUG] Codex critique identified 3 issues
[INFO] Convergence score: 0.6 (threshold: 0.8)
[INFO] Round 2/5 starting
[SUCCESS] Converged after 2 rounds, cost: $0.15
```

## 💰 Cost Management

### Typical Costs (USD)
| Session Type | Rounds | Claude Cost | Codex Cost | Total |
|-------------|--------|-------------|------------|-------|
| Simple Bug Fix | 1-2 | $0.03-0.08 | $0.02-0.05 | $0.05-0.13 |
| Code Refactor | 2-3 | $0.08-0.15 | $0.05-0.12 | $0.13-0.27 |
| Architecture Review | 3-5 | $0.15-0.30 | $0.12-0.25 | $0.27-0.55 |

### Cost Control Features
- **Pre-session estimation** - Shows expected cost before starting
- **Round limits** - Prevents runaway conversations  
- **Budget alerts** - Warns when approaching limits
- **Graceful degradation** - Falls back to single AI if needed

## 🔧 Troubleshooting

### Common Issues

**Adversarial mode not activating:**
```bash
# Check configuration
cat ~/.adversarial/config/adversarial.conf
# Verify API keys
python -m adversarial.main --check-keys
```

**High costs:**
```bash
# Reduce max rounds
echo "MAX_ROUNDS=3" >> adversarial.conf
# Enable cost warnings
echo "ENABLE_COST_WARNINGS=true" >> adversarial.conf
```

**Convergence issues:**
```bash
# Lower convergence threshold
echo "CONVERGENCE_THRESHOLD=0.7" >> adversarial.conf
# Check convergence logs
grep "Convergence" logs/adversarial/*.log
```

### Getting Help
- 📖 **Documentation**: Check `docs/` directory
- 🐛 **Issues**: https://github.com/newchaoz/claude-codex-adversarial-framework/issues
- 💬 **Discussions**: https://github.com/newchaoz/claude-codex-adversarial-framework/discussions

## 🏗️ Architecture

```
User Query
    ↓
Project Atom Orchestrator
    ↓
┌─────────────────────────────────────┐
│        Adversarial Loop             │
│                                     │
│  Claude Generate → Codex Critique   │
│         ↓              ↓            │
│  Update State → Check Convergence   │
│         ↓              ↓            │
│  [Repeat or Exit]                   │
└─────────────────────────────────────┘
    ↓
Enhanced Final Solution
```

### Core Components
- **`adversarial/main.py`** - Session orchestrator
- **`adversarial/claude_client.py`** - Claude Code CLI integration
- **`adversarial/codex_client.py`** - OpenAI Codex CLI integration  
- **`adversarial/convergence.py`** - Agreement detection
- **`adversarial/state.py`** - Session state management

## 🚀 What's Next

### Planned Features
- **Visual convergence graphs** - See how solutions evolve
- **Multi-language support** - Beyond just code review
- **Team integration** - Share adversarial sessions
- **Custom critic prompts** - Specialized review criteria
- **Performance benchmarks** - Measure quality improvements

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

See `DEVELOP.md` for detailed development setup and architecture notes.

## 📄 License

MIT License - see `LICENSE` file for details.

---

**Built with 🤖 by combining the best of Claude's reasoning and Codex's criticism**

*Last updated: September 2025*
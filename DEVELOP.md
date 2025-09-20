# Project Atom: Developer Guide

**Internal documentation for developing and extending the Claude-Codex adversarial framework**

## 🏗️ Architecture Deep Dive

### System Overview
Project Atom extends the proven **claude-gemini-bridge** pattern by adding OpenAI Codex as an adversarial critic to Claude's generative capabilities.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Project Atom Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│  User Interface Layer                                           │
│  ├── CLI Commands (python -m adversarial.main)                 │
│  ├── Claude Code Integration (--use-adversarial flag)          │
│  └── Bridge Hook System (automatic delegation)                 │
├─────────────────────────────────────────────────────────────────┤
│  Orchestration Layer                                           │
│  ├── AdversarialSession (main.py)                             │
│  ├── StateManager (state.py)                                  │
│  ├── ConvergenceDetector (convergence.py)                     │
│  └── CostTracker (cost.py)                                    │
├─────────────────────────────────────────────────────────────────┤
│  AI Client Layer                                              │
│  ├── ClaudeClient (claude_client.py) - Generator              │
│  ├── CodexClient (codex_client.py) - Critic                   │
│  └── ClientFactory (clients.py) - Model switching            │
├─────────────────────────────────────────────────────────────────┤
│  Integration Layer                                             │
│  ├── Bridge Integration (hooks/)                              │
│  ├── Configuration Management (config/)                       │
│  └── Logging & Monitoring (logs/)                            │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
claude-codex-adversarial-framework/
├── README.md                    # User documentation
├── DEVELOP.md                   # This file
├── pyproject.toml              # uv project configuration
├── uv.lock                     # Dependency lock file
├── .env.example                # Environment template
├── .gitignore                  # Git ignore patterns
│
├── src/adversarial/            # Main package
│   ├── __init__.py            # Package initialization
│   ├── __main__.py            # CLI entry point
│   ├── main.py                # Session orchestrator
│   ├── state.py               # State management (dataclasses)
│   ├── convergence.py         # Agreement detection logic
│   ├── claude_client.py       # Claude Code CLI wrapper
│   ├── codex_client.py        # OpenAI Codex CLI wrapper
│   ├── clients.py             # Client factory and switching
│   ├── cost.py                # Cost tracking and estimation
│   └── utils.py               # Utility functions
│
├── config/                     # Configuration files
│   ├── adversarial.conf       # Main configuration
│   ├── prompts/               # AI prompt templates
│   │   ├── claude_generator.txt
│   │   ├── codex_critic.txt
│   │   └── convergence_check.txt
│   └── models.json            # Model configuration
│
├── hooks/                      # Bridge integration
│   ├── claude-adversarial.sh  # Main hook script
│   ├── install.sh             # Installation script
│   └── config/
│       └── debug.conf         # Bridge debug configuration
│
├── logs/                       # Log directory
│   ├── adversarial/           # Session logs
│   ├── debug/                 # Debug logs
│   └── cost/                  # Cost tracking logs
│
├── tests/                      # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   ├── fixtures/              # Test data
│   └── test_adversarial.sh    # Installation test script
│
└── docs/                       # Additional documentation
    ├── api.md                 # API documentation
    ├── architecture.md        # Detailed architecture
    ├── prompts.md             # Prompt engineering guide
    └── troubleshooting.md     # Advanced troubleshooting
```

## 🔧 Development Setup

### Prerequisites
```bash
# Ensure foundation is working
- WSL2 Ubuntu
- Cursor IDE
- claude-gemini-bridge installed and working
- Claude Code CLI (claude --version)
- Node.js for CLI tools
```

### Local Development
```bash
# 1. Clone and setup
git clone https://github.com/newchaoz/claude-codex-adversarial-framework.git
cd claude-codex-adversarial-framework

# 2. Install development dependencies
uv sync --dev

# 3. Setup environment
cp .env.example .env
# Add your API keys to .env

# 4. Install in development mode
uv pip install -e .

# 5. Install Codex CLI (if not already installed)
npm install -g openai-cli

# 6. Run tests
python -m pytest tests/
./tests/test_adversarial.sh

# 7. Run development server with hot reload
python -m adversarial.main --debug --watch
```

### Development Dependencies
```toml
# Additional dev dependencies in pyproject.toml
[tool.uv.dev-dependencies]
pytest = "^7.0.0"
pytest-asyncio = "^0.21.0"
black = "^23.0.0"
mypy = "^1.0.0"
ruff = "^0.1.0"
pre-commit = "^3.0.0"
```

## 🧩 Core Components

### 1. AdversarialSession (`main.py`)
**Purpose**: Orchestrates the entire adversarial conversation loop

```python
@dataclass
class AdversarialSession:
    query: str
    max_rounds: int = 5
    convergence_threshold: float = 0.8
    current_round: int = 0
    
    # State tracking
    claude_responses: List[str] = field(default_factory=list)
    codex_critiques: List[str] = field(default_factory=list)
    convergence_scores: List[float] = field(default_factory=list)
    
    # Cost tracking
    total_cost: float = 0.0
    claude_cost: float = 0.0
    codex_cost: float = 0.0
    
    async def run(self) -> AdversarialResult:
        """Main execution loop"""
        while not self.should_stop():
            # Generate solution with Claude
            claude_response = await self.claude_client.generate(
                query=self.current_query(),
                context=self.get_context()
            )
            
            # Critique with Codex
            codex_critique = await self.codex_client.critique(
                solution=claude_response,
                original_query=self.query
            )
            
            # Check convergence
            score = self.convergence_detector.calculate_score(
                claude_response, codex_critique
            )
            
            # Update state and decide next action
            self.update_state(claude_response, codex_critique, score)
            
        return self.create_result()
```

### 2. ConvergenceDetector (`convergence.py`)
**Purpose**: Determines when Claude and Codex have reached agreement

```python
class ConvergenceDetector:
    """Detects when adversarial conversation has converged"""
    
    def calculate_score(self, solution: str, critique: str) -> float:
        """Calculate convergence score (0.0 = complete disagreement, 1.0 = full agreement)"""
        
        # Keywords indicating agreement
        agreement_keywords = [
            "looks good", "well done", "solid approach", "no major issues",
            "good solution", "approved", "ready to use", "no concerns"
        ]
        
        # Keywords indicating disagreement  
        disagreement_keywords = [
            "missing", "incomplete", "security risk", "performance issue",
            "bug", "error", "problem", "concern", "improvement needed"
        ]
        
        # Sentiment analysis
        agreement_score = self._count_keywords(critique.lower(), agreement_keywords)
        disagreement_score = self._count_keywords(critique.lower(), disagreement_keywords)
        
        # Length analysis (shorter critiques often mean fewer issues)
        length_factor = max(0.1, 1.0 - (len(critique.split()) / 500))
        
        # Combine factors
        base_score = agreement_score / (agreement_score + disagreement_score + 1)
        return min(1.0, base_score * 0.7 + length_factor * 0.3)
    
    def is_converged(self, score: float, threshold: float = 0.8) -> bool:
        return score >= threshold
```

### 3. Client Architecture (`claude_client.py`, `codex_client.py`)

#### Claude Client (Generator)
```python
class ClaudeClient:
    """Wrapper for Claude Code CLI focused on solution generation"""
    
    async def generate(self, query: str, context: dict = None) -> str:
        """Generate initial solution or refinement"""
        
        prompt = self._build_generator_prompt(query, context)
        
        # Use Claude Code CLI
        cmd = ["claude", "--json", "--stdin"]
        result = await self._run_cli_command(cmd, prompt)
        
        return self._parse_response(result)
    
    def _build_generator_prompt(self, query: str, context: dict) -> str:
        """Build prompt optimized for solution generation"""
        
        base_prompt = """
        You are a senior software engineer tasked with providing high-quality solutions.
        
        Query: {query}
        
        Please provide:
        1. Complete, working solution
        2. Clear explanation of approach
        3. Consideration of edge cases
        4. Best practices implementation
        
        Previous feedback to address: {previous_feedback}
        """
        
        return base_prompt.format(
            query=query,
            previous_feedback=context.get('previous_critique', 'None')
        )
```

#### Codex Client (Critic)
```python
class CodexClient:
    """Wrapper for OpenAI Codex CLI focused on code criticism"""
    
    async def critique(self, solution: str, original_query: str) -> str:
        """Provide adversarial critique of solution"""
        
        prompt = self._build_critic_prompt(solution, original_query)
        
        # Use OpenAI Codex CLI  
        cmd = ["openai", "api", "completions.create", 
               "-m", "gpt-5-codex", "--max-tokens", "1000"]
        result = await self._run_cli_command(cmd, prompt)
        
        return self._parse_response(result)
    
    def _build_critic_prompt(self, solution: str, query: str) -> str:
        """Build prompt optimized for finding flaws"""
        
        critic_prompt = """
        You are a senior code reviewer conducting an adversarial review.
        Your job is to find flaws, security issues, and improvements.
        
        Original request: {query}
        
        Proposed solution:
        {solution}
        
        Please provide:
        1. Specific issues found (bugs, security, performance)
        2. Missing edge cases or error handling
        3. Code quality and maintainability concerns
        4. Concrete suggestions for improvement
        
        Be thorough but constructive. If the solution is good, say so clearly.
        """
        
        return critic_prompt.format(query=query, solution=solution)
```

### 4. State Management (`state.py`)
**Purpose**: Track conversation state and enable persistence

```python
@dataclass
class AdversarialState:
    """Immutable state object for adversarial sessions"""
    
    # Session metadata
    session_id: str
    start_time: datetime
    query: str
    config: AdversarialConfig
    
    # Conversation history
    rounds: List[AdversarialRound] = field(default_factory=list)
    
    # Status tracking
    status: SessionStatus = SessionStatus.RUNNING
    convergence_score: float = 0.0
    total_cost: float = 0.0
    
    def add_round(self, claude_response: str, codex_critique: str, 
                  score: float, cost: dict) -> 'AdversarialState':
        """Add new round and return new state object"""
        
        new_round = AdversarialRound(
            round_number=len(self.rounds) + 1,
            claude_response=claude_response,
            codex_critique=codex_critique,
            convergence_score=score,
            timestamp=datetime.now(),
            cost=cost
        )
        
        return replace(self, 
                      rounds=self.rounds + [new_round],
                      convergence_score=score,
                      total_cost=self.total_cost + cost['total'])

@dataclass  
class AdversarialRound:
    round_number: int
    claude_response: str
    codex_critique: str
    convergence_score: float
    timestamp: datetime
    cost: dict
```

## 🎯 Prompt Engineering

### Generator Prompts (Claude)
**Goal**: Generate high-quality, complete solutions

```python
CLAUDE_GENERATOR_TEMPLATE = """
Role: Senior Software Engineer & Solution Architect

Task: Provide a complete, production-ready solution for the following request.

Original Query: {query}

Previous Feedback to Address:
{previous_critique}

Guidelines:
1. Write complete, runnable code (not pseudocode)
2. Include proper error handling and edge cases
3. Follow best practices for the language/framework
4. Add clear comments explaining complex logic
5. Consider security, performance, and maintainability
6. Provide usage examples where applicable

Context:
- This solution will be reviewed by another AI system
- Focus on correctness and completeness
- Explain your design decisions

Please provide your solution:
"""
```

### Critic Prompts (Codex)
**Goal**: Find flaws and suggest improvements adversarially

```python
CODEX_CRITIC_TEMPLATE = """
Role: Senior Code Reviewer & Security Expert

Task: Conduct a thorough adversarial review of the proposed solution.

Original Request: {original_query}

Proposed Solution:
{solution}

Review Criteria:
1. Correctness - Does it solve the problem completely?
2. Security - Any vulnerabilities or security concerns?
3. Performance - Efficiency and scalability issues?
4. Maintainability - Code quality and readability?
5. Error Handling - Robust error handling and edge cases?
6. Best Practices - Following language/framework conventions?

Instructions:
- Be thorough and specific in identifying issues
- Provide concrete examples and suggestions
- If the solution is good, clearly state that
- Focus on actionable feedback
- Consider real-world production scenarios

Your review:
"""
```

## 🔀 Integration with claude-gemini-bridge

### Hook System Integration
```bash
# hooks/claude-adversarial.sh
#!/bin/bash

# Check if adversarial mode is enabled
if [[ "$ADVERSARIAL_MODE" == "true" ]]; then
    # Delegate to adversarial system
    python -m adversarial.main "$@"
else
    # Fall back to normal claude-gemini-bridge behavior
    exec "$ORIGINAL_COMMAND" "$@"
fi
```

### Configuration Integration
```bash
# Extends existing claude-gemini-bridge config
# ~/.claude-gemini-bridge/config/adversarial.conf

# Adversarial settings
ADVERSARIAL_MODE=true
MAX_ROUNDS=5
CONVERGENCE_THRESHOLD=0.8

# Cost controls
MAX_COST_PER_SESSION=2.00
WARN_COST_THRESHOLD=0.50

# Model selection
CLAUDE_MODEL=claude-sonnet-4
CODEX_MODEL=gpt-5-codex

# Integration settings
BRIDGE_INTEGRATION=true
FALLBACK_TO_CLAUDE=true
LOG_ADVERSARIAL_SESSIONS=true
```

## 🧪 Testing Strategy

### Unit Tests
```python
# tests/unit/test_convergence.py
def test_convergence_detection():
    detector = ConvergenceDetector()
    
    # High agreement
    solution = "def add(a, b): return a + b"
    critique = "Looks good, simple and correct solution."
    assert detector.calculate_score(solution, critique) > 0.8
    
    # High disagreement  
    critique = "Missing error handling, type validation, and edge case handling."
    assert detector.calculate_score(solution, critique) < 0.5
```

### Integration Tests
```python
# tests/integration/test_adversarial_session.py
@pytest.mark.asyncio
async def test_full_adversarial_session():
    session = AdversarialSession(
        query="Write a function to validate email addresses",
        max_rounds=3
    )
    
    result = await session.run()
    
    assert result.converged or result.rounds >= 3
    assert result.final_solution is not None
    assert result.total_cost > 0
```

### CLI Tests
```bash
# tests/test_adversarial.sh
#!/bin/bash

echo "Testing adversarial framework installation..."

# Test basic functionality
python -m adversarial.main --test
if [ $? -eq 0 ]; then
    echo "✅ Basic functionality test passed"
else
    echo "❌ Basic functionality test failed"
    exit 1
fi

# Test with mock query
result=$(python -m adversarial.main "write hello world function" --max-rounds 1)
if [[ $result == *"CONVERGED"* ]] || [[ $result == *"MAX_ROUNDS"* ]]; then
    echo "✅ Mock session test passed"
else
    echo "❌ Mock session test failed"
    exit 1
fi

echo "🎉 All tests passed!"
```

## 📊 Monitoring & Observability

### Logging Strategy
```python
# Structured logging for observability
import structlog

logger = structlog.get_logger()

# Session start
logger.info("adversarial_session_started", 
           session_id=session.id,
           query=session.query,
           max_rounds=session.max_rounds)

# Round completion
logger.info("adversarial_round_completed",
           session_id=session.id,
           round_number=round.number,
           convergence_score=round.score,
           claude_tokens=round.claude_usage.tokens,
           codex_tokens=round.codex_usage.tokens,
           round_cost=round.cost)

# Session completion
logger.info("adversarial_session_completed",
           session_id=session.id,
           status=session.status,
           total_rounds=len(session.rounds),
           final_score=session.convergence_score,
           total_cost=session.total_cost,
           duration=session.duration)
```

### Metrics Collection
```python
# Key metrics to track
METRICS = {
    'sessions_started': Counter(),
    'sessions_converged': Counter(),  
    'sessions_max_rounds': Counter(),
    'average_rounds_to_convergence': Histogram(),
    'total_cost_per_session': Histogram(),
    'convergence_score_distribution': Histogram(),
}
```

## 🚀 Performance Optimization

### Concurrent Processing
```python
# Process Claude and Codex in parallel where possible
async def parallel_review(solution: str, query: str):
    claude_task = claude_client.refine(solution)
    codex_task = codex_client.critique(solution)
    
    claude_result, codex_result = await asyncio.gather(
        claude_task, codex_task
    )
    
    return claude_result, codex_result
```

### Caching Strategy
```python
# Cache expensive operations
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_convergence_score(solution_hash: str, critique_hash: str) -> float:
    """Cache convergence calculations for identical inputs"""
    return detector.calculate_score(solution, critique)
```

### Token Optimization
```python
# Optimize token usage
def optimize_prompt_tokens(prompt: str, max_tokens: int = 2000) -> str:
    """Truncate or summarize prompts that exceed token limits"""
    if estimate_tokens(prompt) > max_tokens:
        # Use summarization or intelligent truncation
        return summarize_prompt(prompt, target_tokens=max_tokens * 0.8)
    return prompt
```

## 🔧 Extension Points

### Custom Critics
```python
# Allow pluggable critic implementations
class CustomCritic(BaseCritic):
    def critique(self, solution: str, context: dict) -> str:
        # Custom criticism logic
        pass

# Register custom critic
critic_registry.register('security-focused', SecurityCritic)
critic_registry.register('performance-focused', PerformanceCritic)
```

### Custom Convergence Detection
```python
# Pluggable convergence algorithms
class MLConvergenceDetector(BaseConvergenceDetector):
    def __init__(self):
        self.model = load_convergence_model()
    
    def calculate_score(self, solution: str, critique: str) -> float:
        features = extract_features(solution, critique)
        return self.model.predict(features)
```

### Multi-AI Support
```python
# Framework for adding more AI models
class GeminiCritic(BaseCritic):
    def critique(self, solution: str, context: dict) -> str:
        # Use Gemini's large context for massive code reviews
        pass

# Multi-critic consensus
class ConsensusDetector:
    def __init__(self, critics: List[BaseCritic]):
        self.critics = critics
    
    async def get_consensus(self, solution: str, query: str) -> ConsensusResult:
        """Get consensus from multiple critics"""
        critiques = await asyncio.gather(*[
            critic.critique(solution, {'query': query}) 
            for critic in self.critics
        ])
        
        # Analyze consensus vs disagreement
        scores = [self.calculate_agreement(c) for c in critiques]
        consensus_level = statistics.mean(scores)
        
        return ConsensusResult(
            critiques=critiques,
            consensus_level=consensus_level,
            majority_opinion=self.extract_majority_opinion(critiques)
        )
```

## 🎛️ Configuration Management

### Hierarchical Configuration
```python
# Configuration priority (highest to lowest):
# 1. CLI arguments
# 2. Environment variables  
# 3. Project .adversarial.conf
# 4. User ~/.adversarial/config/adversarial.conf
# 5. System /etc/adversarial/adversarial.conf
# 6. Default values

@dataclass
class AdversarialConfig:
    # Core behavior
    max_rounds: int = 5
    convergence_threshold: float = 0.8
    enable_cost_warnings: bool = True
    
    # Model selection
    claude_model: str = "claude-sonnet-4"
    codex_model: str = "gpt-5-codex"
    
    # Cost controls
    max_cost_per_session: float = 2.00
    warn_cost_threshold: float = 0.50
    
    # Performance
    parallel_processing: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 3600
    
    # Integration
    bridge_integration: bool = True
    fallback_to_claude: bool = True
    
    @classmethod
    def load(cls) -> 'AdversarialConfig':
        """Load configuration from multiple sources"""
        config = cls()
        
        # Load from files
        for config_file in cls.get_config_files():
            config = config.merge_from_file(config_file)
        
        # Override with environment variables
        config = config.merge_from_env()
        
        return config
    
    def merge_from_file(self, config_file: Path) -> 'AdversarialConfig':
        """Merge configuration from file"""
        if config_file.exists():
            with open(config_file) as f:
                file_config = toml.load(f)
            return replace(self, **file_config.get('adversarial', {}))
        return self
```

### Dynamic Configuration
```python
# Runtime configuration adjustment
class DynamicConfig:
    def __init__(self, base_config: AdversarialConfig):
        self.base_config = base_config
        self.runtime_adjustments = {}
    
    def adjust_for_query(self, query: str) -> AdversarialConfig:
        """Adjust configuration based on query characteristics"""
        
        # Simple queries need fewer rounds
        if self.is_simple_query(query):
            return replace(self.base_config, max_rounds=2)
        
        # Complex architecture questions need more rounds
        if self.is_architecture_query(query):
            return replace(self.base_config, 
                          max_rounds=7,
                          convergence_threshold=0.9)
        
        # Security-focused queries need stricter convergence
        if self.is_security_query(query):
            return replace(self.base_config,
                          convergence_threshold=0.95,
                          max_cost_per_session=5.00)
        
        return self.base_config
    
    def is_simple_query(self, query: str) -> bool:
        simple_keywords = ['hello world', 'basic', 'simple', 'quick']
        return any(keyword in query.lower() for keyword in simple_keywords)
```

## 🐛 Error Handling & Recovery

### Graceful Degradation
```python
class RobustAdversarialSession:
    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.fallback_strategies = [
            self.try_codex_fallback,
            self.try_claude_only,
            self.try_cached_response,
            self.try_manual_intervention
        ]
    
    async def run_with_recovery(self, query: str) -> AdversarialResult:
        """Run session with automatic error recovery"""
        
        try:
            return await self.run_normal_session(query)
        except CodexUnavailableError:
            logger.warning("Codex unavailable, trying fallbacks")
            return await self.run_fallback_strategies(query)
        except CostLimitExceededError:
            logger.warning("Cost limit exceeded, terminating gracefully")
            return self.create_partial_result(query)
        except Exception as e:
            logger.error("Unexpected error", error=str(e))
            return await self.emergency_fallback(query)
    
    async def try_codex_fallback(self, query: str) -> AdversarialResult:
        """Try alternative codex models"""
        alternative_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
        
        for model in alternative_models:
            try:
                logger.info(f"Trying fallback model: {model}")
                session = AdversarialSession(query, codex_model=model)
                return await session.run()
            except Exception:
                continue
        
        raise AllCodexModelsUnavailableError()
```

### Circuit Breaker Pattern
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == 'OPEN':
            if self.should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise CircuitBreakerOpenError()
        
        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
```

## 📈 Analytics & Insights

### Session Analytics
```python
class SessionAnalytics:
    def __init__(self, session: AdversarialSession):
        self.session = session
    
    def generate_insights(self) -> SessionInsights:
        """Generate insights from completed session"""
        
        return SessionInsights(
            efficiency_score=self.calculate_efficiency(),
            quality_improvement=self.measure_quality_improvement(),
            cost_effectiveness=self.calculate_cost_effectiveness(),
            convergence_pattern=self.analyze_convergence_pattern(),
            recommendations=self.generate_recommendations()
        )
    
    def calculate_efficiency(self) -> float:
        """How quickly did the session converge?"""
        optimal_rounds = 2  # Ideal scenario
        actual_rounds = len(self.session.rounds)
        return max(0.1, optimal_rounds / actual_rounds)
    
    def measure_quality_improvement(self) -> float:
        """Compare first vs final solution quality"""
        if len(self.session.rounds) < 2:
            return 0.0
        
        first_score = self.session.rounds[0].convergence_score
        final_score = self.session.rounds[-1].convergence_score
        return final_score - first_score
    
    def analyze_convergence_pattern(self) -> ConvergencePattern:
        """Analyze how convergence evolved over rounds"""
        scores = [r.convergence_score for r in self.session.rounds]
        
        if self.is_monotonic_improvement(scores):
            return ConvergencePattern.STEADY_IMPROVEMENT
        elif self.has_oscillation(scores):
            return ConvergencePattern.OSCILLATING
        elif self.is_plateau(scores):
            return ConvergencePattern.PLATEAU
        else:
            return ConvergencePattern.IRREGULAR
```

### System-wide Analytics
```python
class SystemAnalytics:
    def __init__(self, session_store: SessionStore):
        self.session_store = session_store
    
    def generate_daily_report(self, date: datetime) -> DailyReport:
        """Generate daily usage and performance report"""
        sessions = self.session_store.get_sessions_by_date(date)
        
        return DailyReport(
            total_sessions=len(sessions),
            convergence_rate=self.calculate_convergence_rate(sessions),
            average_rounds=self.calculate_average_rounds(sessions),
            total_cost=sum(s.total_cost for s in sessions),
            most_common_queries=self.extract_common_patterns(sessions),
            performance_metrics=self.calculate_performance_metrics(sessions)
        )
    
    def identify_optimization_opportunities(self) -> List[Optimization]:
        """Identify ways to improve the system"""
        recent_sessions = self.session_store.get_recent_sessions(days=7)
        
        optimizations = []
        
        # High-cost, low-convergence sessions
        expensive_failures = [s for s in recent_sessions 
                            if s.total_cost > 1.0 and not s.converged]
        if expensive_failures:
            optimizations.append(Optimization(
                type="cost_control",
                description="Add early termination for non-converging expensive sessions",
                impact="high",
                effort="medium"
            ))
        
        # Queries with consistent convergence patterns
        quick_convergers = self.find_quick_converging_patterns(recent_sessions)
        if quick_convergers:
            optimizations.append(Optimization(
                type="efficiency",
                description=f"Lower max_rounds for queries matching: {quick_convergers}",
                impact="medium", 
                effort="low"
            ))
        
        return optimizations
```

## 🔐 Security Considerations

### Input Sanitization
```python
class SecurePromptBuilder:
    def __init__(self):
        self.dangerous_patterns = [
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__',
            r'subprocess',
            r'os\.system',
            r'rm\s+-rf',
            r'DELETE\s+FROM',
            r'DROP\s+TABLE'
        ]
    
    def sanitize_query(self, query: str) -> str:
        """Sanitize user input before sending to AI models"""
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                raise SecurityError(f"Potentially dangerous pattern detected: {pattern}")
        
        # Limit query length
        if len(query) > 10000:
            raise SecurityError("Query too long")
        
        # Remove or escape sensitive characters
        sanitized = query.replace('`', '\\`').replace(', '\\)
        
        return sanitized
    
    def build_safe_prompt(self, template: str, **kwargs) -> str:
        """Build prompt with safe parameter substitution"""
        
        # Sanitize all inputs
        safe_kwargs = {k: self.sanitize_query(str(v)) for k, v in kwargs.items()}
        
        # Use safe substitution
        return template.format(**safe_kwargs)
```

### API Key Management
```python
class SecureCredentialManager:
    def __init__(self):
        self.key_rotation_schedule = {}
        self.usage_tracker = {}
    
    def get_api_key(self, service: str) -> str:
        """Get API key with automatic rotation and usage tracking"""
        
        # Check if key needs rotation
        if self.should_rotate_key(service):
            self.rotate_key(service)
        
        # Track usage
        self.track_usage(service)
        
        # Return key from secure storage
        return self.retrieve_from_secure_storage(service)
    
    def should_rotate_key(self, service: str) -> bool:
        """Check if key should be rotated based on usage/time"""
        last_rotation = self.key_rotation_schedule.get(service)
        if not last_rotation:
            return False
        
        # Rotate monthly or after high usage
        days_since_rotation = (datetime.now() - last_rotation).days
        usage_count = self.usage_tracker.get(service, 0)
        
        return days_since_rotation > 30 or usage_count > 10000
```

## 🚀 Deployment & Production

### Docker Configuration
```dockerfile
# Dockerfile for production deployment
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for CLI tools
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs

# Create application user
RUN useradd --create-home --shell /bin/bash adversarial
USER adversarial
WORKDIR /home/adversarial

# Install CLI tools
RUN npm install -g @anthropic-ai/claude-code openai-cli

# Copy and install Python dependencies
COPY --chown=adversarial:adversarial pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev

# Copy application code
COPY --chown=adversarial:adversarial src/ ./src/
COPY --chown=adversarial:adversarial config/ ./config/

# Setup entrypoint
COPY --chown=adversarial:adversarial docker/entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "-m", "adversarial.main", "--server"]
```

### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: project-atom
spec:
  replicas: 3
  selector:
    matchLabels:
      app: project-atom
  template:
    metadata:
      labels:
        app: project-atom
    spec:
      containers:
      - name: project-atom
        image: project-atom:latest
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-api-keys
              key: anthropic-key
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-api-keys
              key: openai-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Production Configuration
```python
# Production-specific settings
PRODUCTION_CONFIG = {
    "logging": {
        "level": "INFO",
        "format": "json",
        "output": "stdout"
    },
    "performance": {
        "max_concurrent_sessions": 10,
        "request_timeout": 300,
        "circuit_breaker_enabled": True
    },
    "security": {
        "api_key_rotation_enabled": True,
        "input_sanitization_strict": True,
        "rate_limiting_enabled": True
    },
    "monitoring": {
        "metrics_enabled": True,
        "tracing_enabled": True,
        "health_check_endpoint": "/health"
    }
}
```

## 📚 API Documentation

### REST API Endpoints
```python
# API server for web integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Project Atom API", version="1.0.0")

class AdversarialRequest(BaseModel):
    query: str
    max_rounds: int = 5
    convergence_threshold: float = 0.8
    config_overrides: dict = {}

class AdversarialResponse(BaseModel):
    session_id: str
    status: str
    converged: bool
    rounds: int
    final_solution: str
    convergence_score: float
    total_cost: float

@app.post("/adversarial/sessions", response_model=AdversarialResponse)
async def create_adversarial_session(request: AdversarialRequest):
    """Start new adversarial review session"""
    
    try:
        session = AdversarialSession(
            query=request.query,
            max_rounds=request.max_rounds,
            convergence_threshold=request.convergence_threshold
        )
        
        result = await session.run()
        
        return AdversarialResponse(
            session_id=result.session_id,
            status=result.status,
            converged=result.converged,
            rounds=len(result.rounds),
            final_solution=result.final_solution,
            convergence_score=result.convergence_score,
            total_cost=result.total_cost
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/adversarial/sessions/{session_id}")
async def get_session_details(session_id: str):
    """Get detailed session information"""
    # Implementation here
    pass

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
```

## 🎓 Best Practices

### Code Quality
```python
# Use type hints throughout
from typing import Optional, List, Dict, Union, Protocol

class CriticProtocol(Protocol):
    async def critique(self, solution: str, context: dict) -> str: ...

# Comprehensive error handling
class AdversarialError(Exception):
    """Base exception for adversarial system"""
    pass

class ConvergenceTimeoutError(AdversarialError):
    """Raised when session fails to converge within max rounds"""
    pass

class CostLimitExceededError(AdversarialError):
    """Raised when session exceeds cost limits"""
    pass

# Proper logging
import structlog
logger = structlog.get_logger(__name__)

# Use context managers for resources
from contextlib import asynccontextmanager

@asynccontextmanager
async def adversarial_session_context(config: AdversarialConfig):
    """Context manager for adversarial sessions"""
    session = None
    try:
        session = AdversarialSession(config)
        yield session
    finally:
        if session:
            await session.cleanup()
```

### Performance Guidelines
```python
# Async/await best practices
async def process_large_codebase(files: List[str]) -> Dict[str, AdversarialResult]:
    """Process multiple files concurrently"""
    
    # Limit concurrency to avoid overwhelming APIs
    semaphore = asyncio.Semaphore(5)
    
    async def process_file(file_path: str) -> Tuple[str, AdversarialResult]:
        async with semaphore:
            query = f"Review and improve {file_path}"
            session = AdversarialSession(query)
            result = await session.run()
            return file_path, result
    
    # Process all files concurrently
    results = await asyncio.gather(*[
        process_file(file_path) for file_path in files
    ])
    
    return dict(results)

# Memory management for large sessions
def optimize_memory_usage(session: AdversarialSession):
    """Optimize memory usage for long-running sessions"""
    
    # Keep only last N rounds in memory
    if len(session.rounds) > 10:
        session.rounds = session.rounds[-5:]  # Keep last 5 rounds
    
    # Clear large response text after processing
    for round in session.rounds[:-1]:  # Keep current round
        round.claude_response = "[truncated for memory]"
        round.codex_critique = "[truncated for memory]"
```

## 📖 Documentation Standards

### Code Documentation
```python
def calculate_convergence_score(solution: str, critique: str) -> float:
    """Calculate convergence score between solution and critique.
    
    Args:
        solution: The proposed solution text from Claude
        critique: The critique text from Codex
        
    Returns:
        Float between 0.0 (complete disagreement) and 1.0 (full agreement)
        
    Raises:
        ValueError: If solution or critique is empty
        
    Examples:
        >>> score = calculate_convergence_score("good code", "looks great!")
        >>> assert score > 0.8
        
        >>> score = calculate_convergence_score("bad code", "many issues found")
        >>> assert score < 0.5
    """
```

### Configuration Documentation
```yaml
# config/schema.yaml - Configuration schema documentation
adversarial:
  type: object
  properties:
    max_rounds:
      type: integer
      minimum: 1
      maximum: 20
      default: 5
      description: "Maximum number of adversarial rounds before termination"
      
    convergence_threshold:
      type: number
      minimum: 0.0
      maximum: 1.0
      default: 0.8
      description: "Minimum convergence score required for session completion"
      
    claude_model:
      type: string
      enum: ["claude-sonnet-4", "claude-opus-4"]
      default: "claude-sonnet-4"
      description: "Claude model to use for solution generation"
```

This completes the comprehensive developer guide for Project Atom. The documentation covers everything from architecture and implementation details to deployment and best practices, providing a solid foundation for both current development and future contributions to the adversarial code review system.
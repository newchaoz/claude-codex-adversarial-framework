"""
Claude Client - The "Generator" in our adversarial system
With rate limits and subscription plan information for 2025
Based on official docs and current rate limit information
"""

import os
from typing import List, Dict, Optional
from anthropic import Anthropic


class ClaudeClient:
    """Claude client with rate limits and subscription information"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = Anthropic(api_key=self.api_key)
        
        # Claude 4 models only (3.x deprecated)
        self.available_models = {
            "opus-4.1": "claude-opus-4-1-20250805",       # Highest intelligence  
            "sonnet-4": "claude-sonnet-4-20250514",       # Balanced performance
            "sonnet-3.7": "claude-3-7-sonnet-20250219",   # Transition model
        }
        
        # Default to Sonnet 4
        self.model = model or self.available_models["sonnet-4"]
        
        # Pricing (per 1M tokens) - API pay-per-use
        self.api_pricing = {
            "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},      # Opus 4.1
            "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},       # Sonnet 4
            "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},     # Sonnet 3.7
        }
        
        # Batch pricing (50% discount for non-urgent processing)
        self.batch_pricing = {
            "claude-opus-4-1-20250805": {"input": 7.50, "output": 37.50},
            "claude-sonnet-4-20250514": {"input": 1.50, "output": 7.50},
            "claude-3-7-sonnet-20250219": {"input": 1.50, "output": 7.50},
        }
        
        # Claude subscription plans (rate limited, not cost limited)
        self.subscription_plans = {
            "free": {
                "cost": "$0/month",
                "usage": "Limited messages, rate limited",
                "models": ["sonnet-4"],
                "reset_window": "Not specified",
                "best_for": "Trying out Claude"
            },
            "pro": {
                "cost": "$20/month", 
                "usage": "~45 messages every 5 hours (5x more than free)",
                "weekly_limit": "40-80 hours of Claude Code usage",
                "models": ["sonnet-4", "sonnet-3.7"],
                "reset_window": "5 hours + weekly caps",
                "context_window": "200K tokens (1M for Sonnet 4)",
                "best_for": "Regular users, light coding"
            },
            "max_5x": {
                "cost": "$100/month",
                "usage": "~225 messages every 5 hours",
                "weekly_limit": "140-280 hours Sonnet 4, 15-35 hours Opus 4",
                "models": ["sonnet-4", "opus-4.1", "sonnet-3.7"],
                "reset_window": "5 hours + weekly caps",
                "best_for": "Moderate usage, larger repositories"
            },
            "max_20x": {
                "cost": "$200/month",
                "usage": "~900 messages every 5 hours", 
                "weekly_limit": "240-480 hours Sonnet 4, 24-40 hours Opus 4",
                "models": ["sonnet-4", "opus-4.1", "sonnet-3.7"],
                "reset_window": "5 hours + weekly caps",
                "best_for": "Power users, intensive coding"
            }
        }
        
        # API rate limits (for pay-per-token users)
        self.api_rate_limits = {
            "tier_1": {
                "spending_requirement": "$5/month",
                "requests_per_minute": 50,
                "input_tokens_per_minute": 50000,
                "output_tokens_per_minute": 10000
            },
            "tier_2": {
                "spending_requirement": "$100/month",
                "requests_per_minute": 100,
                "input_tokens_per_minute": 100000,
                "output_tokens_per_minute": 20000
            },
            "tier_3": {
                "spending_requirement": "$500/month", 
                "requests_per_minute": 200,
                "input_tokens_per_minute": 200000,
                "output_tokens_per_minute": 40000
            },
            "tier_4": {
                "spending_requirement": "$1000/month",
                "requests_per_minute": 400,
                "input_tokens_per_minute": 400000,
                "output_tokens_per_minute": 80000
            }
        }
        
        # Model capabilities
        self.model_capabilities = {
            "opus-4.1": {
                "use_case": "Highest intelligence and reasoning",
                "best_for": "Multi-agent frameworks, complex refactoring",
                "context_window": "200K tokens",
                "resource_usage": "5x more intensive than Sonnet 4"
            },
            "sonnet-4": {
                "use_case": "Balance of intelligence and speed", 
                "best_for": "Complex code generation, agentic loops",
                "context_window": "1M tokens",
                "resource_usage": "Standard"
            },
            "sonnet-3.7": {
                "use_case": "Transition model from Claude 3.5",
                "best_for": "Migration from 3.x models",
                "context_window": "200K tokens",
                "resource_usage": "Similar to Sonnet 4"
            }
        }
        
        # User type recommendations
        self.presets = {
            "subscription_pro": {
                "model": "sonnet-4",
                "billing": "subscription",
                "description": "Pro subscriber - rate limited but not cost limited"
            },
            "subscription_max": {
                "model": "opus-4.1", 
                "billing": "subscription",
                "description": "Max subscriber - can use Opus 4.1 without per-token costs"
            },
            "api_budget": {
                "model": "sonnet-4",
                "billing": "api",
                "description": "API user prioritizing cost ($3/$15 per 1M tokens)"
            },
            "api_performance": {
                "model": "opus-4.1",
                "billing": "api", 
                "description": "API user prioritizing quality ($15/$75 per 1M tokens)"
            },
            "adversarial_generator": {
                "model": "sonnet-4",
                "billing": "flexible",
                "description": "Recommended for adversarial framework - good balance"
            }
        }
    
    def show_pricing_comparison(self):
        """Compare subscription vs API pricing"""
        print("Claude Pricing Comparison (2025)")
        print("="*50)
        
        print("\n📋 SUBSCRIPTION PLANS (Rate Limited):")
        for plan_name, plan_info in self.subscription_plans.items():
            print(f"  {plan_name.upper()}: {plan_info['cost']}")
            print(f"    Usage: {plan_info['usage']}")
            if 'weekly_limit' in plan_info:
                print(f"    Weekly: {plan_info['weekly_limit']}")
            print(f"    Best for: {plan_info['best_for']}")
            print()
        
        print("💰 API PRICING (Pay Per Token):")
        for model_key, model_id in self.available_models.items():
            pricing = self.api_pricing.get(model_id, {"input": "?", "output": "?"})
            batch_pricing = self.batch_pricing.get(model_id, {"input": "?", "output": "?"})
            print(f"  {model_key.upper()}: ${pricing['input']}/${pricing['output']} per 1M tokens")
            print(f"    Batch (50% off): ${batch_pricing['input']}/${batch_pricing['output']} per 1M tokens")
        
        print("\n🔄 RATE LIMITS (API Users):")
        for tier, limits in self.api_rate_limits.items():
            print(f"  {tier.upper()}: ${limits['spending_requirement']}/month minimum")
            print(f"    {limits['requests_per_minute']} RPM, {limits['input_tokens_per_minute']:,} input TPM")
    
    def show_user_recommendations(self):
        """Show recommendations for different user types"""
        print("User Type Recommendations")
        print("="*40)
        
        subscription_users = ["subscription_pro", "subscription_max"]
        api_users = ["api_budget", "api_performance"]
        
        print("\n👤 SUBSCRIPTION USERS (You):")
        for user_type in subscription_users:
            preset = self.presets[user_type]
            model_key = preset["model"]
            capabilities = self.model_capabilities[model_key]
            
            print(f"  {user_type.replace('_', ' ').title()}:")
            print(f"    Model: {model_key} ({capabilities['context_window']})")
            print(f"    Billing: Rate limited (not cost limited)")
            print(f"    Best for: {capabilities['best_for']}")
            print()
        
        print("💳 API USERS (Pay Per Token):")
        for user_type in api_users:
            preset = self.presets[user_type]
            model_key = preset["model"] 
            model_id = self.available_models[model_key]
            pricing = self.api_pricing[model_id]
            capabilities = self.model_capabilities[model_key]
            
            print(f"  {user_type.replace('_', ' ').title()}:")
            print(f"    Model: {model_key} (${pricing['input']}/${pricing['output']} per 1M tokens)")
            print(f"    Best for: {capabilities['best_for']}")
            print()
    
    @classmethod  
    def create_for_user_type(cls, user_type: str, api_key: Optional[str] = None):
        """Factory method for different user types"""
        client = cls(api_key=api_key)
        preset = client.presets.get(user_type)
        if preset:
            model_key = preset["model"]
            client.model = client.available_models[model_key]
        return client
    
    def set_model(self, model_key: str):
        """Change the model being used"""
        if model_key in self.available_models:
            self.model = self.available_models[model_key]
        else:
            available = list(self.available_models.keys())
            raise ValueError(f"Unknown model key: {model_key}. Available: {available}")
    
    def generate_initial_solution(self, query: str) -> Dict[str, str]:
        """Generate initial code solution"""
        prompt = self._build_initial_prompt(query)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            result = self._parse_claude_response(content)
            
            # Add metadata
            result['cost_estimate'] = self._estimate_cost(prompt, content)
            result['model_used'] = self.model
            result['model_info'] = self._get_current_model_info()
            
            return result
            
        except Exception as e:
            return {
                "solution": f"Error generating solution: {str(e)}",
                "reasoning": "Failed to connect to Claude API",
                "cost_estimate": 0.0,
                "model_used": self.model,
                "model_info": self._get_current_model_info()
            }
    
    def refine_solution(self, 
                       original_query: str, 
                       previous_solution: str, 
                       openai_critique: str) -> Dict[str, str]:
        """Refine solution based on critique"""
        prompt = self._build_refinement_prompt(
            original_query, previous_solution, openai_critique
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            result = self._parse_claude_response(content)
            
            # Add metadata
            result['cost_estimate'] = self._estimate_cost(prompt, content)
            result['model_used'] = self.model
            result['model_info'] = self._get_current_model_info()
            
            return result
            
        except Exception as e:
            return {
                "solution": f"Error refining solution: {str(e)}",
                "reasoning": "Failed to connect to Claude API", 
                "cost_estimate": 0.0,
                "model_used": self.model,
                "model_info": self._get_current_model_info()
            }
    
    def _get_current_model_info(self) -> Dict:
        """Get information about current model"""
        model_key = None
        for key, model_id in self.available_models.items():
            if model_id == self.model:
                model_key = key
                break
        
        if model_key and model_key in self.model_capabilities:
            capabilities = self.model_capabilities[model_key]
            api_pricing = self.api_pricing.get(self.model, {"input": "unknown", "output": "unknown"})
            
            return {
                "model_key": model_key,
                "model_id": self.model,
                "use_case": capabilities["use_case"],
                "context_window": capabilities["context_window"],
                "resource_usage": capabilities["resource_usage"],
                "api_pricing": f"${api_pricing['input']}/${api_pricing['output']} per 1M tokens"
            }
        
        return {"model_id": self.model, "info": "Unknown model"}
    
    def _estimate_cost(self, prompt: str, response: str) -> float:
        """Estimate API cost (for API users only)"""
        if self.model not in self.api_pricing:
            return 0.0
            
        input_tokens = self.estimate_tokens(prompt)
        output_tokens = self.estimate_tokens(response)
        
        pricing = self.api_pricing[self.model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return round(input_cost + output_cost, 4)
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count"""
        return int(len(text.split()) * 1.3)
    
    def _build_initial_prompt(self, query: str) -> str:
        """Build prompt for initial solution generation"""
        return f"""You are an expert software engineer using Claude 4. Please provide a complete, high-quality solution to this coding request:

QUERY: {query}

Please structure your response as:

## SOLUTION
[Your complete code solution here]

## REASONING
[Explain your approach, design decisions, and why this solution is effective]

Focus on:
- Correct, working code
- Best practices and clean code principles
- Performance considerations
- Error handling
- Security considerations where relevant"""
    
    def _build_refinement_prompt(self, original_query: str, previous_solution: str, critique: str) -> str:
        """Build prompt for solution refinement"""
        return f"""Please improve your solution based on the expert critique provided.

ORIGINAL QUERY: {original_query}

YOUR PREVIOUS SOLUTION:
{previous_solution}

EXPERT CRITIQUE:
{critique}

Please provide an improved solution that addresses the critique points.

Structure your response as:

## REFINED SOLUTION
[Your improved code solution here]

## REASONING
[Explain what changes you made and why, addressing each point in the critique]"""
    
    def _parse_claude_response(self, content: str) -> Dict[str, str]:
        """Parse Claude's response into solution and reasoning"""
        lines = content.strip().split('\n')
        solution_lines = []
        reasoning_lines = []
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith('## solution') or line_lower.startswith('## refined solution'):
                current_section = 'solution'
                continue
            elif line_lower.startswith('## reasoning'):
                current_section = 'reasoning'
                continue
            
            if current_section == 'solution':
                solution_lines.append(line)
            elif current_section == 'reasoning':
                reasoning_lines.append(line)
        
        if not solution_lines and not reasoning_lines:
            solution_lines = lines
            reasoning_lines = ["Solution provided without explicit reasoning section."]
        
        return {
            "solution": '\n'.join(solution_lines).strip(),
            "reasoning": '\n'.join(reasoning_lines).strip()
        }


def test_claude_client():
    """Test Claude client and show pricing information"""
    try:
        print("Claude 4 Client with Rate Limits & Pricing Info")
        print("="*60)
        
        client = ClaudeClient()
        
        # Show pricing comparison
        client.show_pricing_comparison()
        print()
        
        # Show user recommendations  
        client.show_user_recommendations()
        print()
        
        # Test a generation
        result = client.generate_initial_solution("Write a Python function for quicksort")
        model_info = result['model_info']
        
        print("Test Results:")
        print(f"Model: {model_info['model_key']} ({model_info['context_window']})")
        print(f"Resource usage: {model_info['resource_usage']}")
        print(f"API cost estimate: ${result['cost_estimate']} (if using API)")
        print(f"Solution length: {len(result['solution'])} characters")
        
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    test_claude_client()

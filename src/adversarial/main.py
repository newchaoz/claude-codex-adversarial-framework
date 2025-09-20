#!/usr/bin/env python3
"""
Claude-Codex Adversarial Framework - Minimal Working Version
"""

import os
import sys
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from dotenv import load_dotenv
    dotenv_available = True
except ImportError:
    dotenv_available = False

def main():
    # Load environment variables if available
    if dotenv_available:
        load_dotenv()
    
    # Check API keys
    claude_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not claude_key or not openai_key:
        print("❌ Missing API keys!")
        print("1. Copy .env.example to .env")
        print("2. Add your ANTHROPIC_API_KEY and OPENAI_API_KEY")
        print("3. Get Claude key from: console.anthropic.com")
        print("4. Get OpenAI key from: platform.openai.com")
        return 1
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.adversarial 'your code question'")
        print("\nExample:")
        print("  python -m src.adversarial 'optimize this database query'")
        return 1
    
    query = " ".join(sys.argv[1:])
    
    print(f"🥊 Claude-Codex Adversarial Framework v0.1")
    print(f"📝 Query: {query}")
    print(f"🤖 Claude API: {'✅' if claude_key and claude_key.startswith('sk-ant-') else '❌'}")
    print(f"🔍 OpenAI API: {'✅' if openai_key and openai_key.startswith('sk-') else '❌'}")
    print("")
    print("🏗️  Full adversarial system coming soon!")
    print("For now, this validates your setup is working.")
    print("")
    print("Next: Claude will generate → Codex will critique → Repeat until agreement")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

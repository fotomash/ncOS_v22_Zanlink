#!/usr/bin/env python3
"""
ncOS v23 Session Example
Shows how to use the system in a trading session
"""

import asyncio
import json
from datetime import datetime
from ncos_v23_unified_engine import initialize_system

async def trading_session():
    """Example trading session"""

    print("🚀 Starting ncOS v23 Trading Session")
    print("=" * 50)

    # 1. Initialize system
    print("\n1️⃣ Initializing system...")
    system = await initialize_system()
    print(f"✅ System ready: {system['version']}")

    # 2. Display menu
    print("\n2️⃣ Available options:")
    menu = system["menu"].display()
    for key, option in menu["options"].items():
        print(f"  {key}: {option['title']}")

    # 3. Load market data
    print("\n3️⃣ Loading market data...")
    market_data = {
        "pair": "XAUUSD",
        "timeframe": "M5",
        "data_file": "XAUUSD_ticks.csv",  # External data
        "session": "london"  # Optional session filter
    }

    # 4. Run analysis
    print("\n4️⃣ Running multi-agent analysis...")
    signal = await system["orchestrator"].process_market_data(market_data)

    # 5. Display results
    if signal:
        print("\n✅ TRADE SIGNAL GENERATED!")
        print(f"{'='*50}")
        print(f"Pair: {signal.pair}")
        print(f"Action: {signal.action.upper()}")
        print(f"Entry: {signal.entry:.2f}")
        print(f"Stop Loss: {signal.stop_loss:.2f} ({abs(signal.entry - signal.stop_loss):.2f} points)")
        print(f"Take Profits:")
        for i, tp in enumerate(signal.take_profits, 1):
            profit_points = abs(tp - signal.entry)
            print(f"  TP{i}: {tp:.2f} (+{profit_points:.2f} points)")
        print(f"Confidence: {signal.confidence:.1%}")
        print(f"Strategies: {', '.join(signal.strategies_aligned)}")
        print(f"\nReasoning:")
        for strategy, reason in signal.reasoning.items():
            print(f"  • {strategy}: {reason}")
    else:
        print("\n❌ No trade setup found")
        print("Market conditions do not meet confluence requirements")

    # 6. Check agent status
    print("\n6️⃣ Agent Status:")
    for name, agent in system["orchestrator"].agents.items():
        vectors = len(agent.vector_store.vectors)
        print(f"  • {name}: Active ({vectors} vectors stored)")

    # 7. Session summary
    print("\n7️⃣ Session Summary:")
    print(f"Session ID: {system['orchestrator'].session_id}")
    print(f"Vectors stored: {len(system['orchestrator'].vector_store.vectors)}")
    print(f"Analysis complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # Run the session
    asyncio.run(trading_session())

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import openai  # Change this
import os
import time
import json
from datetime import datetime

app = FastAPI(title="Trade Guardian", version="1.0")

# Use OpenAI instead
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

trade_history = []

class TradeSetup(BaseModel):
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    setup_type: str
    timestamp: str
    account_balance: Optional[float] = 25000.0
    daily_pnl: Optional[float] = 0.0
    vix: Optional[float] = 15.0
    consecutive_losses: Optional[int] = 0

class TradeDecision(BaseModel):
    proceed: bool
    confidence: int
    reason: str
    size_multiplier: float
    suggested_stop: Optional[float] = None

@app.post("/analyze-trade")
async def analyze_trade(setup: TradeSetup):
    start_time = time.time()
    
    recent_trades = [t for t in trade_history if t['symbol'] == setup.symbol][-20:]
    win_rate = 50
    if recent_trades:
        wins = sum(1 for t in recent_trades if t.get('win', False))
        win_rate = (wins / len(recent_trades)) * 100
    
    prompt = f"""You are an elite futures risk manager. Return JSON only.

TRADE: {setup.symbol} {setup.direction} @ {setup.entry_price}
VIX: {setup.vix} | Consecutive Losses: {setup.consecutive_losses}
Win Rate (20 samples): {win_rate:.1f}%

RULES:
- If consecutive_losses >= 2 → BLOCK or 0.5x
- If daily_pnl < -1000 → BLOCK
- If win_rate < 35% → BLOCK

Return JSON: {{"proceed": bool, "confidence": 0-100, "reason": "text", "size_multiplier": 0.5/1.0/1.5/2.0}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=200
        )
        
        decision = json.loads(response.choices[0].message.content)
        decision.setdefault('proceed', True)
        decision.setdefault('confidence', 50)
        decision.setdefault('reason', 'No pattern')
        decision.setdefault('size_multiplier', 1.0)
        
        trade_history.append({
            'trade_id': setup.trade_id,
            'symbol': setup.symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'decision': decision
        })
        
        latency = int((time.time() - start_time) * 1000)
        print(f"✅ {setup.trade_id}: {'APPROVED' if decision['proceed'] else 'BLOCKED'} in {latency}ms")
        

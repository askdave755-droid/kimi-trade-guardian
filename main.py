from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import anthropic
import os
import time
from datetime import datetime

app = FastAPI(title="Claude Trade Guardian", version="1.0")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# In-memory storage (resets on deploy, but works immediately)
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
    
    # Simple in-memory analysis (no DB needed)
    recent_trades = [t for t in trade_history if t['symbol'] == setup.symbol][-20:]
    win_rate = 50
    if recent_trades:
        wins = sum(1 for t in recent_trades if t.get('win', False))
        win_rate = (wins / len(recent_trades)) * 100
    
    prompt = f"""You are an elite futures risk manager. JSON only.

TRADE: {setup.symbol} {setup.direction} @ {setup.entry_price}
VIX: {setup.vix} | Consecutive Losses: {setup.consecutive_losses}
Win Rate (20 samples): {win_rate:.1f}%

RULES:
1. If consecutive_losses >= 2 → BLOCK or 0.5x size
2. If daily_pnl < -1000 → BLOCK
3. If win_rate < 35% → BLOCK

Return: {{"proceed": bool, "confidence": 0-100, "reason": "text", "size_multiplier": 0.5/1.0/1.5/2.0, "suggested_stop": null}}"""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=500,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
            
        import json
        decision = json.loads(response_text.strip())
        
        decision.setdefault('proceed', True)
        decision.setdefault('confidence', 50)
        decision.setdefault('reason', 'No pattern')
        decision.setdefault('size_multiplier', 1.0)
        
        # Store in memory
        trade_history.append({
            'trade_id': setup.trade_id,
            'symbol': setup.symbol,
            'direction': setup.direction,
            'timestamp': datetime.utcnow().isoformat(),
            'decision': decision
        })
        
        latency = int((time.time() - start_time) * 1000)
        print(f"✅ {setup.trade_id}: {'APPROVED' if decision['proceed'] else 'BLOCKED'} in {latency}ms")
        
        return TradeDecision(**decision)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return TradeDecision(proceed=True, confidence=0, reason=f"Error: {str(e)[:50]}", size_multiplier=1.0)

@app.get("/health")
def health():
    return {"status": "ok", "trades_stored": len(trade_history), "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import anthropic
import os
import time
import asyncpg
import json
from datetime import datetime
import asyncio

app = FastAPI(title="Claude Trade Guardian", version="1.0")

# Initialize Claude
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Database pool (asyncpg)
db_pool = None

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(
        os.getenv("DATABASE_URL"),
        min_size=2,
        max_size=10
    )

@app.on_event("shutdown")
async def shutdown():
    await db_pool.close()

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
    risk_warning: Optional[str] = None

async def get_recent_performance(symbol: str, setup_type: str):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT pnl, win, timestamp, session_time
            FROM trade_outcomes 
            WHERE symbol = $1 
            AND setup_type = $2
            AND timestamp > NOW() - INTERVAL '30 days'
            ORDER BY timestamp DESC 
            LIMIT 20
        """, symbol, setup_type)
        
        if not rows:
            return {"avg_pnl": 0, "win_rate": 50, "recent_trades": []}
        
        wins = sum(1 for r in rows if r['win'])
        return {
            "avg_pnl": sum(r['pnl'] for r in rows) / len(rows),
            "win_rate": (wins / len(rows)) * 100,
            "recent_trades": [dict(r) for r in rows[:5]]
        }

async def check_todays_stats():
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT 
                SUM(pnl) as daily_pnl,
                COUNT(*) as trade_count,
                SUM(CASE WHEN win THEN 1 ELSE 0 END) as wins
            FROM trade_outcomes 
            WHERE DATE(timestamp) = CURRENT_DATE
        """)
        return row or {"daily_pnl": 0, "trade_count": 0, "wins": 0}

@app.post("/analyze-trade", response_model=TradeDecision)
async def analyze_trade(setup: TradeSetup):
    start_time = time.time()
    
    try:
        # Fetch data concurrently
        perf_data, today_stats = await asyncio.gather(
            get_recent_performance(setup.symbol, setup.setup_type),
            check_todays_stats()
        )
        
        time_str = "Unknown"
        if 'T' in setup.timestamp:
            dt = datetime.fromisoformat(setup.timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime('%H:%M')
        
        prompt = f"""You are an elite futures risk manager analyzing a trade setup in real-time. Respond ONLY with valid JSON.

TRADE SETUP:
- Symbol: {setup.symbol}
- Direction: {setup.direction}
- Entry: {setup.entry_price}, Stop: {setup.stop_loss}, Target: {setup.take_profit}
- Risk/Reward: 1:{abs((setup.take_profit - setup.entry_price) / (setup.entry_price - setup.stop_loss)):.1f}
- Setup Type: {setup.setup_type}
- Time (ET): {time_str}
- VIX: {setup.vix}
- Consecutive Losses Today: {setup.consecutive_losses}

PERFORMANCE CONTEXT (Last 30 Days):
- This setup avg PnL: ${perf_data['avg_pnl']:.2f}
- This setup win rate: {perf_data['win_rate']:.1f}%
- Today's PnL so far: ${today_stats['daily_pnl'] or 0:.2f}
- Today's trades: {today_stats['trade_count'] or 0}

RECENT TRADES (Last 5):
{json.dumps(perf_data['recent_trades'], indent=2, default=str)}

DECISION RULES:
1. If consecutive_losses >= 2, recommend HALF SIZE or REJECT
2. If daily_pnl < -1000, REJECT (protect capital)
3. If win_rate < 35% for this setup, REJECT or require 1:3 R/R
4. If time is after 14:00 ET and not trend-following, REJECT
5. If VIX > 20 and stop < 10 ticks, REJECT (volatility too high)

Return EXACTLY this JSON:
{{
    "proceed": true or false,
    "confidence": 0-100,
    "reason": "Brief explanation",
    "size_multiplier": 0.5, 1.0, 1.5, or 2.0,
    "suggested_stop": null or float,
    "risk_warning": "Optional warning or null"
}}"""

        # Call Claude
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
            
        decision = json.loads(response_text.strip())
        
        # Ensure fields
        decision.setdefault('proceed', True)
        decision.setdefault('confidence', 50)
        decision.setdefault('reason', 'No pattern detected')
        decision.setdefault('size_multiplier', 1.0)
        
        # Log decision
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO ai_trade_decisions 
                (trade_id, timestamp, setup_json, decision_json, latency_ms, model_used)
                VALUES ($1, NOW(), $2, $3, $4, 'claude-3.5-sonnet')
            """, 
                setup.trade_id,
                json.dumps(setup.dict()),
                json.dumps(decision),
                int((time.time() - start_time) * 1000)
            )
        
        latency = int((time.time() - start_time) * 1000)
        print(f"✅ {setup.trade_id}: {'APPROVED' if decision['proceed'] else 'BLOCKED'} ({decision['confidence']}%) in {latency}ms")
        
        return TradeDecision(**decision)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return TradeDecision(
            proceed=True,
            confidence=0,
            reason=f"AI Error: {str(e)[:50]}",
            size_multiplier=1.0,
            risk_warning="System error - manual discretion"
        )

@app.get("/health")
async def health():
    return {"status": "ok", "model": "claude-3.5-sonnet", "timestamp": datetime.utcnow().isoformat()}

@app.post("/init-db")
async def init_database():
    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_trade_decisions (
                id SERIAL PRIMARY KEY,
                trade_id VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                setup_json JSONB,
                decision_json JSONB,
                latency_ms INTEGER,
                model_used VARCHAR(20)
            );
            
            CREATE INDEX IF NOT EXISTS idx_trade_id ON ai_trade_decisions(trade_id);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON ai_trade_decisions(timestamp);
        """)
    return {"message": "Database initialized"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

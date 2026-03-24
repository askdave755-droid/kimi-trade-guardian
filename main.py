from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import anthropic
import os
import time
import pg8000
from datetime import datetime
import asyncio
from contextlib import contextmanager

app = FastAPI(title="Claude Trade Guardian", version="1.0")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# pg8000 connection (pure Python, no libpq needed)
def get_db_conn():
    return pg8000.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

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

@app.on_event("startup")
async def startup():
    # Test connection and create table
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_trade_decisions (
            id SERIAL PRIMARY KEY,
            trade_id VARCHAR(50),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            setup_json JSONB,
            decision_json JSONB,
            latency_ms INTEGER,
            model_used VARCHAR(20)
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def get_recent_performance(symbol: str, setup_type: str):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT pnl, win, timestamp, session_time
        FROM trade_outcomes 
        WHERE symbol = %s 
        AND setup_type = %s
        AND timestamp > NOW() - INTERVAL '30 days'
        ORDER BY timestamp DESC 
        LIMIT 20
    """, (symbol, setup_type))
    
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    if not rows:
        return {"avg_pnl": 0, "win_rate": 50, "recent_trades": []}
    
    wins = sum(1 for r in rows if r[1])  # win is 2nd column
    return {
        "avg_pnl": sum(r[0] for r in rows) / len(rows),
        "win_rate": (wins / len(rows)) * 100,
        "recent_trades": rows[:5]
    }

def check_todays_stats():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            SUM(pnl) as daily_pnl,
            COUNT(*) as trade_count,
            SUM(CASE WHEN win THEN 1 ELSE 0 END) as wins
        FROM trade_outcomes 
        WHERE DATE(timestamp) = CURRENT_DATE
    """)
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row or (0, 0, 0)

@app.post("/analyze-trade", response_model=TradeDecision)
async def analyze_trade(setup: TradeSetup):
    start_time = time.time()
    
    try:
        # Get data (run in thread to not block)
        loop = asyncio.get_event_loop()
        perf_data = await loop.run_in_executor(None, get_recent_performance, setup.symbol, setup.setup_type)
        today_stats = await loop.run_in_executor(None, check_todays_stats)
        
        time_str = "Unknown"
        if 'T' in setup.timestamp:
            dt = datetime.fromisoformat(setup.timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime('%H:%M')
        
        prompt = f"""You are an elite futures risk manager. Respond with JSON only.

TRADE:
- Symbol: {setup.symbol}, Direction: {setup.direction}
- Entry: {setup.entry_price}, Stop: {setup.stop_loss}, Target: {setup.take_profit}
- R:R: 1:{abs((setup.take_profit - setup.entry_price) / (setup.entry_price - setup.stop_loss)):.1f}
- Time: {time_str}, VIX: {setup.vix}
- Consecutive Losses: {setup.consecutive_losses}

HISTORY:
- Setup win rate: {perf_data['win_rate']:.1f}%
- Today's PnL: ${today_stats[0] or 0:.2f}

RULES:
1. If consecutive_losses >= 2, BLOCK or HALF SIZE
2. If daily_pnl < -1000, BLOCK
3. If win_rate < 35%, BLOCK

Return: {{"proceed": bool, "confidence": 0-100, "reason": "text", "size_multiplier": 0.5/1.0/1.5/2.0, "suggested_stop": null or float}}"""

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
            
        import json
        decision = json.loads(response_text.strip())
        
        # Ensure fields
        decision.setdefault('proceed', True)
        decision.setdefault('confidence', 50)
        decision.setdefault('reason', 'No pattern')
        decision.setdefault('size_multiplier', 1.0)
        
        # Log to DB
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO ai_trade_decisions (trade_id, setup_json, decision_json, latency_ms, model_used)
            VALUES (%s, %s, %s, %s, 'claude-3.5-sonnet')
        """, (
            setup.trade_id,
            json.dumps(setup.dict()),
            json.dumps(decision),
            int((time.time() - start_time) * 1000)
        ))
        conn.commit()
        cur.close()
        conn.close()
        
        latency = int((time.time() - start_time) * 1000)
        print(f"✅ {setup.trade_id}: {'APPROVED' if decision['proceed'] else 'BLOCKED'} ({decision['confidence']}%) in {latency}ms")
        
        return TradeDecision(**decision)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return TradeDecision(proceed=True, confidence=0, reason=f"Error: {str(e)[:50]}", size_multiplier=1.0)

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

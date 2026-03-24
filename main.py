from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import openai
import os
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Kimi Trade Guardian", version="1.0")

# Initialize Kimi client
client = openai.OpenAI(
    api_key=os.getenv("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

# Database pool for speed
db_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=2, maxconn=10,
    dsn=os.getenv("DATABASE_URL")
)

# Thread pool for async DB queries
executor = ThreadPoolExecutor(max_workers=4)

class TradeSetup(BaseModel):
    trade_id: str
    symbol: str
    direction: str  # Long or Short
    entry_price: float
    stop_loss: float
    take_profit: float
    setup_type: str  # MTF_Fib_618, EMA_Cross, etc.
    timestamp: str  # ISO format
    account_balance: Optional[float] = 25000.0
    daily_pnl: Optional[float] = 0.0
    vix: Optional[float] = 15.0
    consecutive_losses: Optional[int] = 0

class TradeDecision(BaseModel):
    proceed: bool
    confidence: int  # 0-100
    reason: str
    size_multiplier: float  # 0.0 to 2.0
    suggested_stop: Optional[float] = None
    risk_warning: Optional[str] = None

def get_recent_performance(conn, symbol: str, setup_type: str):
    """Get last 20 trades for context - optimized query"""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT pnl, win, timestamp, session_time
        FROM trade_outcomes 
        WHERE symbol = %s 
        AND setup_type = %s
        AND timestamp > NOW() - INTERVAL '30 days'
        ORDER BY timestamp DESC 
        LIMIT 20
    """, (symbol, setup_type))
    
    results = cur.fetchall()
    cur.close()
    
    if not results:
        return {"avg_pnl": 0, "win_rate": 50, "recent_trades": []}
    
    wins = sum(1 for r in results if r['win'])
    return {
        "avg_pnl": sum(r['pnl'] for r in results) / len(results),
        "win_rate": (wins / len(results)) * 100,
        "recent_trades": results[:5]  # Last 5 only
    }

def check_todays_stats(conn):
    """Check if hitting daily limits"""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT 
            SUM(pnl) as daily_pnl,
            COUNT(*) as trade_count,
            SUM(CASE WHEN win THEN 1 ELSE 0 END) as wins
        FROM trade_outcomes 
        WHERE DATE(timestamp) = CURRENT_DATE
    """)
    result = cur.fetchone()
    cur.close()
    return result

@app.post("/analyze-trade", response_model=TradeDecision)
async def analyze_trade(setup: TradeSetup):
    """
    Endpoint called by NinjaTrader before every entry.
    Target response time: <300ms
    """
    start_time = time.time()
    
    # Get DB connection from pool
    conn = db_pool.getconn()
    
    try:
        # Fetch context data (async in thread pool to not block)
        loop = asyncio.get_event_loop()
        perf_data = await loop.run_in_executor(
            executor, get_recent_performance, conn, setup.symbol, setup.setup_type
        )
        today_stats = await loop.run_in_executor(executor, check_todays_stats, conn)
        
        # Build the prompt for Kimi
        prompt = f"""You are an elite futures risk manager analyzing a trade setup in real-time.

TRADE SETUP:
- Symbol: {setup.symbol}
- Direction: {setup.direction}
- Entry: {setup.entry_price}, Stop: {setup.stop_loss}, Target: {setup.take_profit}
- Risk/Reward: 1:{abs((setup.take_profit - setup.entry_price) / (setup.entry_price - setup.stop_loss)):.1f}
- Setup Type: {setup.setup_type}
- Time (ET): {datetime.fromisoformat(setup.timestamp.replace('Z', '+00:00')).strftime('%H:%M') if 'T' in setup.timestamp else 'Unknown'}
- VIX: {setup.vix}
- Consecutive Losses Today: {setup.consecutive_losses}

PERFORMANCE CONTEXT (Last 30 Days):
- This setup avg PnL: ${perf_data['avg_pnl']:.2f}
- This setup win rate: {perf_data['win_rate']:.1f}%
- Today's PnL so far: ${today_stats['daily_pnl'] or 0:.2f}
- Today's trades: {today_stats['trade_count'] or 0}

RECENT TRADES (Last 5):
{json.dumps([{'pnl': r['pnl'], 'win': r['win'], 'time': str(r['timestamp'])} for r in perf_data['recent_trades']], indent=2)}

DECISION RULES:
1. If consecutive_losses >= 2, RECOMMEND HALF SIZE or REJECT
2. If daily_pnl < -1000, REJECT (protect capital)
3. If win_rate < 35% for this setup, REJECT or require 1:3 R/R
4. If time is after 14:00 ET and not trend-following, REJECT
5. If VIX > 20 and stop < 10 ticks, REJECT (volatility too high)

Respond in JSON:
{{
    "proceed": true/false,
    "confidence": 0-100,
    "reason": "Brief explanation",
    "size_multiplier": 0.5/1.0/1.5/2.0,
    "suggested_stop": null or price,
    "risk_warning": "Optional warning"
}}"""

        # Call Kimi with low latency settings
        response = client.chat.completions.create(
            model="kimi-k2",  # Fast model for quick decisions
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Very deterministic
            max_tokens=200,   # Short response for speed
            response_format={"type": "json_object"}
        )
        
        decision = json.loads(response.choices[0].message.content)
        
        # Ensure all fields exist
        decision.setdefault('proceed', True)
        decision.setdefault('confidence', 50)
        decision.setdefault('reason', 'No specific pattern detected')
        decision.setdefault('size_multiplier', 1.0)
        
        # Log the decision for learning
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO ai_trade_decisions 
            (trade_id, timestamp, setup_json, decision_json, latency_ms, model_used)
            VALUES (%s, NOW(), %s, %s, %s, 'kimi-k2')
        """, (
            setup.trade_id,
            json.dumps(setup.dict()),
            json.dumps(decision),
            int((time.time() - start_time) * 1000)
        ))
        conn.commit()
        cur.close()
        
        print(f"✅ Trade {setup.trade_id}: {'APPROVED' if decision['proceed'] else 'BLOCKED'} ({decision['confidence']}%) in {(time.time()-start_time)*1000:.0f}ms")
        
        return TradeDecision(**decision)
        
    except Exception as e:
        print(f"❌ Error analyzing trade: {e}")
        # Fail open (allow trade) but warn
        return TradeDecision(
            proceed=True,
            confidence=0,
            reason=f"AI Error: {str(e)}. Trading without validation.",
            size_multiplier=1.0,
            risk_warning="System error - use manual discretion"
        )
    finally:
        db_pool.putconn(conn)

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow()}

# Database initialization endpoint (run once)
@app.post("/init-db")
def init_database():
    conn = db_pool.getconn()
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
        );
        
        CREATE INDEX IF NOT EXISTS idx_trade_id ON ai_trade_decisions(trade_id);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON ai_trade_decisions(timestamp);
    """)
    
    conn.commit()
    cur.close()
    db_pool.putconn(conn)
    return {"message": "Database initialized"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

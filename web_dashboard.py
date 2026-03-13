"""V4: FastAPI web dashboard — portfolio history, trade log, signal analysis, health check."""

import logging
import time as _time
from datetime import datetime

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

import config
import database

logger = logging.getLogger(__name__)

app = FastAPI(title="Trading Bot V4 Dashboard", docs_url=None, redoc_url=None)

_start_time = _time.time()


@app.get("/health")
async def health():
    """Health check for Docker / monitoring."""
    uptime_sec = _time.time() - _start_time
    try:
        positions = database.load_open_positions()
        open_count = len(positions)
    except Exception:
        open_count = -1

    vix_level = None
    if config.VIX_RISK_SCALING_ENABLED:
        try:
            from risk import get_vix_level
            vix_level = get_vix_level()
        except Exception:
            pass

    return {
        "status": "ok",
        "uptime_seconds": round(uptime_sec),
        "open_positions": open_count,
        "vix": vix_level,
        "paper_mode": config.PAPER_MODE,
    }


# --- API Endpoints ---

@app.get("/api/portfolio_history")
async def portfolio_history(days: int = Query(30, ge=1, le=365)):
    """Return daily portfolio snapshots."""
    return database.get_daily_snapshots(days=days)


@app.get("/api/trades")
async def trades(limit: int = Query(100, ge=1, le=1000),
                 offset: int = Query(0, ge=0),
                 strategy: str = Query(None)):
    """Return recent trades, optionally filtered by strategy."""
    return database.get_trades_paginated(limit=limit, offset=offset, strategy=strategy)


@app.get("/api/stats")
async def stats():
    """Return current performance statistics."""
    try:
        from analytics import compute_analytics
        analytics = compute_analytics()
    except Exception:
        analytics = {}
    return analytics


@app.get("/api/signals")
async def signals(date: str = Query(None)):
    """Return signals for a specific date, or today."""
    if date is None:
        date = datetime.now(config.ET).strftime("%Y-%m-%d")
    return database.get_signals_by_date(date)


@app.get("/api/positions")
async def positions():
    """Return current open positions."""
    return database.load_open_positions()


@app.get("/api/signal_stats")
async def signal_stats(days: int = Query(7, ge=1, le=90)):
    """Return signal skip reason breakdown."""
    return database.get_signal_skip_reasons(days=days)


@app.get("/api/shadow_trades")
async def shadow_trades(days: int = Query(14, ge=1, le=90)):
    """Return shadow trade data and performance."""
    try:
        open_shadows = database.get_open_shadow_trades()
        performance = database.get_shadow_performance(days=days)
        return {"open": open_shadows, "performance": performance}
    except Exception as e:
        return {"open": [], "performance": [], "error": str(e)}


@app.get("/api/trade_analysis")
async def trade_analysis(days: int = Query(7, ge=1, le=90)):
    """Return exit reason breakdown and filter block summary."""
    try:
        exit_breakdown = database.get_exit_reason_breakdown(days=days)
        filter_blocks = database.get_filter_block_summary()
        return {"exit_breakdown": exit_breakdown, "filter_blocks": filter_blocks}
    except Exception as e:
        return {"exit_breakdown": [], "filter_blocks": {}, "error": str(e)}


# --- HTML Dashboard ---

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Single-page dashboard with Chart.js equity curve and trade table."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Bot V3</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Courier New',monospace;background:#0d1117;color:#c9d1d9;padding:20px}
h1{color:#58a6ff;margin-bottom:20px;font-size:1.4em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}
.card .label{color:#8b949e;font-size:0.75em;text-transform:uppercase}
.card .value{font-size:1.4em;font-weight:bold;margin-top:4px}
.green{color:#3fb950}.red{color:#f85149}.blue{color:#58a6ff}
.chart-container{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:24px}
table{width:100%;border-collapse:collapse;font-size:0.85em}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:normal;text-transform:uppercase;font-size:0.7em}
tr:hover{background:#161b22}
.filters{margin-bottom:12px}
.filters select{background:#161b22;color:#c9d1d9;border:1px solid #30363d;padding:6px 12px;border-radius:4px}
.section{margin-bottom:24px}
.section h2{color:#8b949e;font-size:0.9em;margin-bottom:12px;text-transform:uppercase}
.footer{color:#484f58;font-size:0.75em;margin-top:32px;text-align:center}
</style>
</head>
<body>
<h1>ALGO TRADING BOT V3</h1>

<div class="grid" id="stats-grid"></div>

<div class="chart-container">
<canvas id="equityChart" height="80"></canvas>
</div>

<div class="section">
<h2>Recent Trades</h2>
<div class="filters">
<select id="stratFilter" onchange="loadTrades()">
<option value="">All Strategies</option>
<option value="ORB">ORB</option>
<option value="VWAP">VWAP</option>
<option value="MOMENTUM">Momentum</option>
<option value="GAP_GO">Gap & Go</option>
<option value="EMA_SCALP">EMA Scalp</option>
</select>
</div>
<table>
<thead><tr><th>Time</th><th>Symbol</th><th>Strategy</th><th>Side</th><th>Entry</th><th>Exit</th><th>Qty</th><th>P&L</th><th>%</th><th>Reason</th></tr></thead>
<tbody id="trades-body"></tbody>
</table>
</div>

<div class="section">
<h2>Trade Analysis (7 days)</h2>
<div id="trade-analysis"></div>
</div>

<div class="section">
<h2>Shadow Trades</h2>
<div id="shadow-trades"></div>
</div>

<div class="section">
<h2>Signal Filter Reasons (7 days)</h2>
<div id="signal-stats"></div>
</div>

<div class="footer">Auto-refreshes every 30s | Trading Bot V5</div>

<script>
let chart=null;

async function loadStats(){
 try{
  const r=await fetch('/api/stats');
  const d=await r.json();
  const g=document.getElementById('stats-grid');
  g.innerHTML=`
   <div class="card"><div class="label">Sharpe (7d)</div><div class="value blue">${(d.sharpe_7d||0).toFixed(2)}</div></div>
   <div class="card"><div class="label">Win Rate</div><div class="value">${((d.win_rate||0)*100).toFixed(0)}%</div></div>
   <div class="card"><div class="label">Profit Factor</div><div class="value">${(d.profit_factor||0).toFixed(2)}</div></div>
   <div class="card"><div class="label">Max Drawdown</div><div class="value red">${((d.max_drawdown||0)*100).toFixed(1)}%</div></div>
   <div class="card"><div class="label">Week P&L</div><div class="value ${(d.week_pnl||0)>=0?'green':'red'}">$${(d.week_pnl||0).toFixed(0)}</div></div>
   <div class="card"><div class="label">Trades (7d)</div><div class="value">${d.total_trades_7d||0}</div></div>
  `;
 }catch(e){console.error(e)}
}

async function loadChart(){
 try{
  const r=await fetch('/api/portfolio_history?days=60');
  const d=await r.json();
  if(!d.length)return;
  d.reverse();
  const labels=d.map(x=>x.date);
  const values=d.map(x=>x.portfolio_value);
  const ctx=document.getElementById('equityChart').getContext('2d');
  if(chart)chart.destroy();
  chart=new Chart(ctx,{
   type:'line',
   data:{labels,datasets:[{label:'Portfolio Value',data:values,
    borderColor:'#58a6ff',backgroundColor:'rgba(88,166,255,0.1)',fill:true,tension:0.3,pointRadius:2}]},
   options:{responsive:true,plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:'#484f58',maxTicksLimit:10},grid:{color:'#21262d'}},
     y:{ticks:{color:'#484f58',callback:v=>'$'+v.toLocaleString()},grid:{color:'#21262d'}}}}
  });
 }catch(e){console.error(e)}
}

async function loadTrades(){
 try{
  const strat=document.getElementById('stratFilter').value;
  const url='/api/trades?limit=50'+(strat?'&strategy='+strat:'');
  const r=await fetch(url);
  const d=await r.json();
  const tb=document.getElementById('trades-body');
  tb.innerHTML=d.map(t=>{
   const pnlClass=(t.pnl||0)>=0?'green':'red';
   const time=(t.exit_time||'').substring(5,16);
   return `<tr><td>${time}</td><td>${t.symbol}</td><td>${t.strategy}</td><td>${t.side}</td>
    <td>$${(t.entry_price||0).toFixed(2)}</td><td>$${(t.exit_price||0).toFixed(2)}</td>
    <td>${t.qty}</td><td class="${pnlClass}">$${(t.pnl||0).toFixed(2)}</td>
    <td class="${pnlClass}">${((t.pnl_pct||0)*100).toFixed(1)}%</td>
    <td>${t.exit_reason||''}</td></tr>`;
  }).join('');
 }catch(e){console.error(e)}
}

async function loadSignalStats(){
 try{
  const r=await fetch('/api/signal_stats?days=7');
  const d=await r.json();
  const el=document.getElementById('signal-stats');
  const items=Object.entries(d).map(([k,v])=>`<span style="margin-right:16px">${k}: <b>${v}</b></span>`);
  el.innerHTML=items.join('')||'<span>No filtered signals</span>';
 }catch(e){console.error(e)}
}

async function loadTradeAnalysis(){
 try{
  const r=await fetch('/api/trade_analysis?days=7');
  const d=await r.json();
  const el=document.getElementById('trade-analysis');
  let html='';
  if(d.exit_breakdown&&d.exit_breakdown.length){
   html+='<div style="margin-bottom:12px"><b>Exit Reasons:</b></div>';
   html+='<table><thead><tr><th>Reason</th><th>Count</th><th>Avg P&L</th><th>Avg %</th></tr></thead><tbody>';
   d.exit_breakdown.forEach(r=>{
    const pnlClass=(r.avg_pnl||0)>=0?'green':'red';
    html+=`<tr><td>${r.exit_reason||'unknown'}</td><td>${r.count}</td><td class="${pnlClass}">$${(r.avg_pnl||0).toFixed(2)}</td><td class="${pnlClass}">${((r.avg_pnl_pct||0)*100).toFixed(1)}%</td></tr>`;
   });
   html+='</tbody></table>';
  }
  if(d.filter_blocks&&Object.keys(d.filter_blocks).length){
   html+='<div style="margin-top:12px;margin-bottom:8px"><b>Filter Blocks (today):</b></div>';
   Object.entries(d.filter_blocks).forEach(([k,v])=>{html+=`<span style="margin-right:16px">${k}: <b>${v}</b></span>`;});
  }
  el.innerHTML=html||'<span>No trade analysis data</span>';
 }catch(e){console.error(e)}
}

async function loadShadowTrades(){
 try{
  const r=await fetch('/api/shadow_trades?days=14');
  const d=await r.json();
  const el=document.getElementById('shadow-trades');
  let html='';
  if(d.performance&&d.performance.length){
   html+='<div style="margin-bottom:8px"><b>Shadow Performance (14d):</b></div>';
   html+='<table><thead><tr><th>Strategy</th><th>Trades</th><th>Wins</th><th>Total P&L</th><th>Avg %</th></tr></thead><tbody>';
   d.performance.forEach(p=>{
    const wr=p.trades>0?(p.wins/p.trades*100).toFixed(0)+'%':'0%';
    const pnlClass=(p.total_pnl||0)>=0?'green':'red';
    html+=`<tr><td>${p.strategy}</td><td>${p.trades} (${wr} win)</td><td>${p.wins}</td><td class="${pnlClass}">$${(p.total_pnl||0).toFixed(2)}</td><td>${((p.avg_pnl_pct||0)*100).toFixed(2)}%</td></tr>`;
   });
   html+='</tbody></table>';
  }
  if(d.open&&d.open.length){
   html+='<div style="margin-top:12px;margin-bottom:8px"><b>Open Shadow Trades:</b></div>';
   html+='<table><thead><tr><th>Symbol</th><th>Strategy</th><th>Side</th><th>Entry</th><th>TP</th><th>SL</th></tr></thead><tbody>';
   d.open.forEach(t=>{
    html+=`<tr><td>${t.symbol}</td><td>${t.strategy}</td><td>${t.side}</td><td>$${(t.entry_price||0).toFixed(2)}</td><td>$${(t.take_profit||0).toFixed(2)}</td><td>$${(t.stop_loss||0).toFixed(2)}</td></tr>`;
   });
   html+='</tbody></table>';
  }
  el.innerHTML=html||'<span>No shadow trades</span>';
 }catch(e){console.error(e)}
}

function refresh(){loadStats();loadChart();loadTrades();loadSignalStats();loadTradeAnalysis();loadShadowTrades()}
refresh();
setInterval(refresh,30000);
</script>
</body>
</html>"""


def start_web_dashboard():
    """Start the web dashboard in a background thread."""
    import threading
    import uvicorn

    def _run():
        try:
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=config.WEB_DASHBOARD_PORT,
                log_level="error",
            )
        except Exception as e:
            logger.error(f"Web dashboard crashed: {e}")

    thread = threading.Thread(target=_run, daemon=True, name="web-dashboard")
    thread.start()
    logger.info(f"Web dashboard started at http://localhost:{config.WEB_DASHBOARD_PORT}")

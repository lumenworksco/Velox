"""MON-005: Position Reconciliation — compare bot state to broker, detect and auto-heal discrepancies.

Runs every 30 minutes and at market open/close.

Detects:
  - Phantom positions: in DB but not at broker
  - Ghost positions: at broker but not tracked in DB
  - Quantity mismatches: tracked qty differs from broker qty

Auto-heals:
  - Creates tracking records for untracked broker positions
  - Updates DB qty for mismatches
  - Removes phantom positions from DB

Extends the existing watchdog.PositionReconciler with enhanced reporting
and auto-heal capabilities.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PositionDiscrepancy:
    """A single reconciliation discrepancy."""

    symbol: str
    discrepancy_type: str  # "phantom", "ghost", "qty_mismatch", "price_discrepancy"
    db_qty: float = 0.0
    broker_qty: float = 0.0
    db_strategy: str = ""
    broker_side: str = ""
    auto_healed: bool = False
    heal_action: str = ""
    detail: str = ""


@dataclass
class ReconciliationReport:
    """Full reconciliation report."""

    timestamp: datetime
    positions_checked: int = 0
    discrepancies: list[PositionDiscrepancy] = field(default_factory=list)
    all_reconciled: bool = True
    phantom_count: int = 0
    ghost_count: int = 0
    mismatch_count: int = 0
    auto_healed_count: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return len(self.discrepancies) > 0


@dataclass
class SyncResult:
    """Result of a force-sync operation."""

    timestamp: datetime
    actions_taken: list[str] = field(default_factory=list)
    success: bool = True
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PositionReconciler
# ---------------------------------------------------------------------------

class PositionReconciler:
    """Enhanced position reconciler with auto-heal and detailed reporting.

    Usage:
        reconciler = PositionReconciler(trading_client_fn=get_client)
        report = reconciler.reconcile()

        if report.has_issues:
            print(f"Found {len(report.discrepancies)} discrepancies")

        # Force full sync
        result = reconciler.force_sync()
    """

    def __init__(self, trading_client_fn=None, alert_callback=None,
                 report_log_path: str | None = None):
        """
        Args:
            trading_client_fn: Callable returning the Alpaca TradingClient.
            alert_callback: Optional callable(level, message, source) for alerts.
            report_log_path: Path for reconciliation report log.
        """
        self._trading_client_fn = trading_client_fn
        self._alert_callback = alert_callback
        self._report_log = Path(report_log_path or "reconciliation_log.jsonl")
        self._history: list[ReconciliationReport] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconcile(self, bot_positions: list[dict] | None = None,
                  broker_positions: list | None = None) -> ReconciliationReport:
        """Run a full reconciliation check.

        Compares bot positions against broker positions.  When called with
        explicit arguments, uses those directly.  When called without
        arguments, loads positions from the database and broker API.

        Auto-heals discrepancies when safe to do so.

        Args:
            bot_positions: List of dicts with at least ``symbol`` and ``qty`` keys.
                           If *None*, loads from the ``open_positions`` DB table.
            broker_positions: List of broker position objects (each must expose
                              ``symbol``, ``qty``, ``side``, ``avg_entry_price``
                              attributes).  If *None*, queries the broker API.

        Never raises — returns a report with errors on failure.
        """
        try:
            report = self._reconcile_inner(bot_positions, broker_positions)
        except Exception as e:
            logger.error(f"PositionReconciler.reconcile failed: {e}")
            report = ReconciliationReport(
                timestamp=datetime.now(config.ET),
                errors=[f"reconciliation_failed: {e}"],
                all_reconciled=False,
            )

        # Store history
        with self._lock:
            self._history.append(report)
            if len(self._history) > 200:
                self._history = self._history[-200:]

        # Log report
        self._log_report(report)

        # Alert on issues
        if report.has_issues:
            self._send_alerts(report)

        return report

    def force_sync(self, broker_client=None, risk_manager=None) -> SyncResult:
        """Force-sync DB state to match broker positions exactly.

        More aggressive than reconcile() — overwrites DB with broker truth.

        Args:
            broker_client: Optional broker client override.  If provided, uses
                           this instead of ``self._trading_client_fn()``.
            risk_manager: Optional risk manager.  If provided, its position
                          tracking is updated after the sync.

        Never raises — returns SyncResult with errors on failure.
        """
        try:
            return self._force_sync_inner(broker_client, risk_manager)
        except Exception as e:
            logger.error(f"PositionReconciler.force_sync failed: {e}")
            return SyncResult(
                timestamp=datetime.now(config.ET),
                success=False,
                errors=[str(e)],
            )

    @property
    def last_report(self) -> Optional[ReconciliationReport]:
        with self._lock:
            return self._history[-1] if self._history else None

    @property
    def report_history(self) -> list[ReconciliationReport]:
        with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------
    # Internal — reconciliation
    # ------------------------------------------------------------------

    def _reconcile_inner(self, bot_positions=None, broker_positions_arg=None) -> ReconciliationReport:
        """Core reconciliation logic."""
        import database

        now = datetime.now(config.ET)
        report = ReconciliationReport(timestamp=now)

        # Load DB positions (use argument if provided)
        if bot_positions is not None:
            db_positions = bot_positions
        else:
            try:
                db_positions = database.load_open_positions()
            except Exception as e:
                report.errors.append(f"Failed to load DB positions: {e}")
                report.all_reconciled = False
                return report

        db_by_symbol: dict[str, dict] = {p["symbol"]: p for p in db_positions}

        # Load broker positions (use argument if provided)
        if broker_positions_arg is not None:
            broker_positions = broker_positions_arg
        elif self._trading_client_fn is not None:
            try:
                client = self._trading_client_fn()
                broker_positions = client.get_all_positions() or []
            except Exception as e:
                report.errors.append(f"Failed to load broker positions: {e}")
                report.all_reconciled = False
                return report
        else:
            logger.debug("PositionReconciler: No trading client configured, skipping broker check")
            report.positions_checked = len(db_positions)
            return report

        broker_by_symbol: dict = {
            pos.symbol: pos for pos in broker_positions
        }

        all_symbols = set(db_by_symbol.keys()) | set(broker_by_symbol.keys())
        report.positions_checked = len(all_symbols)

        for symbol in sorted(all_symbols):
            in_db = symbol in db_by_symbol
            at_broker = symbol in broker_by_symbol

            if in_db and not at_broker:
                # Phantom: in DB but not at broker
                disc = PositionDiscrepancy(
                    symbol=symbol,
                    discrepancy_type="phantom",
                    db_qty=float(db_by_symbol[symbol].get("qty", 0)),
                    db_strategy=db_by_symbol[symbol].get("strategy", ""),
                    detail="Position in DB but not at broker — likely filled externally",
                )

                # Auto-heal: remove phantom from DB
                try:
                    self._remove_phantom(symbol)
                    disc.auto_healed = True
                    disc.heal_action = "removed_from_db"
                    logger.warning(f"RECONCILER: Removed phantom position {symbol} from DB")
                except Exception as e:
                    disc.detail += f" | heal_failed: {e}"
                    report.errors.append(f"Failed to remove phantom {symbol}: {e}")

                report.discrepancies.append(disc)
                report.phantom_count += 1

            elif at_broker and not in_db:
                # Ghost: at broker but not in DB
                broker_pos = broker_by_symbol[symbol]
                broker_qty = float(getattr(broker_pos, "qty", 0))
                broker_side = getattr(broker_pos, "side", "unknown")
                avg_price = float(getattr(broker_pos, "avg_entry_price", 0))

                disc = PositionDiscrepancy(
                    symbol=symbol,
                    discrepancy_type="ghost",
                    broker_qty=broker_qty,
                    broker_side=str(broker_side),
                    detail=f"Position at broker (qty={broker_qty}, side={broker_side}, "
                           f"avg_price={avg_price}) not tracked in DB",
                )

                # Auto-heal: create tracking record
                try:
                    self._create_tracking_record(symbol, broker_pos)
                    disc.auto_healed = True
                    disc.heal_action = "created_tracking_record"
                    logger.warning(f"RECONCILER: Created tracking record for ghost position {symbol}")
                except Exception as e:
                    disc.detail += f" | heal_failed: {e}"
                    report.errors.append(f"Failed to create tracking for {symbol}: {e}")

                report.discrepancies.append(disc)
                report.ghost_count += 1

            else:
                # Both exist — check quantity
                db_qty = float(db_by_symbol[symbol].get("qty", 0))
                broker_qty = float(getattr(broker_by_symbol[symbol], "qty", 0))

                if abs(db_qty - broker_qty) > 0.001:
                    disc = PositionDiscrepancy(
                        symbol=symbol,
                        discrepancy_type="qty_mismatch",
                        db_qty=db_qty,
                        broker_qty=broker_qty,
                        db_strategy=db_by_symbol[symbol].get("strategy", ""),
                        detail=f"DB qty={db_qty} vs broker qty={broker_qty}",
                    )

                    # Auto-heal: update DB to match broker
                    try:
                        self._update_qty(symbol, broker_qty)
                        disc.auto_healed = True
                        disc.heal_action = f"updated_db_qty_to_{broker_qty}"
                        logger.warning(
                            f"RECONCILER: Updated {symbol} qty from {db_qty} to {broker_qty}"
                        )
                    except Exception as e:
                        disc.detail += f" | heal_failed: {e}"
                        report.errors.append(f"Failed to update qty for {symbol}: {e}")

                    report.discrepancies.append(disc)
                    report.mismatch_count += 1

                # Check for price discrepancy (> 1% difference)
                db_price = float(db_by_symbol[symbol].get("entry_price", 0))
                broker_price = float(getattr(broker_by_symbol[symbol], "avg_entry_price", 0))
                if db_price > 0 and broker_price > 0:
                    price_diff_pct = abs(db_price - broker_price) / db_price
                    if price_diff_pct > 0.01:
                        disc = PositionDiscrepancy(
                            symbol=symbol,
                            discrepancy_type="price_discrepancy",
                            db_qty=float(db_by_symbol[symbol].get("qty", 0)),
                            broker_qty=float(getattr(broker_by_symbol[symbol], "qty", 0)),
                            db_strategy=db_by_symbol[symbol].get("strategy", ""),
                            detail=(
                                f"Entry price mismatch: DB=${db_price:.2f} vs "
                                f"broker=${broker_price:.2f} ({price_diff_pct:.1%} diff)"
                            ),
                        )
                        report.discrepancies.append(disc)
                        report.mismatch_count += 1

        report.auto_healed_count = sum(1 for d in report.discrepancies if d.auto_healed)
        report.all_reconciled = not report.has_issues or (
            report.auto_healed_count == len(report.discrepancies) and not report.errors
        )

        return report

    # ------------------------------------------------------------------
    # Internal — force sync
    # ------------------------------------------------------------------

    def _force_sync_inner(self, broker_client=None, risk_manager=None) -> SyncResult:
        """Overwrite DB positions with broker positions."""
        import database

        now = datetime.now(config.ET)
        result = SyncResult(timestamp=now)

        # Resolve client
        client = broker_client
        if client is None and self._trading_client_fn is not None:
            client = self._trading_client_fn()
        if client is None:
            result.errors.append("No trading client configured")
            result.success = False
            return result

        try:
            broker_positions = client.get_all_positions() or []
        except Exception as e:
            result.errors.append(f"Failed to load broker positions: {e}")
            result.success = False
            return result

        # Clear all DB positions
        try:
            conn = database._get_conn()
            conn.execute("DELETE FROM open_positions")
            conn.commit()
            result.actions_taken.append("cleared_all_db_positions")
        except Exception as e:
            result.errors.append(f"Failed to clear DB: {e}")
            result.success = False
            return result

        # Re-insert from broker
        for pos in broker_positions:
            try:
                self._create_tracking_record(pos.symbol, pos)
                result.actions_taken.append(f"synced_{pos.symbol}")
            except Exception as e:
                result.errors.append(f"Failed to sync {pos.symbol}: {e}")

        result.success = len(result.errors) == 0

        # Notify risk manager if provided
        if risk_manager is not None:
            try:
                if hasattr(risk_manager, "reload_positions"):
                    risk_manager.reload_positions()
                    result.actions_taken.append("notified_risk_manager")
            except Exception as e:
                result.errors.append(f"Failed to notify risk manager: {e}")

        logger.info(
            f"RECONCILER: Force sync complete — "
            f"{len(result.actions_taken)} actions, {len(result.errors)} errors"
        )
        return result

    # ------------------------------------------------------------------
    # Internal — DB operations
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_phantom(symbol: str):
        """Remove a phantom position from the database."""
        import database
        conn = database._get_conn()
        conn.execute("DELETE FROM open_positions WHERE symbol = ?", (symbol,))
        conn.commit()

    @staticmethod
    def _update_qty(symbol: str, new_qty: float):
        """Update position qty in DB to match broker."""
        import database
        conn = database._get_conn()
        conn.execute(
            "UPDATE open_positions SET qty = ? WHERE symbol = ?",
            (new_qty, symbol),
        )
        conn.commit()

    @staticmethod
    def _create_tracking_record(symbol: str, broker_pos):
        """Create an open_positions record for a ghost broker position."""
        import database

        qty = float(getattr(broker_pos, "qty", 0))
        side = str(getattr(broker_pos, "side", "long"))
        avg_price = float(getattr(broker_pos, "avg_entry_price", 0))

        # Normalize side to match our DB convention
        if hasattr(side, "value"):
            side = side.value
        side_str = "buy" if "long" in side.lower() else "sell"

        now = datetime.now(config.ET)
        conn = database._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO open_positions "
            "(symbol, strategy, side, entry_price, qty, entry_time, take_profit, stop_loss) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                symbol,
                "RECONCILED",  # Mark as auto-reconciled
                side_str,
                avg_price,
                qty,
                now.isoformat(),
                avg_price * (1.05 if side_str == "buy" else 0.95),  # 5% default TP
                avg_price * (0.97 if side_str == "buy" else 1.03),  # 3% default SL
            ),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Internal — reporting and alerting
    # ------------------------------------------------------------------

    def _log_report(self, report: ReconciliationReport):
        """Append reconciliation report to log file."""
        try:
            entry = {
                "timestamp": report.timestamp.isoformat(),
                "positions_checked": report.positions_checked,
                "phantoms": report.phantom_count,
                "ghosts": report.ghost_count,
                "mismatches": report.mismatch_count,
                "auto_healed": report.auto_healed_count,
                "all_reconciled": report.all_reconciled,
                "errors": report.errors,
                "discrepancies": [
                    {
                        "symbol": d.symbol,
                        "type": d.discrepancy_type,
                        "db_qty": d.db_qty,
                        "broker_qty": d.broker_qty,
                        "auto_healed": d.auto_healed,
                        "heal_action": d.heal_action,
                    }
                    for d in report.discrepancies
                ],
            }
            with open(self._report_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"RECONCILER: Failed to write report log: {e}")

    def _send_alerts(self, report: ReconciliationReport):
        """Send alerts for reconciliation discrepancies."""
        if not self._alert_callback:
            return

        for disc in report.discrepancies:
            if disc.auto_healed:
                level = "WARNING"
            else:
                level = "CRITICAL"

            msg = (
                f"Position reconciliation: {disc.discrepancy_type} "
                f"for {disc.symbol} — {disc.detail}"
                + (f" [auto-healed: {disc.heal_action}]" if disc.auto_healed else "")
            )

            try:
                self._alert_callback(level, msg, "reconciler")
            except Exception as e:
                logger.error(f"RECONCILER: Alert callback failed: {e}")

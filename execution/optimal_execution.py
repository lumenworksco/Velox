"""EXEC-002: Almgren-Chriss Optimal Execution Model.

Computes execution schedules that minimize the combination of:
- Temporary market impact (instantaneous price pressure)
- Permanent market impact (information leakage)
- Timing risk (variance from price drift while executing)

Front-loads for momentum (decaying signals), back-loads for mean reversion
(persistent signals).

Reference: Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions"
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import config

logger = logging.getLogger(__name__)


@dataclass
class ExecutionSlice:
    """A single slice in an execution schedule."""
    slice_index: int
    qty: int
    target_time_offset_sec: float   # Seconds from schedule start
    expected_price_impact_bps: float
    cumulative_pct: float           # Cumulative % of order filled after this slice
    is_final: bool = False


@dataclass
class ExecutionSchedule:
    """Complete execution schedule for an order."""
    total_qty: int
    slices: list[ExecutionSlice]
    total_duration_sec: float
    expected_total_impact_bps: float
    expected_timing_risk_bps: float
    strategy_type: str               # "front_load", "back_load", "uniform"
    lambda_urgency: float            # Urgency parameter used


@dataclass
class ImpactParams:
    """Calibrated market impact parameters."""
    # Temporary impact: eta * (v_k / V)^alpha
    eta: float = 0.05         # Temporary impact coefficient
    alpha: float = 0.6        # Temporary impact exponent (sqrt-like)

    # Permanent impact: gamma * (v_k / V)
    gamma: float = 0.10       # Permanent impact coefficient

    # Volatility of the asset (daily)
    sigma: float = 0.02       # Daily volatility (2%)

    # Last calibration timestamp
    calibrated_at: datetime | None = None


# Large order threshold: use Almgren-Chriss closed-form when order notional
# exceeds this value (in dollars). Fall back to TWAP if solver fails.
LARGE_ORDER_THRESHOLD = 50_000  # $50k notional

# Strategy -> execution style mapping
_STRATEGY_STYLE: dict[str, str] = {
    # Decaying signals: front-load to capture alpha before it decays
    "MICRO_MOM": "front_load",
    "ORB": "front_load",

    # Persistent signals: back-load to minimize market impact
    "STAT_MR": "back_load",
    "KALMAN_PAIRS": "back_load",
    "VWAP": "back_load",

    # Multi-day: uniform execution
    "PEAD": "uniform",
}


class AlmgrenChriss:
    """Almgren-Chriss optimal execution model.

    Computes trade schedules that balance urgency (signal decay risk)
    against market impact (cost of trading too fast).

    Usage:
        ac = AlmgrenChriss()
        schedule = ac.compute_execution_schedule(
            order_size=500, urgency=0.5, volatility=0.02, adv=1_000_000
        )
        for s in schedule.slices:
            print(f"  Slice {s.slice_index}: {s.qty} shares at +{s.target_time_offset_sec}s")
    """

    def __init__(self, impact_params: ImpactParams | None = None):
        self._params = impact_params or ImpactParams()
        # Historical fills for calibration
        self._fill_history: list[dict] = []
        self._max_history = 5000

    def compute_execution_schedule(
        self,
        order_size: int,
        urgency: float,
        volatility: float,
        adv: float,
        strategy: str = "",
        total_duration_sec: float | None = None,
    ) -> ExecutionSchedule:
        """Compute the optimal execution schedule.

        Args:
            order_size: Total shares to execute.
            urgency: Lambda parameter [0, 1]. 0 = patient, 1 = immediate.
            volatility: Daily volatility of the asset (e.g. 0.02 = 2%).
            adv: Average daily volume in shares.
            strategy: Strategy name for style selection.
            total_duration_sec: Override execution duration (default: auto-computed).

        Returns:
            ExecutionSchedule with optimal slice breakdown.
        """
        if order_size <= 0:
            logger.warning("compute_execution_schedule called with order_size <= 0")
            return ExecutionSchedule(
                total_qty=0, slices=[], total_duration_sec=0,
                expected_total_impact_bps=0, expected_timing_risk_bps=0,
                strategy_type="uniform", lambda_urgency=urgency,
            )

        # Determine execution style
        style = _STRATEGY_STYLE.get(strategy, "uniform")

        # Auto-compute duration based on size participation rate
        if total_duration_sec is None:
            total_duration_sec = self._auto_duration(order_size, adv, urgency)

        # Number of slices (more for larger orders)
        participation_rate = order_size / adv if adv > 0 else 0.01
        n_slices = self._compute_num_slices(order_size, participation_rate, urgency)

        # Compute the trajectory (how much to trade in each period)
        trajectory = self._compute_trajectory(
            n_slices=n_slices,
            total_qty=order_size,
            urgency=urgency,
            style=style,
        )

        # Build slices with impact estimates
        # CRIT-026: Use actual trajectory length (after zero-qty filtering) to
        # avoid timing gaps when some slices were removed.
        actual_n_slices = len(trajectory)
        interval_sec = total_duration_sec / actual_n_slices if actual_n_slices > 0 else 0
        slices: list[ExecutionSlice] = []
        cumulative = 0

        for i, qty in enumerate(trajectory):
            if qty <= 0:
                continue
            cumulative += qty
            impact_bps = self._estimate_slice_impact(
                qty, order_size, adv, volatility
            )
            slices.append(ExecutionSlice(
                slice_index=i,
                qty=qty,
                target_time_offset_sec=round(i * interval_sec, 1),
                expected_price_impact_bps=round(impact_bps, 2),
                cumulative_pct=round(cumulative / order_size, 4),
                is_final=(i == len(trajectory) - 1),
            ))

        # Total expected impact
        total_impact_bps = self._estimate_total_impact(
            order_size, adv, volatility, n_slices
        )
        timing_risk_bps = self._estimate_timing_risk(
            order_size, volatility, total_duration_sec
        )

        schedule = ExecutionSchedule(
            total_qty=order_size,
            slices=slices,
            total_duration_sec=total_duration_sec,
            expected_total_impact_bps=round(total_impact_bps, 2),
            expected_timing_risk_bps=round(timing_risk_bps, 2),
            strategy_type=style,
            lambda_urgency=urgency,
        )

        logger.info(
            f"AlmgrenChriss: {order_size} shares in {len(slices)} slices "
            f"over {total_duration_sec:.0f}s ({style}). "
            f"Impact={total_impact_bps:.1f}bps, TimingRisk={timing_risk_bps:.1f}bps"
        )
        return schedule

    def calibrate_from_fills(self, fills: list[dict]) -> None:
        """Calibrate impact parameters from historical fill data.

        Each fill dict should contain:
            - qty: int (shares filled)
            - adv: float (average daily volume)
            - expected_price: float (arrival/decision price)
            - filled_price: float (actual fill price)
            - side: str ("buy" or "sell")
            - volatility: float (daily vol at time of fill)

        Updates internal impact parameters (eta, gamma, alpha).
        """
        if not fills:
            logger.warning("calibrate_from_fills called with empty fills list")
            return

        self._fill_history.extend(fills)
        if len(self._fill_history) > self._max_history:
            self._fill_history = self._fill_history[-self._max_history:]

        # Simple OLS-style calibration of temporary impact coefficient
        # Model: impact_bps = eta * (qty/ADV)^alpha
        # We use log-linear regression: log(impact) = log(eta) + alpha * log(qty/ADV)
        valid_fills = []
        for f in self._fill_history:
            if f.get("adv", 0) <= 0 or f.get("qty", 0) <= 0:
                continue
            if f.get("expected_price", 0) <= 0:
                continue

            side_mult = 1.0 if f["side"] == "buy" else -1.0
            impact_pct = side_mult * (f["filled_price"] - f["expected_price"]) / f["expected_price"]
            participation = f["qty"] / f["adv"]

            if impact_pct > 0 and participation > 0:
                valid_fills.append((math.log(participation), math.log(impact_pct * 10_000)))

        if len(valid_fills) < 20:
            logger.debug(
                f"Insufficient fills for calibration ({len(valid_fills)}/20 required)"
            )
            return

        # Log-linear regression: Y = a + b*X
        n = len(valid_fills)
        sum_x = sum(x for x, _ in valid_fills)
        sum_y = sum(y for _, y in valid_fills)
        sum_xy = sum(x * y for x, y in valid_fills)
        sum_xx = sum(x * x for x, _ in valid_fills)

        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-12:
            logger.warning("Calibration failed: degenerate regression")
            return

        b = (n * sum_xy - sum_x * sum_y) / denom
        a = (sum_y - b * sum_x) / n

        new_alpha = max(0.3, min(1.0, b))  # Clamp to reasonable range
        new_eta = max(0.01, min(0.50, math.exp(a) / 10_000))

        logger.info(
            f"AlmgrenChriss calibration: eta={new_eta:.4f} (was {self._params.eta:.4f}), "
            f"alpha={new_alpha:.3f} (was {self._params.alpha:.3f}) from {n} fills"
        )

        self._params.eta = new_eta
        self._params.alpha = new_alpha
        self._params.calibrated_at = datetime.now()

    def auto_calibrate_eta(self, spread: float, adv: float, price: float = 100.0) -> float:
        """Auto-calibrate temporary impact coefficient eta from bid-ask spread and ADV.

        V11.2: eta ~ half-spread / sqrt(ADV_in_dollars) per Almgren (2005).

        Args:
            spread: Bid-ask spread in dollars.
            adv: Average daily volume in shares.
            price: Current price (used to normalize).

        Returns:
            Calibrated eta value (also updates internal params).
        """
        if adv <= 0 or price <= 0:
            return self._params.eta

        half_spread_pct = (spread / 2) / price
        adv_dollars = adv * price
        # eta scales as half-spread normalized by sqrt of dollar volume
        new_eta = half_spread_pct / math.sqrt(adv_dollars / 1e6) if adv_dollars > 0 else 0.05
        new_eta = max(0.005, min(0.50, new_eta))

        logger.info(
            f"Auto-calibrated eta={new_eta:.4f} from spread=${spread:.2f}, "
            f"ADV={adv:,.0f}, price=${price:.2f} (was {self._params.eta:.4f})"
        )
        self._params.eta = new_eta
        self._params.calibrated_at = datetime.now()
        return new_eta

    def compute_ac_closed_form(
        self,
        total_qty: int,
        n_slices: int,
        urgency: float,
        volatility: float,
        adv: float,
    ) -> list[int]:
        """Almgren-Chriss closed-form optimal trajectory.

        x_k = X * sinh(kappa * (T - t_k)) / sinh(kappa * T)

        where kappa^2 = (lambda * sigma^2) / eta, lambda = urgency parameter.

        Args:
            total_qty: Total shares to execute.
            n_slices: Number of time periods T.
            urgency: Risk-aversion parameter lambda in [0, 1].
            volatility: Daily volatility sigma.
            adv: Average daily volume.

        Returns:
            List of per-slice quantities (integers summing to total_qty).
            Falls back to TWAP if solver fails.
        """
        if n_slices <= 0:
            return [total_qty] if total_qty > 0 else []
        if n_slices == 1:
            return [total_qty]

        try:
            eta = self._params.eta
            sigma = volatility
            lam = max(urgency * 5.0, 0.01)  # Scale urgency to meaningful lambda

            # kappa^2 = lambda * sigma^2 / eta
            kappa_sq = (lam * sigma * sigma) / max(eta, 1e-8)
            kappa = math.sqrt(kappa_sq)
            T = float(n_slices)

            sinh_kT = math.sinh(kappa * T)
            if abs(sinh_kT) < 1e-12:
                raise ValueError("sinh(kappa*T) near zero")

            # Remaining inventory at each time step: x_k = X * sinh(kappa*(T-k)) / sinh(kappa*T)
            inventory = []
            for k in range(n_slices + 1):
                x_k = total_qty * math.sinh(kappa * (T - k)) / sinh_kT
                inventory.append(x_k)

            # Per-slice quantity = inventory[k] - inventory[k+1]
            raw_qty = [inventory[k] - inventory[k + 1] for k in range(n_slices)]

            # Convert to integers
            int_qty = [max(0, int(round(q))) for q in raw_qty]
            remainder = total_qty - sum(int_qty)

            # Distribute remainder
            if remainder > 0:
                # Add to slices with largest fractional parts
                fracs = [(raw_qty[i] - int_qty[i], i) for i in range(n_slices)]
                fracs.sort(reverse=True)
                for j in range(min(abs(remainder), n_slices)):
                    int_qty[fracs[j][1]] += 1
            elif remainder < 0:
                # Remove from slices with smallest fractional parts
                fracs = [(raw_qty[i] - int_qty[i], i) for i in range(n_slices)]
                fracs.sort()
                for j in range(min(abs(remainder), n_slices)):
                    if int_qty[fracs[j][1]] > 0:
                        int_qty[fracs[j][1]] -= 1

            # Filter zero-qty slices
            result = [q for q in int_qty if q > 0]
            lost = total_qty - sum(result)
            if lost > 0 and result:
                for j in range(lost):
                    result[j % len(result)] += 1

            logger.debug(
                f"AC closed-form: kappa={kappa:.4f}, {len(result)} slices, "
                f"front/back ratio={result[0]}/{result[-1] if result else 0}"
            )
            return result

        except Exception as e:
            logger.warning(f"AC closed-form failed ({e}), falling back to TWAP")
            # TWAP fallback: uniform distribution
            base = total_qty // n_slices
            remainder = total_qty - base * n_slices
            twap = [base] * n_slices
            for j in range(remainder):
                twap[j] += 1
            return [q for q in twap if q > 0]

    def _compute_trajectory(
        self, n_slices: int, total_qty: int, urgency: float, style: str
    ) -> list[int]:
        """Compute the optimal trading trajectory (shares per slice).

        Args:
            n_slices: Number of execution periods.
            total_qty: Total shares to trade.
            urgency: Urgency parameter [0, 1].
            style: "front_load", "back_load", or "uniform".

        Returns:
            List of per-slice quantities summing to total_qty.
        """
        if n_slices <= 0:
            return [total_qty] if total_qty > 0 else []
        if n_slices == 1:
            return [total_qty]

        # Generate raw weights based on style
        weights: list[float] = []

        if style == "front_load":
            # Exponential decay: more at the start
            decay = 1.0 + urgency * 3.0  # Higher urgency = steeper front-load
            for i in range(n_slices):
                weights.append(math.exp(-decay * i / n_slices))

        elif style == "back_load":
            # Exponential ramp: more at the end
            ramp = 1.0 + (1 - urgency) * 2.0  # Lower urgency = steeper back-load
            for i in range(n_slices):
                weights.append(math.exp(ramp * i / n_slices))

        else:
            # V11.2: Use Almgren-Chriss closed-form for uniform style
            # (delegates to sinh-based optimal trajectory internally)
            for i in range(n_slices):
                t = i / (n_slices - 1) if n_slices > 1 else 0.5
                # Sinh-based optimal trajectory: x_k ~ cosh(kappa*(1-t))
                kappa = urgency * 3.0 + 0.1
                w = math.cosh(kappa * (1 - t)) / math.cosh(kappa)
                weights.append(max(w, 0.1))

        # Normalize to sum to total_qty (integer allocation)
        total_weight = sum(weights)
        if total_weight <= 0:
            return [total_qty // n_slices] * n_slices

        raw_alloc = [total_qty * w / total_weight for w in weights]
        int_alloc = [int(q) for q in raw_alloc]

        # Distribute remainder to slices with largest fractional parts
        remainder = total_qty - sum(int_alloc)
        fractional = [(raw_alloc[i] - int_alloc[i], i) for i in range(n_slices)]
        fractional.sort(reverse=True)
        for j in range(min(remainder, n_slices)):
            int_alloc[fractional[j][1]] += 1

        # CRIT-026: Remove zero-quantity slices and redistribute their shares
        # to avoid gaps in the execution schedule.
        result = [q for q in int_alloc if q > 0]
        lost = total_qty - sum(result)
        if lost > 0 and result:
            # Distribute lost shares round-robin across remaining slices
            for j in range(lost):
                result[j % len(result)] += 1
        return result

    def _auto_duration(self, order_size: int, adv: float, urgency: float) -> float:
        """Auto-compute execution duration in seconds."""
        if adv <= 0:
            return 60.0  # 1 minute default

        participation = order_size / adv

        # Base duration: scale with participation rate
        # 0.1% ADV -> ~60s, 1% ADV -> ~300s, 5% ADV -> ~900s
        base_sec = max(30.0, min(1800.0, participation * 30_000))

        # Urgency compression: high urgency shortens duration
        urgency_mult = 1.0 - urgency * 0.7  # urgency=1 -> 30% of base
        return max(30.0, base_sec * urgency_mult)

    def _compute_num_slices(
        self, order_size: int, participation: float, urgency: float
    ) -> int:
        """Determine optimal number of slices."""
        if order_size <= 50:
            return 1
        if participation < 0.001:
            return max(1, min(3, order_size // 100))

        # More slices for larger participation, fewer for high urgency
        base = max(3, min(20, int(participation * 500)))
        urgency_adj = max(1, int(base * (1.0 - urgency * 0.5)))
        return urgency_adj

    def _estimate_slice_impact(
        self, slice_qty: int, total_qty: int, adv: float, volatility: float
    ) -> float:
        """Estimate temporary impact for a single slice in basis points."""
        if adv <= 0 or slice_qty <= 0:
            return 0.0

        participation = slice_qty / adv
        # Temporary impact: eta * sigma * (q/V)^alpha * 10000
        impact = (
            self._params.eta
            * volatility
            * math.pow(participation, self._params.alpha)
            * 10_000
        )
        return impact

    def _estimate_total_impact(
        self, order_size: int, adv: float, volatility: float, n_slices: int
    ) -> float:
        """Estimate total expected market impact in basis points."""
        if adv <= 0 or order_size <= 0:
            return 0.0

        participation = order_size / adv

        # Permanent impact
        permanent = self._params.gamma * participation * 10_000

        # Temporary impact (aggregate, reduced by splitting)
        temp_per_slice = (
            self._params.eta
            * volatility
            * math.pow(participation / max(1, n_slices), self._params.alpha)
            * 10_000
        )
        temporary = temp_per_slice * math.sqrt(n_slices)  # Sqrt aggregation

        return permanent + temporary

    def _estimate_timing_risk(
        self, order_size: int, volatility: float, duration_sec: float
    ) -> float:
        """Estimate timing risk (price variance exposure) in basis points."""
        if duration_sec <= 0 or volatility <= 0:
            return 0.0

        # Convert daily vol to per-second vol
        # Assume 6.5 trading hours = 23400 seconds
        vol_per_sec = volatility / math.sqrt(23400)

        # Timing risk = sigma * sqrt(T) * 10000
        risk = vol_per_sec * math.sqrt(duration_sec) * 10_000
        return risk

    @property
    def impact_params(self) -> ImpactParams:
        """Current impact parameters."""
        return self._params

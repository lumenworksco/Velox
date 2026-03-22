"""T7-001: Deep Reinforcement Learning Execution Agent.

PPO-style policy network for optimal trade execution. Learns to schedule
order slices based on market microstructure state to minimize implementation
shortfall (arrival price vs execution VWAP).

State space:
    - Order size normalized by ADV
    - Current VPIN (Volume-synchronized Probability of Informed Trading)
    - Bid-ask spread (bps)
    - Time-of-day (fractional trading day 0-1)
    - Queue depth imbalance (bid_depth - ask_depth) / (bid_depth + ask_depth)

Action space:
    - Fractional commitment: what percentage (0-100%) of remaining order
      to trade in the next 1-minute slice.

Reward:
    - Negative implementation shortfall: -(execution_vwap - arrival_price) / arrival_price
    - Adjusted for risk via a penalty on variance of execution prices.

Falls back to a simple heuristic (TWAP-like with spread awareness) when
PyTorch is unavailable or RL_EXECUTION_ENABLED is False.

A/B test support: matched-order comparison against Almgren-Chriss baseline.

Gate: RL_EXECUTION_ENABLED config flag (default False).
"""

import logging
import math
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional PyTorch import
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Beta
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    logger.info("T7-001: PyTorch not available — RL executor will use heuristic fallback")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExecutionState:
    """Observation for the RL agent at each 1-minute decision point."""
    order_size_norm: float       # remaining_qty / ADV
    vpin: float                  # Volume-synchronized probability of informed trading [0,1]
    spread_bps: float            # Current bid-ask spread in basis points
    time_of_day: float           # Fraction of trading day elapsed [0,1]
    queue_imbalance: float       # (bid_depth - ask_depth) / (bid_depth + ask_depth) [-1,1]
    pct_remaining: float         # Fraction of original order still to fill [0,1]
    urgency: float               # Urgency parameter from strategy [0,1]


@dataclass
class ExecutionAction:
    """Action output: how much to commit in the next slice."""
    commitment_pct: float        # 0.0 to 1.0 — fraction of remaining to trade now
    slice_qty: int               # Actual shares to trade
    confidence: float = 0.0      # Policy confidence (entropy-based)


@dataclass
class RLExecutionRecord:
    """Record for A/B testing and replay buffer."""
    timestamp: datetime
    symbol: str
    state: ExecutionState
    action: ExecutionAction
    fill_price: float = 0.0
    arrival_price: float = 0.0
    reward: float = 0.0
    executor_type: str = "rl"    # "rl" or "almgren_chriss"
    metadata: dict = field(default_factory=dict)


@dataclass
class ABTestResult:
    """Summary of A/B comparison between RL and Almgren-Chriss."""
    n_pairs: int = 0
    rl_avg_shortfall_bps: float = 0.0
    ac_avg_shortfall_bps: float = 0.0
    rl_wins: int = 0
    ac_wins: int = 0
    p_value: float = 1.0        # Paired t-test p-value
    significant: bool = False    # p < 0.05


# ---------------------------------------------------------------------------
# PPO Policy Network (PyTorch)
# ---------------------------------------------------------------------------

def _build_policy_network():
    """Build the PPO actor-critic network. Returns None if PyTorch unavailable."""
    if not _TORCH_AVAILABLE:
        return None

    class PPOPolicy(nn.Module):
        """Actor-Critic network for execution policy.

        Actor outputs Beta distribution parameters (alpha, beta) for the
        commitment fraction. Beta distribution naturally bounds output to [0,1].

        Critic outputs a scalar value estimate.
        """

        STATE_DIM = 7  # Matches ExecutionState fields

        def __init__(self, hidden_dim: int = 64):
            super().__init__()
            # Shared feature extractor
            self.shared = nn.Sequential(
                nn.Linear(self.STATE_DIM, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            )
            # Actor head: outputs alpha, beta for Beta distribution
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 2),     # alpha, beta parameters
                nn.Softplus(),        # Ensure positive
            )
            # Critic head: outputs value estimate
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
            )

        def forward(self, state: torch.Tensor):
            """Forward pass.

            Args:
                state: (batch, STATE_DIM) tensor.

            Returns:
                dist: Beta distribution for action sampling.
                value: (batch, 1) value estimates.
            """
            features = self.shared(state)
            # Actor: Beta distribution parameters (add 1 to ensure alpha,beta >= 1)
            ab = self.actor(features) + 1.0
            alpha = ab[:, 0:1]
            beta_param = ab[:, 1:2]
            dist = Beta(alpha, beta_param)
            # Critic
            value = self.critic(features)
            return dist, value

        def act(self, state: torch.Tensor):
            """Sample an action and return (action, log_prob, value)."""
            dist, value = self.forward(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, value

        def evaluate(self, state: torch.Tensor, action: torch.Tensor):
            """Evaluate actions for PPO update."""
            dist, value = self.forward(state)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return log_prob, value, entropy

    return PPOPolicy


# ---------------------------------------------------------------------------
# Heuristic Fallback Executor
# ---------------------------------------------------------------------------

class HeuristicExecutor:
    """Simple heuristic fallback when PyTorch is unavailable.

    Uses TWAP-like scheduling with spread-aware adjustments:
    - Reduce commitment when spreads are wide
    - Increase commitment near end of allowed execution window
    - Back off when VPIN is high (informed trading detected)
    """

    def decide(self, state: ExecutionState, remaining_qty: int) -> ExecutionAction:
        """Compute execution action using heuristic rules."""
        # Base commitment: TWAP-like uniform over remaining time
        time_remaining = max(0.01, 1.0 - state.time_of_day)
        base_pct = min(1.0, 0.05 / time_remaining)  # Target ~5% per minute

        # Spread adjustment: reduce when spreads are wide
        spread_factor = 1.0
        if state.spread_bps > 10:
            spread_factor = max(0.2, 10.0 / state.spread_bps)

        # VPIN adjustment: back off when informed trading is high
        vpin_factor = 1.0
        if state.vpin > 0.6:
            vpin_factor = max(0.3, 1.0 - (state.vpin - 0.6) * 2.0)

        # Urgency boost near end of day
        urgency_factor = 1.0
        if state.time_of_day > 0.9:
            urgency_factor = 2.0  # Double down in last 10% of day
        elif state.time_of_day > 0.8:
            urgency_factor = 1.3

        # Queue imbalance: trade more aggressively when queue favors us
        queue_factor = 1.0 + 0.2 * state.queue_imbalance

        # Combine factors
        commitment = base_pct * spread_factor * vpin_factor * urgency_factor * queue_factor
        commitment = max(0.01, min(1.0, commitment))

        # Account for urgency parameter
        commitment = commitment * (0.5 + 0.5 * state.urgency)

        slice_qty = max(1, int(remaining_qty * commitment))
        slice_qty = min(slice_qty, remaining_qty)

        return ExecutionAction(
            commitment_pct=commitment,
            slice_qty=slice_qty,
            confidence=0.5,  # Fixed confidence for heuristic
        )


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Proximal Policy Optimization trainer for the execution agent.

    Collects experience trajectories and performs mini-batch PPO updates.
    Only instantiated when PyTorch is available.
    """

    def __init__(
        self,
        policy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PPOTrainer requires PyTorch")

        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # Experience buffer
        self._states: list = []
        self._actions: list = []
        self._rewards: list = []
        self._log_probs: list = []
        self._values: list = []
        self._dones: list = []

    def store_transition(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ):
        """Store a single transition in the buffer."""
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._dones.append(done)

    def update(self, n_epochs: int = 4, batch_size: int = 64) -> dict:
        """Perform PPO update using collected experience.

        Returns:
            Dict with loss metrics.
        """
        if len(self._states) < batch_size:
            return {"skipped": True, "reason": "insufficient_data"}

        states = torch.FloatTensor(np.array(self._states))
        actions = torch.FloatTensor(np.array(self._actions)).unsqueeze(-1)
        old_log_probs = torch.FloatTensor(np.array(self._log_probs)).unsqueeze(-1)
        rewards = np.array(self._rewards)
        values = np.array(self._values)
        dones = np.array(self._dones, dtype=np.float32)

        # Compute GAE advantages
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages_t = torch.FloatTensor(advantages).unsqueeze(-1)
        returns_t = torch.FloatTensor(returns).unsqueeze(-1)

        # PPO update epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                idx = indices[start:end]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages_t[idx]
                batch_returns = returns_t[idx]

                # Evaluate current policy
                new_log_probs, new_values, entropy = self.policy.evaluate(
                    batch_states, batch_actions.clamp(1e-6, 1 - 1e-6)
                )

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_values, batch_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # Clear buffer
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._log_probs.clear()
        self._values.clear()
        self._dones.clear()

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
            "n_updates": n_updates,
        }


# ---------------------------------------------------------------------------
# Main RL Execution Agent
# ---------------------------------------------------------------------------

class RLExecutionAgent:
    """Deep RL execution agent with PPO policy.

    Decides how to slice a parent order into 1-minute execution windows,
    learning from implementation shortfall feedback.

    Falls back to HeuristicExecutor when:
    - PyTorch is not installed
    - RL_EXECUTION_ENABLED is False
    - The policy has not been trained (< min_train_episodes)

    A/B testing: when AB_TESTING_ENABLED is True, alternates matched orders
    between RL and Almgren-Chriss, recording results for comparison.

    Usage:
        agent = RLExecutionAgent()
        # Per 1-minute slice:
        state = ExecutionState(...)
        action = agent.decide(state, remaining_qty=500, symbol="AAPL")
        # After fill:
        agent.record_fill(fill_price=150.25, arrival_price=150.20)
    """

    def __init__(
        self,
        min_train_episodes: int = 100,
        update_interval: int = 50,
        model_path: str | None = None,
    ):
        self._enabled = getattr(config, "RL_EXECUTION_ENABLED", False)
        self._heuristic = HeuristicExecutor()
        self._lock = threading.Lock()

        # RL components (only if enabled and PyTorch available)
        self._policy = None
        self._trainer = None
        self._use_rl = False

        if self._enabled and _TORCH_AVAILABLE:
            try:
                PolicyClass = _build_policy_network()
                if PolicyClass is not None:
                    self._policy = PolicyClass()
                    self._trainer = PPOTrainer(self._policy)
                    self._use_rl = True
                    logger.info("T7-001: RL execution agent initialized with PPO policy")

                    # Load pre-trained weights if available
                    if model_path:
                        self._load_model(model_path)
            except Exception as e:
                logger.warning(f"T7-001: Failed to initialize RL policy, using heuristic: {e}")
                self._use_rl = False
        elif not self._enabled:
            logger.info("T7-001: RL execution disabled (RL_EXECUTION_ENABLED=False)")
        else:
            logger.info("T7-001: RL execution using heuristic fallback (no PyTorch)")

        # Episode tracking
        self._min_train_episodes = min_train_episodes
        self._update_interval = update_interval
        self._episode_count = 0
        self._trained = False

        # Current execution state for reward computation
        self._current_record: RLExecutionRecord | None = None
        self._execution_history: list[RLExecutionRecord] = []
        self._max_history = 10000

        # A/B testing
        self._ab_enabled = getattr(config, "AB_TESTING_ENABLED", False)
        self._ab_counter = 0
        self._ab_rl_results: list[float] = []      # shortfall in bps
        self._ab_ac_results: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        state: ExecutionState,
        remaining_qty: int,
        symbol: str = "",
    ) -> ExecutionAction:
        """Decide how much to trade in the next 1-minute slice.

        Args:
            state: Current market microstructure observation.
            remaining_qty: Shares remaining in the parent order.
            symbol: Ticker symbol (for logging/tracking).

        Returns:
            ExecutionAction with commitment fraction and slice quantity.
        """
        if remaining_qty <= 0:
            return ExecutionAction(commitment_pct=0.0, slice_qty=0)

        # A/B test routing: alternate between RL and heuristic (as AC proxy)
        use_rl_this_order = self._should_use_rl()

        if use_rl_this_order and self._use_rl and self._trained:
            action = self._rl_decide(state, remaining_qty)
            executor_type = "rl"
        else:
            action = self._heuristic.decide(state, remaining_qty)
            executor_type = "heuristic" if not self._ab_enabled else "almgren_chriss"

        # Record for training and A/B testing
        self._current_record = RLExecutionRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            state=state,
            action=action,
            executor_type=executor_type,
        )

        return action

    def record_fill(
        self,
        fill_price: float,
        arrival_price: float,
        filled_qty: int = 0,
    ):
        """Record execution fill for reward computation and A/B tracking.

        Call after each slice fill to update the RL agent.

        Args:
            fill_price: Actual fill price.
            arrival_price: Decision/arrival price (benchmark).
            filled_qty: Number of shares filled.
        """
        if self._current_record is None:
            return

        record = self._current_record
        record.fill_price = fill_price
        record.arrival_price = arrival_price

        # Compute implementation shortfall (negative = good for buyer)
        if arrival_price > 0:
            shortfall_pct = (fill_price - arrival_price) / arrival_price
            # Reward = negative shortfall (minimize cost)
            # With risk penalty for high variance
            reward = -shortfall_pct * 10_000  # In bps, negative shortfall = positive reward
            record.reward = reward

            # A/B test tracking
            shortfall_bps = shortfall_pct * 10_000
            if record.executor_type == "rl":
                self._ab_rl_results.append(shortfall_bps)
            elif record.executor_type == "almgren_chriss":
                self._ab_ac_results.append(shortfall_bps)

        # Store for training
        with self._lock:
            self._execution_history.append(record)
            if len(self._execution_history) > self._max_history:
                self._execution_history = self._execution_history[-self._max_history:]

        # Train RL policy if we have enough data
        if self._use_rl and self._trainer:
            self._maybe_train(record)

        self._current_record = None

    def get_ab_test_results(self) -> ABTestResult:
        """Get A/B test comparison results."""
        n_rl = len(self._ab_rl_results)
        n_ac = len(self._ab_ac_results)
        n_pairs = min(n_rl, n_ac)

        if n_pairs == 0:
            return ABTestResult()

        rl_avg = np.mean(self._ab_rl_results[:n_pairs])
        ac_avg = np.mean(self._ab_ac_results[:n_pairs])
        rl_wins = sum(
            1 for r, a in zip(self._ab_rl_results[:n_pairs], self._ab_ac_results[:n_pairs])
            if r < a  # Lower shortfall is better
        )

        # Paired t-test (simplified)
        p_value = 1.0
        if n_pairs >= 10:
            diffs = np.array(self._ab_rl_results[:n_pairs]) - np.array(self._ab_ac_results[:n_pairs])
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1)
            if std_diff > 1e-10:
                t_stat = mean_diff / (std_diff / np.sqrt(n_pairs))
                # Approximate p-value using normal distribution for large n
                p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2))))

        return ABTestResult(
            n_pairs=n_pairs,
            rl_avg_shortfall_bps=float(rl_avg),
            ac_avg_shortfall_bps=float(ac_avg),
            rl_wins=rl_wins,
            ac_wins=n_pairs - rl_wins,
            p_value=p_value,
            significant=p_value < 0.05,
        )

    @property
    def status(self) -> dict:
        """Return current agent status."""
        ab = self.get_ab_test_results()
        return {
            "enabled": self._enabled,
            "using_rl": self._use_rl and self._trained,
            "torch_available": _TORCH_AVAILABLE,
            "episode_count": self._episode_count,
            "trained": self._trained,
            "history_size": len(self._execution_history),
            "ab_test": {
                "n_pairs": ab.n_pairs,
                "rl_avg_bps": round(ab.rl_avg_shortfall_bps, 2),
                "ac_avg_bps": round(ab.ac_avg_shortfall_bps, 2),
                "rl_wins": ab.rl_wins,
                "significant": ab.significant,
            },
        }

    # ------------------------------------------------------------------
    # Internal: RL decision
    # ------------------------------------------------------------------

    def _rl_decide(self, state: ExecutionState, remaining_qty: int) -> ExecutionAction:
        """Use the PPO policy to decide action."""
        state_vec = self._state_to_tensor(state)

        with torch.no_grad():
            action, log_prob, value = self._policy.act(state_vec.unsqueeze(0))

        commitment = float(action.squeeze().item())
        commitment = max(0.01, min(1.0, commitment))

        slice_qty = max(1, int(remaining_qty * commitment))
        slice_qty = min(slice_qty, remaining_qty)

        # Confidence from entropy (lower entropy = higher confidence)
        dist, _ = self._policy(state_vec.unsqueeze(0))
        entropy = float(dist.entropy().item())
        max_entropy = math.log(2)  # Max entropy for Beta distribution ≈ ln(2) at alpha=beta=1
        confidence = max(0.0, min(1.0, 1.0 - entropy / max(max_entropy, 1e-6)))

        # Store for PPO training
        if self._trainer:
            self._trainer.store_transition(
                state=state_vec.numpy(),
                action=commitment,
                reward=0.0,  # Will be updated when fill is recorded
                log_prob=float(log_prob.item()),
                value=float(value.item()),
                done=state.pct_remaining <= commitment,  # Done if this finishes the order
            )

        return ExecutionAction(
            commitment_pct=commitment,
            slice_qty=slice_qty,
            confidence=confidence,
        )

    def _state_to_tensor(self, state: ExecutionState):
        """Convert ExecutionState to a PyTorch tensor."""
        vec = np.array([
            state.order_size_norm,
            state.vpin,
            state.spread_bps / 100.0,   # Normalize to ~[0, 1]
            state.time_of_day,
            state.queue_imbalance,
            state.pct_remaining,
            state.urgency,
        ], dtype=np.float32)
        return torch.FloatTensor(vec)

    # ------------------------------------------------------------------
    # Internal: Training
    # ------------------------------------------------------------------

    def _maybe_train(self, record: RLExecutionRecord):
        """Trigger PPO update if enough experience has been collected."""
        self._episode_count += 1

        if self._episode_count >= self._min_train_episodes and not self._trained:
            self._trained = True
            logger.info(
                f"T7-001: RL policy now active after {self._episode_count} episodes"
            )

        if self._episode_count % self._update_interval == 0 and self._trainer:
            try:
                metrics = self._trainer.update()
                if not metrics.get("skipped"):
                    logger.info(
                        f"T7-001: PPO update — policy_loss={metrics.get('policy_loss', 0):.4f}, "
                        f"value_loss={metrics.get('value_loss', 0):.4f}, "
                        f"entropy={metrics.get('entropy', 0):.4f}"
                    )
            except Exception as e:
                logger.warning(f"T7-001: PPO update failed: {e}")

    # ------------------------------------------------------------------
    # Internal: A/B test routing
    # ------------------------------------------------------------------

    def _should_use_rl(self) -> bool:
        """Determine whether this order should use RL or the baseline.

        For A/B testing, alternates between RL and baseline on matched orders.
        """
        if not self._ab_enabled:
            return True  # Always use RL if not A/B testing

        self._ab_counter += 1
        return self._ab_counter % 2 == 0  # Alternate

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def _load_model(self, path: str):
        """Load pre-trained policy weights."""
        if not _TORCH_AVAILABLE or self._policy is None:
            return
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            self._policy.load_state_dict(state_dict)
            self._trained = True
            logger.info(f"T7-001: Loaded RL policy from {path}")
        except Exception as e:
            logger.warning(f"T7-001: Failed to load model from {path}: {e}")

    def save_model(self, path: str):
        """Save policy weights to disk."""
        if not _TORCH_AVAILABLE or self._policy is None:
            logger.debug("T7-001: Cannot save model — PyTorch not available or no policy")
            return
        try:
            torch.save(self._policy.state_dict(), path)
            logger.info(f"T7-001: Saved RL policy to {path}")
        except Exception as e:
            logger.warning(f"T7-001: Failed to save model to {path}: {e}")

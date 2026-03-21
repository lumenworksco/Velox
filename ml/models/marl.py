"""
EDGE-002: Multi-Agent Reinforcement Learning (MARL) for Trading
================================================================

Three cooperating agents jointly decide trading actions:

  1. **Alpha Agent** -- generates directional signals (long / short / flat)
  2. **Risk Agent**  -- sizes positions given the signal and portfolio state
  3. **Execution Agent** -- decides order timing (now / wait / cancel)

Each agent maintains its own Q-table (tabular) or small neural policy network.
A coordinator merges their outputs into a single actionable decision.

The framework is intentionally lightweight so it can run without heavy RL
libraries.  If PyTorch is available, agents use small policy-gradient networks;
otherwise, they fall back to epsilon-greedy Q-tables backed by NumPy.

Conforms to the AlphaModel interface:
    fit(X, y)     -- train agents via episode replay
    predict(X)    -- produce trading signals
    score(X, y)   -- evaluate signal quality
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PyTorch for neural policy variant
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    logger.info(
        "EDGE-002: PyTorch not found. Agents will use tabular Q-learning."
    )


# ===================================================================
# Q-Table Agent (numpy-only fallback)
# ===================================================================

class QTableAgent:
    """Epsilon-greedy Q-learning agent with discrete state/action spaces.

    States are discretized into bins; actions are a small finite set.
    """

    def __init__(self, name: str, n_state_bins: int, n_actions: int,
                 lr: float = 0.1, gamma: float = 0.99, epsilon: float = 0.15):
        self.name = name
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table: (state_hash) -> np.array of shape (n_actions,)
        self._q: Dict[int, np.ndarray] = {}
        self.n_state_bins = n_state_bins

    def _discretize(self, state: np.ndarray) -> int:
        """Hash a continuous state vector into a discrete bin index."""
        clipped = np.clip(state, -3.0, 3.0)
        bins = ((clipped + 3.0) / 6.0 * self.n_state_bins).astype(int)
        bins = np.clip(bins, 0, self.n_state_bins - 1)
        return hash(bins.tobytes())

    def _get_q(self, key: int) -> np.ndarray:
        if key not in self._q:
            self._q[key] = np.zeros(self.n_actions, dtype=np.float64)
        return self._q[key]

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        key = self._discretize(state)
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self._get_q(key)))

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> float:
        key = self._discretize(state)
        next_key = self._discretize(next_state)
        q = self._get_q(key)
        next_q = self._get_q(next_key)
        target = reward + (0.0 if done else self.gamma * np.max(next_q))
        td_error = target - q[action]
        q[action] += self.lr * td_error
        return td_error

    def decay_epsilon(self, factor: float = 0.995, minimum: float = 0.01):
        self.epsilon = max(minimum, self.epsilon * factor)


# ===================================================================
# Neural Policy Agent (when PyTorch is available)
# ===================================================================

if _TORCH_AVAILABLE:

    class PolicyNetwork(nn.Module):
        """Small 2-layer policy network for REINFORCE."""

        def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.head = nn.Linear(hidden, n_actions)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return F.softmax(self.head(x), dim=-1)


    class NeuralPolicyAgent:
        """REINFORCE policy-gradient agent."""

        def __init__(self, name: str, state_dim: int, n_actions: int,
                     lr: float = 1e-3, gamma: float = 0.99):
            self.name = name
            self.n_actions = n_actions
            self.gamma = gamma
            self.net = PolicyNetwork(state_dim, n_actions)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
            self._log_probs: List[torch.Tensor] = []
            self._rewards: List[float] = []

        def select_action(self, state: np.ndarray, explore: bool = True) -> int:
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = self.net(s)
            if explore:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                self._log_probs.append(dist.log_prob(action))
            else:
                action = probs.argmax(dim=-1)
            return int(action.item())

        def store_reward(self, reward: float):
            self._rewards.append(reward)

        def update_episode(self) -> float:
            """Run REINFORCE update at end of episode. Returns mean loss."""
            if not self._log_probs:
                return 0.0
            # Compute discounted returns
            returns = []
            G = 0.0
            for r in reversed(self._rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            if returns.std() > 1e-8:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            loss = torch.tensor(0.0)
            for lp, G_t in zip(self._log_probs, returns):
                loss -= lp * G_t

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()

            mean_loss = loss.item() / max(len(self._log_probs), 1)
            self._log_probs.clear()
            self._rewards.clear()
            return mean_loss


# ===================================================================
# Agent Coordinator
# ===================================================================

class AgentCoordinator:
    """Combines outputs of alpha, risk, and execution agents into a
    single trading signal in [-1, 1]."""

    # Alpha actions: 0=short, 1=flat, 2=long
    ALPHA_MAP = {0: -1.0, 1: 0.0, 2: 1.0}
    # Risk actions: 0=small, 1=medium, 2=large
    RISK_MAP = {0: 0.25, 1: 0.5, 2: 1.0}
    # Execution actions: 0=cancel, 1=wait, 2=execute_now
    EXEC_MAP = {0: 0.0, 1: 0.5, 2: 1.0}

    @staticmethod
    def combine(alpha_action: int, risk_action: int, exec_action: int) -> float:
        """Return composite signal in [-1, 1]."""
        direction = AgentCoordinator.ALPHA_MAP.get(alpha_action, 0.0)
        size = AgentCoordinator.RISK_MAP.get(risk_action, 0.5)
        urgency = AgentCoordinator.EXEC_MAP.get(exec_action, 1.0)
        return float(np.clip(direction * size * urgency, -1.0, 1.0))


# ===================================================================
# Reward shaper
# ===================================================================

def compute_step_reward(signal: float, actual_return: float,
                        transaction_cost: float = 0.0005) -> float:
    """PnL-based reward with transaction penalty."""
    pnl = signal * actual_return
    cost = abs(signal) * transaction_cost
    return pnl - cost


# ===================================================================
# Public API: MARLTradingModel (AlphaModel interface)
# ===================================================================

class MARLTradingModel:
    """Multi-Agent RL model for trading signal generation.

    Parameters
    ----------
    n_features : int
        Number of features per observation.
    use_neural : bool or None
        Force neural (True) or tabular (False) agents.
        None = auto-detect (neural if PyTorch available).
    n_episodes : int
        Training episodes for fit().
    episode_len : int
        Steps per episode (sampled from data).
    gamma : float
        Discount factor.
    lr : float
        Learning rate.
    """

    def __init__(self, *, n_features: int = 10, use_neural: Optional[bool] = None,
                 n_episodes: int = 200, episode_len: int = 50,
                 gamma: float = 0.99, lr: float = 0.01,
                 n_state_bins: int = 20, **kwargs):
        self.n_features = n_features
        self.use_neural = use_neural if use_neural is not None else _TORCH_AVAILABLE
        self.n_episodes = n_episodes
        self.episode_len = episode_len
        self.gamma = gamma
        self.lr = lr
        self.n_state_bins = n_state_bins
        self._fitted = False

        self._build_agents()

    def _build_agents(self):
        if self.use_neural and _TORCH_AVAILABLE:
            logger.info("EDGE-002: Building neural policy agents.")
            self.alpha_agent = NeuralPolicyAgent("alpha", self.n_features, 3, self.lr, self.gamma)
            self.risk_agent = NeuralPolicyAgent("risk", self.n_features + 1, 3, self.lr, self.gamma)
            self.exec_agent = NeuralPolicyAgent("exec", self.n_features + 2, 3, self.lr, self.gamma)
        else:
            if self.use_neural and not _TORCH_AVAILABLE:
                logger.warning("EDGE-002: Neural requested but PyTorch unavailable. Using Q-tables.")
            logger.info("EDGE-002: Building tabular Q-learning agents.")
            self.alpha_agent = QTableAgent("alpha", self.n_state_bins, 3, self.lr, self.gamma)
            self.risk_agent = QTableAgent("risk", self.n_state_bins, 3, self.lr, self.gamma)
            self.exec_agent = QTableAgent("exec", self.n_state_bins, 3, self.lr, self.gamma)

    # ------------------------------------------------------------------
    # AlphaModel interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MARLTradingModel":
        """Train agents via episodic replay on historical data.

        Args:
            X: (n_samples, n_features) — feature matrix.
            y: (n_samples,) — next-bar returns used as reward signal.
        """
        X = np.asarray(X, dtype=np.float64)
        if y is None:
            raise ValueError("EDGE-002: y (returns) required for training.")
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples = X.shape[0]
        is_neural = self.use_neural and _TORCH_AVAILABLE

        for ep in range(self.n_episodes):
            start = np.random.randint(0, max(1, n_samples - self.episode_len))
            ep_reward = 0.0

            for t in range(self.episode_len):
                idx = start + t
                if idx >= n_samples:
                    break

                state = X[idx]

                # Alpha agent acts on raw features
                a_alpha = self.alpha_agent.select_action(state)

                # Risk agent sees features + alpha action
                risk_state = np.append(state, a_alpha)
                a_risk = self.risk_agent.select_action(risk_state)

                # Exec agent sees features + alpha + risk
                exec_state = np.append(state, [a_alpha, a_risk])
                a_exec = self.exec_agent.select_action(exec_state)

                # Compute combined signal and reward
                signal = AgentCoordinator.combine(a_alpha, a_risk, a_exec)
                reward = compute_step_reward(signal, y[idx])
                ep_reward += reward

                # Update agents
                done = (t == self.episode_len - 1) or (idx + 1 >= n_samples)
                next_idx = min(idx + 1, n_samples - 1)
                next_state = X[next_idx]

                if is_neural:
                    self.alpha_agent.store_reward(reward)
                    self.risk_agent.store_reward(reward)
                    self.exec_agent.store_reward(reward)
                else:
                    self.alpha_agent.update(state, a_alpha, reward, next_state, done)
                    self.risk_agent.update(risk_state, a_risk, reward,
                                           np.append(next_state, a_alpha), done)
                    self.exec_agent.update(exec_state, a_exec, reward,
                                           np.append(next_state, [a_alpha, a_risk]), done)

            # End-of-episode updates
            if is_neural:
                self.alpha_agent.update_episode()
                self.risk_agent.update_episode()
                self.exec_agent.update_episode()
            else:
                self.alpha_agent.decay_epsilon()
                self.risk_agent.decay_epsilon()
                self.exec_agent.decay_epsilon()

            if (ep + 1) % 50 == 0:
                logger.debug("EDGE-002 episode %d/%d  reward=%.4f",
                             ep + 1, self.n_episodes, ep_reward)

        self._fitted = True
        logger.info("EDGE-002: MARL trained for %d episodes on %d samples.",
                     self.n_episodes, n_samples)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate trading signals in [-1, 1].

        Args:
            X: (n_samples, n_features)

        Returns:
            signals: (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        signals = np.empty(X.shape[0], dtype=np.float64)
        for i, state in enumerate(X):
            a_alpha = self.alpha_agent.select_action(state, explore=False)
            risk_state = np.append(state, a_alpha)
            a_risk = self.risk_agent.select_action(risk_state, explore=False)
            exec_state = np.append(state, [a_alpha, a_risk])
            a_exec = self.exec_agent.select_action(exec_state, explore=False)
            signals[i] = AgentCoordinator.combine(a_alpha, a_risk, a_exec)

        return signals

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate as mean PnL of predicted signals against actual returns."""
        signals = self.predict(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        pnl = signals * y
        return float(np.mean(pnl))

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_features": self.n_features,
            "use_neural": self.use_neural,
            "n_episodes": self.n_episodes,
            "episode_len": self.episode_len,
            "gamma": self.gamma,
            "lr": self.lr,
        }

    def __repr__(self) -> str:
        mode = "neural" if (self.use_neural and _TORCH_AVAILABLE) else "tabular"
        status = "fitted" if self._fitted else "unfitted"
        return f"MARLTradingModel(mode={mode}, {status})"

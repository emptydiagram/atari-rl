# Background

Recall the MDP setup: there's an agent, who makes observations of the environment and takes actions, that can possibly change the state of the environment. Upon each new observation, additionally a reward value is received. Hence, over time, the agent receives a sequence of rewards over time, each of which comes in response to an action, and an agent must use this ability to take actions in an environment to learn how to maximize the total reward.

There are three collections, the states $\mathcal{S}$, the actions $\mathcal{A}$, and the possible rewards $\mathcal{R}$, all assumed finite. The agent observes an initial state $s_0$, and must choose an initial action $a_0$. Subsequently, for any $t \geq 0$, having observed $s_t$ and taken action $a_t$, the agent receives $(r_{t+1}, s_{t+1})$. The sequence of states observed, actions taken, and rewards received looks like:

$$(s_0, a_0, r_1, s_1, a_1, \ldots, r_t, s_t, a_t, \ldots)$$

Atari games are organized into episodes, and in each episode there is some step $T$ when the sequence terminates:

$$(s_0, a_0, r_1, s_1, a_1, \ldots, r_T, s_T)$$

You can also model as a sequence of random variables, $S_t, A_t, R_t$ for all $t$:

$$(S_0, A_0, R_1, S_1, A_1, \ldots, R_T, S_T)$$

There is a (possibly unknown) conditional joint probability distribution $p(R, S' | s, a)$ over the next reward and next state , $(R, S')$, given state-action pair $(s, a)$.

By marginalizing over $r \in \mathcal{R}$, we can obtain the **state-transition distribution**:

$$p(s' | s, a) = \sum_{r \in \mathcal{R}} p(r, s' | s, a)$$

The agent is assumed to have a conditional probability distribution over actions $\mathcal{A}$ given a state $s$, called the agent's **policy**: $\pi(\cdot | s)$

The **total discounted future reward** starting at time step $t$, using discount rate $\gamma \in \mathbb{R}$, $0 \leq \gamma \leq 1$, is:

$$G_t := \sum_{k = t+1}^{T} \gamma^{k - t - 1} r_{k}$$

This obeys the recursive equation:

$$G_t = r_{t+1} + \gamma G_{t+1}$$

Reward is often a deterministic function of $(s, a)$ ($r: \mathcal{S} \times \mathcal{A} \to \mathcal{R}$) or of $(s, a, s')$ ($r: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathcal{R}$). In this case:

$$p(q, s' | s, a) = \begin{cases}
1 & \text{if } q = r(s, a) \ [r(s, a, s')] \\
0 & \text{otherwise }
\end{cases}$$

However, even in the case when the reward is stochastic, we can define:

$$r(s, a) := \mathbb{E}[R_t | S_t = s, A_t = a] = \sum_r r \cdot \sum_{s'} p(r, s' | s, a)$$

$$r(s, a, s') := \mathbb{E}[R_t | S_t = s, A_t = a, S_{t+1} = s'] \sum_r r \cdot p(r | s, a, s') = \sum_r r \cdot \frac{p(r, s' | s, a)}{p(s' | s, a)}$$

For a given policy $\pi$, we can define functions $V^{\pi}: \mathcal{S} \to \mathbb{R}$ and $Q^{\pi}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$:

$$V^{\pi}(s) := \mathbb{E}_{\pi}[ G_t | S_t = s] = \mathbb{E}_{\pi}[ \sum_{k = t+1}^T \gamma^{k - t - 1} r_{k} | S_t = s ]$$

$$Q^{\pi}(s, a) := \mathbb{E}_{\pi}[ G_t | S_t = s, A_t = a] = \mathbb{E}_{\pi}[ \sum_{k = t+1}^T \gamma^{k - t - 1} r_{k} | S_t = s, A_t = a ]$$

We can define optimal value functions:

$$V^{\ast}(s) := \max_{\pi} V^{\pi}(s)$$

$$Q^{\ast}(s, a) := \max_a Q^{\pi}(s, a)$$


# Vanilla DQN and (some of) its extensions


In order to perform model-free temporal difference learning, we model the Q function $Q: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ estimating the value of each (state, action) pair $(s, a) \in \mathcal{S} \times \mathcal{A}$.

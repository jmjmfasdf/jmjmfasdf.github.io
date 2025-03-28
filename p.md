# REINFORCE 흐름도

┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. Initialize policy network π(a|s; θ)                                      │
│    - Outputs a probability distribution over actions given state s           │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. For each episode:                                                         │
│    - Reset environment and get initial state s₀                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. For each time step t in the episode:                                     │
│   1) Sample action from current policy:                                      │
│        aₜ ~ π(a|sₜ; θ)                                                      │
│   2) Execute action aₜ in the environment                                   │
│        → observe reward rₜ₊₁ and next state sₜ₊₁                             │
│   3) Store (sₜ, aₜ, rₜ₊₁) in trajectory buffer                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. After episode ends (trajectory complete):                                │
│   For each time step t in the trajectory:                                   │
│   1) Compute return Gₜ = ∑_{k=0}^{T−t−1} γ^k · r_{t+k+1}                     │
│        → This is the total future discounted reward from time t             │
│   2) Compute policy loss (gradient ascent):                                 │
│        Lₜ = -log π(aₜ|sₜ; θ) · Gₜ                                           │
│        → Encourages actions that led to high return                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. Backpropagate total loss and update policy network:                      │
│     θ ← θ - α ∇θ ∑_t Lₜ                                                     │
│     → Gradient ascent to maximize expected return                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. Repeat for next episode                                                  │
└──────────────────────────────────────────────────────────────────────────────┘


# Dueling DQN 흐름도

┌─────────────────────────────────────────────┐
│ Initialize main Q-network θ                 │
│ - Shared layers                             │
│ - Value stream:      V(s; θ_V)              │
│ - Advantage stream:  A(s, a; θ_A)           │
│ Combine: Q(s,a) = V(s) + [A(s,a) - mean(A)] │
└─────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────┐
│ Initialize target network θ⁻ ← θ            │
│ (for stable target computation)             │
└─────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────┐
│ For each episode:                           │
└─────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────┐
│ For each time step t:                       │
│ 1. Observe current state sₜ                  │
│ 2. Compute Q(sₜ, a; θ) for all a             │ ← ✅ Policy Evaluation (Value Estimation)
│ 3. Choose action aₜ ~ ε-greedy(Q(sₜ, a))      │ ← 🎯 Policy Improvement (implicit via ε-greedy)
│ 4. Take action aₜ, observe rₜ₊₁, sₜ₊₁         │
│ 5. Store (sₜ, aₜ, rₜ₊₁, sₜ₊₁) in buffer        │
└─────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────┐
│ Sample random mini-batch from replay buffer │  ← for desired td update step(normally each time step ends;)
└─────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────┐
│ For each sample (s, a, r, s'):              │
│ 1. Compute target:                          │
│    y = r + γ·max_{a'} Q(s', a'; θ⁻)         │ ← ✅ TD Target (Value Improvement)
│ 2. Compute predicted Q(s, a; θ)             │
│ 3. Compute loss:                            │
│    L = (Q(s, a; θ) - y)²                    │
└─────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────┐
│ Backpropagate loss, update θ                │ ← ✅ Gradient Step (Value Network Improvement)
└─────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────┐
│ Periodically update target network θ⁻ ← θ   │
└─────────────────────────────────────────────┘


우리가 최적화하고자 하는 것은 다음과 같은 expected return입니다:

$$
J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} r_t \right]
$$

우선 trajectory의 확률을 생각해봅시다. 정책에 따라 어떤 trajectory τ가 나올 확률은 다음과 같이 쓸 수 있어요:

$$
P(\tau) = \rho(s_0) \prod_{t=0}^{T} \pi_\theta(a_t | s_t) \cdot p(s_{t+1} | s_t, a_t)
$$

따라서 기대 return은 다음처럼 쓸 수 있습니다:

$$
J(\theta) = \int_\tau P(\tau) R(\tau) d\tau
$$

기대값의 gradient는 다음과 같이 미분할 수 있어요:

$$
\nabla_\theta J(\theta)
= \nabla_\theta \int_\tau P(\tau) R(\tau) d\tau
= \int_\tau \nabla_\theta P(\tau) R(\tau) d\tau
$$

여기서 ∇θP(τ)에 log-derivative trick을 적용합니다:

$$
\nabla_\theta P(\tau) = P(\tau) \cdot \nabla_\theta \log P(\tau)
$$

환경의 dynamics는 정확히 알 수 없기 때문에 경로적분과 합쳐서 estimation에 대한 표현으로 바꾸기:

$$
\nabla_\theta J(\theta)
= \int_\tau P(\tau) \nabla_\theta \log P(\tau) R(\tau) d\tau
= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau) \cdot R(\tau) \right]
$$

환경의 dynamics 𝑝(𝑠′∣𝑠,𝑎)는 𝜃와 무관하기 때문에 미분할 필요가 없습니다. 그래서 log𝑃(𝜏)에서 정책 부분만 남습니다:

$$
\log P(\tau) = \log \left( \rho(s_0) \prod_{t=0}^{T} \pi_\theta(a_t | s_t) \cdot p(s_{t+1} | s_t, a_t) \right)
= \sum_{t=0}^{T} \log \pi_\theta(a_t | s_t) + \text{(정책과 무관한 부분)}
$$

따라서 미분해도 정책 부분만 남습니다:

$$
\nabla_\theta \log P(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t)
$$

최종적으로 다음과 같이 정리됩니다:

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau \sim \pi_\theta} \left[
    \left( \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \right)
    \cdot R(\tau)
\right]
$$

이를 시간 t마다 나눠서 쓰면:

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau \sim \pi_\theta} \left[
    \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t
\right]
$$

$$
\boxed{
\nabla_\theta J(\theta)
= \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]
}
$$

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi}(s, a) \right]
$$

# DDPG의 흐름도

┌────────────────────────────────────────────┐
│ Initialize actor network μ(s; θ^μ)         │
│ Initialize critic network Q(s, a; θ^Q)     │
│ Initialize target networks:                │
│     μ' ← μ,     Q' ← Q                     │
└────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────┐
│ Initialize replay buffer D                 │
└────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────┐
│ For each episode:                          │
│   Initialize state s₀                      │
└────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────┐
│ For each time step t:                      │
│ 1. Select action:                          │
│    aₜ = μ(sₜ; θ^μ) + noise (exploration)   │ ← 🎯 행동 선택 (Deterministic Policy + Noise)  
│ 2. Execute action aₜ, observe rₜ₊₁, sₜ₊₁   │
│ 3. Store (sₜ, aₜ, rₜ₊₁, sₜ₊₁) in buffer D │
└────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────┐
│ Sample random mini-batch from buffer D     │
└────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────┐
│ For each sample (s, a, r, s'):             │
│ 1. Compute target action:                  │
│     a' = μ'(s'; θ^{μ'})                     │
│ 2. Compute TD target:                      │
│     y = r + γ·Q'(s', a'; θ^{Q'})            │ ← ✅ TD Target (Policy Evaluation)
│ 3. Compute critic loss:                    │
│     L = (Q(s,a; θ^Q) - y)²                  │
│ 4. Update critic θ^Q ← minimize L          │ ← ✅ Value Network Update
└────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────┐
│ 5. Compute actor gradient:                 │
│     ∇θ^μ J ≈ ∇_a Q(s,a; θ^Q) · ∇_θ^μ μ(s)   │ ← ✅ Policy Gradient (Deterministic)
│ 6. Update actor θ^μ ← gradient ascent      │ ← ✅ Policy Improvement
└────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────┐
│ Soft update target networks:               │
│     θ^{μ'} ← τ θ^μ + (1−τ) θ^{μ'}          │
│     θ^{Q'} ← τ θ^Q + (1−τ) θ^{Q'}          │
└────────────────────────────────────────────┘


# Advantage Actor Critic

┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. Initialize networks                                                       │
│   - Actor network:       π(a|s; θ^π) → outputs action probabilities          │
│   - Critic network:      V(s; θ^V) → outputs scalar state value              │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. For each episode:                                                         │
│     - Reset environment and get initial state s₀                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. For each time step t in the episode:                                     │
│   1) Sample action from current policy:                                      │
│        aₜ ~ π(a|sₜ; θ^π)                                                    │
│   2) Execute action aₜ in the environment                                   │
│        → observe reward rₜ₊₁ and next state sₜ₊₁                             │
│   3) Store (sₜ, aₜ, rₜ₊₁, sₜ₊₁) in trajectory buffer                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. At update step (after fixed steps or episode end):                       │
│   For each transition (s, a, r, s') in trajectory:                          │
│   1) Compute TD target (bootstrapped return):                               │
│        R = r + γ · V(s'; θ^V)                                               │
│   2) Compute advantage:                                                     │
│        A = R - V(s; θ^V)                                                    │
│   3) Compute actor loss:                                                    │
│        L_actor = -log π(a|s; θ^π) · A                                       │
│   4) Compute critic loss:                                                   │
│        L_critic = (R - V(s))²                                               │
│   5) Compute total loss:                                                    │
│        L_total = L_actor + c_v · L_critic - c_e · H[π(a|s)]                 │
│        (Optional: include entropy bonus H to encourage exploration)         │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. Backpropagate total loss and update parameters:                          │
│     θ^π ← θ^π - α_π ∇θ^π L_actor                                             │
│     θ^V ← θ^V - α_V ∇θ^V L_critic                                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. Repeat for next time step or episode                                     │
└──────────────────────────────────────────────────────────────────────────────┘

# TRPO

일반 policy gradient는 다음 식에 따라 단순히 확률을 높이는 방향으로 학습합니다:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A^\pi(s, a) \right]
$$

하지만 이 방식은 한 번의 업데이트로 policy가 너무 크게 변할 수 있고 결과적으로 성능이 오히려 떨어지는 catastrophic collapse가 발생할 수 있다

TRPO의 목적함수

$$
\max_\theta \quad \mathbb{E}_{s,a \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s,a) \right]
$$

기존 policy를 기준으로 sample을 수집하고, 새 policy의 performance 개선을 기대값으로 평가하도록 하며 정책이 너무 많이 바뀌지 않도록 KL divergence constraint를 둠

┌──────────────────────────────────────────────────────────────────────────────┐
│ Example: TRPO Agent in a CartPole Environment                               │
│ (Actor-Critic Structure: Separate Policy and Value Networks)                │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. Initialize networks:                                                     │
│   - Actor network π(a|s; θ^π): stochastic policy                            │
│   - Critic network V(s; θ^V): value estimator for baseline/advantage        │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. For each iteration (e.g., every 2048 steps):                             │
│   - Reset environment and observe initial state s₀                         │
│   - Roll out episodes using current policy π_old (θ_old):                  │
│       For each step t:                                                      │
│         - Sample action aₜ ~ π(a|sₜ; θ_old)                                 │
│         - Execute action aₜ, observe rₜ₊₁ and sₜ₊₁                          │
│         - Store (sₜ, aₜ, rₜ₊₁, sₜ₊₁) in buffer                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. Compute returns and value predictions:                                   │
│   - Compute returns Rₜ = ∑ γ^k · rₜ₊ₖ                                       │
│   - Use Critic network to get V(sₜ; θ^V)                                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. Compute advantage estimates A(s,a):                                     │
│   Aₜ = Rₜ − V(sₜ; θ^V)                                                      │
│   - Or use Generalized Advantage Estimation (GAE)                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. Compute surrogate policy loss:                                           │
│   L_actor = E[ (π_θ(a|s)/π_old(a|s)) · A(s,a) ]                             │
│   - Based on old policy's samples                                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. Estimate policy gradient and curvature:                                 │
│   - Compute ∇_θ^π L_actor                                                   │
│   - Estimate Fisher Information Matrix F from policy outputs                │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 7. Solve natural gradient direction:                                         │
│   - Use conjugate gradient to solve F·x = ∇_θ^π L_actor                      │
│    F x = g  ⟶  x = F^{-1} g → x                                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 8. Line search along direction x:                                           │
│   - Find step size η satisfying:                                           │
│     KL(π_old || π_new) ≤ δ and improved surrogate loss                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 9. Update actor (policy) parameters using natural gradient:                │
│     θ^π ← θ^π + η · x                                                       │
│   - Safe update within trust region using optimized surrogate loss         │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│10. Update critic (value function) parameters using TD loss:                │
│     L_critic = (Rₜ − V(sₜ; θ^V))²                                           │
│     θ^V ← θ^V - α_V ∇_θ^V L_critic                                          │
└──────────────────────────────────────────────────────────────────────────────┘

# PPO

┌──────────────────────────────────────────────────────────────────────────────┐
│ Example: PPO Agent in a CartPole Environment                               │
│ (Actor-Critic Structure: Policy and Value Networks)                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. Initialize networks:                                                     │
│   - Actor network π(a|s; θ^π): stochastic policy                            │
│   - Critic network V(s; θ^V): value function                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. For each iteration:                                                     │
│   - Collect trajectories D = {τ₁, ..., τ_N} using current policy π_θ       │
│     For each transition (s_t, a_t, r_{t+1}, s_{t+1}):                       │
│       - Store log π_θ(a_t | s_t) and value V(s_t)                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. Compute returns and advantages:                                         │
│   - R_t = r_t + γ·r_{t+1} + ... (Monte Carlo or GAE)                       │
│   - A_t = R_t - V(s_t; θ^V)                                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. Compute PPO clipped surrogate objective:                                │
│   r(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)                                     │
│   L_actor = E_t [ min( r(θ)·A_t, clip(r(θ), 1−ε, 1+ε)·A_t ) ]              │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. Compute critic loss:                                                    │
│   L_critic = (R_t − V(s_t; θ^V))²                                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. Compute total loss and update networks:                                 │
│   - L_total = L_actor + c_v·L_critic − c_e·EntropyBonus                    │
│   - θ^π ← θ^π − α_π ∇_θ^π L_total                                          │
│   - θ^V ← θ^V − α_V ∇_θ^V L_critic                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 7. Repeat update for K epochs over same data                               │
│   - Improves stability and sample efficiency                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 8. Repeat for next iteration (sample new batch)                            │
└──────────────────────────────────────────────────────────────────────────────┘


# Dreamer

```math
V_\lambda(s_t) =
(1 - \lambda) \sum_{n=1}^{H-1} \lambda^{n-1} 
\left( \sum_{i=0}^{n-1} \gamma^i \hat{r}_{t+i} + \gamma^n v_\psi(s_{t+n}) \right)
+ \lambda^{H-1} 
\left( \sum_{i=0}^{H-1} \gamma^i \hat{r}_{t+i} + \gamma^H v_\psi(s_{t+H}) \right)
```

Transition model	✅ imagination rollout	❌   
Reward model	✅ rollout 중 보상 예측	❌   
Value network	✅ λ-return 계산	❌   
Action network (policy)	✅ imagination에 사용됨	✅ 실제 행동 선택   
Representation model	❌	✅ 현재 상태 인코딩 (o_t → s_t)

# Dreamer V2



# Dreamer V3
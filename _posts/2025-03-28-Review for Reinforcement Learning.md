---
title: "Review for Reinforcement Learning"
date: 2025-03-28
tags:
    - Reinforcement Learning
categories: 
    - Summary
toc: true
toc_sticky:  true
---

현재까지 강화학습 알고리즘은 매우 다양한 방면으로 발전해 왔다. 하지만 강화학습을 처음 공부하는 사람의 입장에서는 이런 알고리즘이 어떻게 얽혀있는지 감을 잡기 쉽지 않다. 그래서 강화학습과 관련된 논문을 읽을 때 이 알고리즘이 어떠한 방법론을 사용하는 알고리즘인지 스스로 감을 쉽게 잡기 위해서 정리 페이지를 만들었다. 강화학습의 아주 기본적인 내용이나 세세한 수학적인 개념까지 자세하게 다루지는 않겠지만 헷갈리는 개념들 위주로 알아보면 좋을 것 같다.

<br>

# 1. 어디에 중점을 두냐에 대한 구분: Policy based approach / Value based approach

강화학습의 목표는 항상 다음의 기댓값을 최대화하는 정책 $\pi$를 찾는 것이었다.

$$
J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

이러한 목표를 달성하기 위해 우리는 두 가지 종류의 접근 방법을 생각할 수 있다.(물론 나중에 더 발전된 알고리즘에서는 두 방식이 혼합된 듯한 방식을 사용한다.) 첫 번째는 **policy를 직접 모델링하지 않고, 상태 또는 상태-행동 쌍의 가치를 추정해서 행동을 선택**하는 것이고, 두 번째는 **policy를 명시적으로 지정해서 policy를 직접 업데이트**하는 것이다. 전자가 흔히 말하는 value iteration, 후자가 policy iteration이라고 할 수 있다.

## Value based approach - example.Q-learning

state value function V는 다음과 같이 표현되며

$$
V^\pi(s) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s \right]
$$

state-action value Q는 다음과 같이 표현된다.

$$
Q^\pi(s, a) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s, a_0 = a \right]
$$

여기서 V는 현재 상태에서 선택할 수 있는 모든 행동 a와 다음 상태들을 모두 고려해서 현재 상태의 기댓값을 표현하며, Q는 현재 상태의 **어떤 행동 a**를 선택했을 때 그 이후를 고려해서 현재 상태-행동의 기댓값을 표현한다. 따라서 어떤 policy를 따르는 V와 Q의 관계는 다음과 같이 표현할 수 있다.

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ Q^\pi(s, a) \right]
$$

$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \cdot Q^\pi(s, a)
$$

이러한 value와 관련된 값을 직접 업데이트하는 방식을 사용하는 알고리즘에는 대표적으로 SARSA, DQN이 있다. 이러한 value based approach에서는 알고리즘이 직관적이고 안정적이며, action space가 discrete한 경우 매우 안정적으로 동작하지만 continuous action space에서는 동작이 어렵다는 단점이 있다. 또한 **policy $\pi$**에 간접적으로 접근한다는 점 또한 문제가 될 수 있다. 어디까지나 이러한 방법론은 점차 행동을 개선시켜가며 정책을 implicit하게 발전시키기 때문이다.

## Policy based approach - example.REINFORCE

강화학습의 목표 함수를 한 번 더 다시 보자.

$$
J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

이번에는 policy $\pi_\theta(a \mid s)$를 명시적으로 직접 파라미터화 해서 학습을 해 볼 것이다. 그렇다면 value function을 사용하지 않느냐? 한다면 답은 **그럴 수도 있고 아닐 수도 있다.** 사실 최근 발전된 기법들은 **Actor-Critic** 구조라고 해서 대부분 이 value와 policy를 동시에 사용한다. 이 기법들은 뒤에서 다루기로 하고, 여기서는 policy만 명시적으로 발전시키는 경우에 한해서 알아보도록 하자.

명시적으로 나타나 있는 policy를 변화시키려면 policy에 대한 gradient가 필요할 것이라고 짐작할 수 있다. 이것을 한 번 구해볼 것이다. 우선 trajectory의 확률을 생각해 보자. 정책에 따라 어떤 trajectory τ가 나올 확률은 다음과 같이 쓸 수 있다. 여기서 $\pi$는 상태를 받아 행동으로 치환하는 어떠한 분포, $\rho$는 초기 상태가 선택될 확률이라고 생각하면 된다.

$$
P(\tau) = \rho(s_0) \prod_{t=0}^{T} \pi_\theta(a_t | s_t) \cdot p(s_{t+1} | s_t, a_t)
$$

따라서 기대 return을 trajectory에 대한 적분 형태로 쓰면 다음처럼 쓸 수 있다.

$$
J(\theta) = \int_\tau P(\tau) R(\tau) d\tau
$$

기대값의 gradient는 다음과 같이 미분할 수 있다.

$$
\nabla_\theta J(\theta)
= \nabla_\theta \int_\tau P(\tau) R(\tau) d\tau
= \int_\tau \nabla_\theta P(\tau) R(\tau) d\tau
$$

여기서 ∇θP(τ)에 log-derivative trick을 적용하면 다음과 같이 쓸 수 있다. (이는 증명이 되어 있고, 자세한 증명 과정은 생략한다.)

$$
\nabla_\theta P(\tau) = P(\tau) \cdot \nabla_\theta \log P(\tau)
$$

환경의 dynamics는 정확히 알 수 없기 때문에 경로적분과 합쳐서 estimation에 대한 표현으로 바꾸면 다음과 같이 표현할 수 있다.

$$
\nabla_\theta J(\theta)
= \int_\tau P(\tau) \nabla_\theta \log P(\tau) R(\tau) d\tau
= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau) \cdot R(\tau) \right]
$$

환경의 dynamics 𝑝(𝑠′∣𝑠,𝑎)는 𝜃와 무관하기 때문에 미분할 필요가 없다. 따라서 log𝑃(𝜏)에서 정책 부분만 남는다.

$$
\log P(\tau) = \log \left( \rho(s_0) \prod_{t=0}^{T} \pi_\theta(a_t | s_t) \cdot p(s_{t+1} | s_t, a_t) \right)
= \sum_{t=0}^{T} \log \pi_\theta(a_t | s_t) + \text{(정책과 무관한 부분)}
$$

따라서 미분해도 정책 부분만 남는다.

$$
\nabla_\theta \log P(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t)
$$

최종적으로 다음과 같이 정리할 수 있다.

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau \sim \pi_\theta} \left[
    \left( \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \right)
    \cdot R(\tau)
\right]
$$

이제 보상 R을 시간 t마다 나눠서 쓰면:

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau \sim \pi_\theta} \left[
    \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t
\right]
$$

사실 원래대로라면 Q 값을 사용하는 것이 맞다. 

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a) \right]
$$

하지만 policy based approach에서는 policy를 **확률분포에 근사하여 사용하기 때문에 Q function의 값이나 Value function을 정확하게 알 수 없다는 치명적인 단점이 존재한다.** 즉 업데이트 시 Q값을 추산할 수 없기 때문에 episode가 끝난 후인 보상 G로 대체하여 업데이트한다. 이렇게 episode가 끝날 때마다 업데이트하는 방식을 **Monte-Carlo**방식이라고 하며, Policy based approach를 취하며 MC 방식으로 파라미터를 업데이트하는 이런 방법론을 취한 알고리즘이 바로 **REINFORCE** 알고리즘이다. 여기서 우리가 추론할 수 있는 점은 여기서 파생해서 **MC 방식으로 업데이트하지 않을 수도, 단순히 policy based approach만을 취하지 않을 수도** 있다는 것이다.

REINFORCE의 흐름도는 다음과 같이 간단하게 정리할 수 있다.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. Initialize policy network π(a|s; θ)                                       │
│    - Outputs probability distribution over actions for a given state s       │
│    - Example output: logits → softmax → π(a|s)                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. For each episode:                                                         │
│    - Reset environment → receive initial state s₀                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. For each time step t in the episode:                                     │
│   1) Sample action from policy:                                              │
│        aₜ ~ π(a|sₜ; θ)   ← stochastic sampling                               │
│   2) Execute action aₜ in the environment                                    │
│        → Observe reward rₜ₊₁ and next state sₜ₊₁                             │
│   3) Store (sₜ, aₜ, rₜ₊₁) into trajectory buffer                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. After episode ends (trajectory complete):                                 │
│   For each time step t = 0 to T:                                             │
│   1) Compute return Gₜ from t onward:                                        │
│        Gₜ = ∑_{k=0}^{T−t−1} γ^k · r_{t+k+1}                                   │
│        → Total discounted future reward from step t                          │
│   2) Compute policy loss:                                                    │
│        Lₜ = -log π(aₜ|sₜ; θ) · Gₜ                                           │
│        → Encourages high-return actions by increasing their log-prob         │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. Backpropagate total loss and update policy network:                      │
│     Total Loss = ∑_t Lₜ                                                      │
│     θ ← θ - α ∇θ Total Loss   ← Gradient ascent step                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. Repeat for next episode                                                  │
│     → Continue collecting trajectories and updating policy                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 특이점

여기서부터는 위의 이분법적인 구조에서 벗어나 둘을 융합하기도, 또는 새로운 개념을 도입하여 학습을 더 robust하게 일어나게 하는 여러 방법에 대해 알아볼 것이다. 그 뒤의 PPO, RLHF 등의 현재 많이 사용되는 알고리즘은 밑의 알고리즘을 발전시킨 것이다.

### target network의 추가 - example.Dueling DQN

dueling DQN은 DQN에서 두 가지가 바뀐 DQN이다. 즉 바뀐 부분을 제외한 나머지 부분은 기존 DQN과 완전히 동일하다. 여기서는 vanila DQN에 대해서 이해하고 있다는 가정 하에 설명을 진행한다.

핵심 아이디어는 Q(s, a) 를 상태의 value와 행동의 advantage로 나눠서 계산하는 것이다. (여기서의 advantage는 뒤에서 나올 trpo의 advantage와는 개념도, 역할도 다름에 유의)

$$
Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)
$$

여기서 V는 상태의 절대적인 가치를 나타내며, A는 **특정 행동이 다른 행동들에 비해 얼마나 좋은지를 나타낸다.** 여기서의 advantage의 역할은 Q value를 계산하는 데 있어서 안정성을 부여하기 위한 장치이다. 이를 계산할 때에는 둘 다 동일하게 CNN/MLP layer를 공유하며 이후 두 갈래로 나뉘어서 각각 V를 반환하는 value head, advantage를 반환하는 advantage head를 통해 값이 출력된다. 이를 흐름도로 나타내면 다음과 같다.

```

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
```

기존 DQN과 다른 차이점은 target network를 따로 두어서 일정한 주기로 main network로 덮어씌우는 거이다. 이는 기존 DQN에서 매 step마다 network을 freeze해서 업데이트의 기준으로 두었던 것과 비교해서 안정성이 조금 더 확보되는 장점이 있다.

### Deterministic Policy Gradient, actor-critic structure - example.DDPG

맨 처음 살펴보았던 REINFORCE에서는 policy가 stochastic한 확률분포로써 이루어졌다. 반면 여기서는 policy에 따라 action이 고정되어 있는 **deterministic policy**를 사용하는 모델에 대해 알아볼 것이다. 사실 이 모델은 DQN을 연속 제어에서 사용하고자 하는 아이디어에서 출발한 모델이다.

DDPG는 이전에 Dueling DQN에서 보았던 target network를 채용함과 동시에, actor critic 구조를 사용한 모델이다. 네트워크는 두 개의 메인 네트워크인 actor network(상태 s를 받아 행동 a를 반환), critic network(상태 s, 행동 a를 받아 보상 또는 가치를 반환)으로 이루어져 있으며, 이들의 target 역할을 하는 target actor, target critic network까지 합해 총 4개의 네트워크로 구성된다.

DDPG가 행동을 실행할 때에는 policy마다 action은 deterministic 하지만 이 때 noise를 첨가해서 어느 정도 variance를 보장한다. replay buffer의 아이디어는 DQN에서 그대로 가져왔으니 생략하고, 그 이후 과정을 알아보자. 

replay buffer에는 $(s_t, a_t, r_{t+1}, s_{t+1})$이 저장되어 있다. 그렇다면 target을 계산하는 데 있어서 $\text{td target y} = r + \gamma * Q^{'}(s', a')$이므로 s', r은 있으므로 상태 s를 **target actor network에 집어넣어** 다음 행동 a'을 반환(예측)하도록 한 뒤, 이 (s', a') 쌍을 **target critic network**에 집어넣어 Q'값을 알 수 있다. 

그리고 (s, a) 쌍을 이용해서 **main actor network, main critic network**에 통과시켜 예측 Q값을 알 수 있다. 이 두 Q, Q'값의 차이를 이용해 loss를 구하는 것이다. 

$$
\begin{aligned}
&\text{Target:} \quad y = r + \gamma \cdot Q'(s', \mu'(s')) \\\\
&\text{Loss:} \quad L = \left( Q(s, a; \theta^Q) - y \right)^2
\end{aligned}
$$
이 loss 를 이용해 critic network를 업데이트하고,

$$
\theta^Q \leftarrow \theta^Q - \alpha_Q \nabla_{\theta^Q} L
$$

$$
\nabla_{\theta^Q} L = \mathbb{E}_{(s,a)} \left[
  2 \cdot \left( Q(s,a;\theta^Q) - y \right) \cdot \nabla_{\theta^Q} Q(s,a;\theta^Q)
\right]
$$

다음과 같이 계산할 수 있다. 하지만 actor loss를 계산하는 것은 조금 더 복잡한데, 일단 main actor network의 파라미터를 업데이트하는 과정은 다음과 같은 식으로 나타낼 수 있다.

$$
\theta^\mu \leftarrow \theta^\mu + \alpha_\mu \nabla_{\theta^\mu} J
$$

우리가 알고 싶은 것은:

$$
\nabla_{\theta^\mu} J = \nabla_{\theta^\mu} Q(s, \mu(s))
$$

그런데 이 Q는 a에 대해 정의된 함수이므로, chain rule을 써야 한다. DDPG에서 actor의 목적은 

$$
J(\theta^\mu) = \mathbb{E}_{s \sim D} \left[ Q(s, \mu(s; \theta^\mu)) \right]
$$

즉, actor는 자신의 정책이 만들어내는 행동이 높은 Q값을 가지게 하는 것이 목적이다. 이제 목적함수 J를 파라미터 𝜃𝜇에 대해 미분한다.

$$
\nabla_{\theta^\mu} J(\theta^\mu)
= \nabla_{\theta^\mu} \mathbb{E}_{s \sim D} \left[ Q(s, \mu(s; \theta^\mu)) \right]
$$

$$
= \mathbb{E}_{s \sim D} \left[ \nabla_{\theta^\mu} Q(s, \mu(s)) \right]
$$

Q(s, a)는 a에 대한 함수이고, a는 $\mu(s; \theta^{\mu})$로 정의되는 함수이므로 합성함수이다. 이제 chain rule을 사용하면

$$
\nabla_{\theta^\mu} Q(s, \mu(s))
= \nabla_a Q(s,a) \Big|_{a = \mu(s)} \cdot \nabla_{\theta^\mu} \mu(s)
$$

따라서 전체 gradient는:

$$
\nabla_{\theta^\mu} J(\theta^\mu)
= \mathbb{E}_{s \sim D} \left[
  \nabla_a Q(s,a) \Big|_{a = \mu(s)} \cdot \nabla_{\theta^\mu} \mu(s)
\right]
$$

이러한 과정이 끝나면 일정 주기마다 main network를 target network로 복사하는 soft target update를 통해 학습 안정성을 개선할 수 있다.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. 네트워크 초기화                                                           │
│   - Actor network μ(s; θ^μ): Deterministic Policy                            │
│   - Critic network Q(s,a; θ^Q): Action-value function                        │
│   - Target networks: μ′, Q′ (soft target updates용)                           │
│     θ^{μ'} ← θ^μ,   θ^{Q'} ← θ^Q                                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. Replay buffer D 초기화                                                   │
│   - 경험을 저장하고 샘플링하기 위한 메모리                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. For each episode:                                                        │
│   - 환경 초기화 → 초기 상태 s₀                                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. For each time step t in episode:                                         │
│   1) 행동 선택:                                                             │
│       aₜ = μ(sₜ; θ^μ) + noise                                               │
│       → Exploration 위해 OU noise 등 사용 가능                             │
│   2) 행동 실행 → 보상 rₜ₊₁, 다음 상태 sₜ₊₁ 관측                             │
│   3) (sₜ, aₜ, rₜ₊₁, sₜ₊₁) → replay buffer D 저장                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. Mini-batch 샘플링                                                        │
│   - D에서 무작위 transition 샘플 K개 추출                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. For each sample (s, a, r, s'):                                           │
│   1) Target action 계산: a' = μ′(s')                                         │
│   2) Target Q 계산:                                                         │
│        y = r + γ · Q′(s', a')             ← ✅ TD Target                     │
│   3) Critic loss 계산:                                                      │
│        L = (Q(s,a; θ^Q) - y)²             ← TD error                         │
│   4) Critic 네트워크 업데이트:                                              │
│        θ^Q ← θ^Q - α_Q ∇_θ^Q L            ← ✅ Value Learning                │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 7. Actor 업데이트 (Policy Improvement):                                     │
│   1) Critic을 통해 gradient 추출:                                           │
│        ∇_θ^μ J ≈ ∇_a Q(s,a) · ∇_θ^μ μ(s)                                     │
│   2) Actor 업데이트:                                                        │
│        θ^μ ← θ^μ + α_μ ∇_θ^μ J                                               │
│      → deterministic policy를 Q-value가 높은 방향으로 개선                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 8. Soft target network 업데이트 (Polyak averaging)                          │
│     θ^{μ'} ← τ θ^μ + (1 - τ) θ^{μ'}                                          │
│     θ^{Q'} ← τ θ^Q + (1 - τ) θ^{Q'}                                          │
│     → 천천히 따라가는 target이 학습 안정성 향상                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

| 구성 요소 | 설명 |
|------------|------|
| **Actor** | 상태 → 행동 (Deterministic policy μ(s)) |
| **Critic** | Q(s,a) 추정. Actor의 성능 피드백 제공 |
| **Replay buffer** | Off-policy 학습 가능하게 함 |
| **Target networks** | 안정적 TD target 생성에 필수 |
| **TD target** | `y = r + γ·Q'(s', μ'(s'))` (bootstrapping) |
| **Policy Gradient** | `∇_θ^μ J ≈ ∇_a Q(s,a) · ∇_θ^μ μ(s)` |


### Advanced actor-critic

Advanced actor-critic(A2C)라는 이름은 상대적으로 나중에 정의되었다. 많은 강화학습 논문에서 A2C를 차용하고 있지만 이는 이전까지 설명했던 여러 기법들의 종합적인 방법론에 가깝다. 그 중 핵심은 **actor-critic 구조, Advantage 함수 사용 그리고 batch 기반 동기화**이다. 보통 연구자들이 인용을 할 때에는 Mnih et al.의 A3C(Asynchronous Advantage Actor-Critic) 의 동기화된 버전 (Synchronous) 으로 많이 이야기한다.

밑의 흐름도에서, adavantage function의 개념이 나온다. 이는 어떤 상태 𝑠에서 어떤 행동 𝑎를 취하는 것이 얼마나 더 (or 덜) 좋은지를 측정하는 함수이다. 즉 **그 행동이 평균보다 얼마나 나았는가?**를 의미하기도 한다.

이전의 policy gradient는 다음과 같이 정의되었다.

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s,a} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a) \right]
$$

하지만 Q값은 실제로는 높은 분산값을 가질 수 있고, MC 방식으로 추정할 경우 특히 더 불안정해진다는 문제가 있었다.(REINFORCE 참고) Advantage 함수는 그 자체가 variance를 줄이는 baseline 역할을 한다.

$$
A(s,a) = Q(s,a) - b(s)
$$

보통 b(s) = V(s)가 가장 일반적인 baseline으로 쓰인다.

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi(a|s) \cdot A(s,a) \right]
$$

이 형태는 A2C, PPO, TRPO 등 거의 모든 modern PG 알고리즘에서 공통적으로 사용된다. 그리고 A(s, a)는 아래와 같이 근사할 수 있다.

$$
A(s,a) ≈ G_t - V(s)            ← A2C \\
A(s,a) ≈ Q(s,a) - V(s)         ← General case \\
A(s,a) ≈ GAE(λ)                ← PPO, TRPO
$$


```
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
```

<br>

# 2. 가치를 업데이트 하는 방식에 대한 구분: MC / TD

강화학습에서 에이전트는 주어진 환경과 상호작용하면서 어떤 상태에서 어떤 행동을 선택했을 때 얼마나 좋은 결과를 얻을 수 있을지를 학습한다. 이를 위해 상태 또는 상태-행동 쌍의 가치(value) 를 추정하는데, 그 대표적인 방법이 Monte Carlo 방식과 Temporal Difference 방식이다. 이 두 방법은 서로 다른 학습철학과 수렴 성질을 가지며, 대부분의 강화학습 알고리즘은 이 둘 중 하나 또는 중간 형태를 따른다.

## Monte carlo

Monte Carlo 방식은 한 에피소드가 끝날 때까지 기다린 뒤, 실제로 관측한 보상의 누적 합을 계산하여 학습에 사용한다. 예를 들어, 상태 $$s_t$$에서 시작된 에피소드의 총 리턴 $$G_t$$는 다음과 같이 정의된다:

$$
G_t = ∑_{k=0}^{T−t−1} γ^k · r_{t+k+1}
$$

이 리턴을 기반으로 가치 함수 $$V(s_t)$$를 업데이트하는 방식은 다음과 같다:

$$
V(s_t) ← V(s_t) + α · (G_t − V(s_t))
$$

이 방식은 리턴을 온전히 관측하기 때문에 bias는 없지만, 보상의 편차가 클 경우 variance가 크고, 전체 에피소드가 끝나야만 업데이트가 가능하다는 점에서 느리고 불안정한 학습이 될 수 있다. 또한 종결되지 않는 환경에서는 사용이 제한된다.

## Temporal difference

TD 방식은 리턴을 끝까지 기다리지 않고, 다음 상태의 예측된 가치를 사용하여 바로 학습한다. 즉, 미래를 직접 관측하는 대신 현재 상태에서 예측된 가치를 이용하는 부트스트랩 방식이다. TD 방식에서의 타깃 값 $$y_t$$는 다음과 같이 계산된다:

$$
y_t = r_t + γ · V(s_{t+1})
$$

이를 바탕으로 현재 상태의 가치 업데이트는 다음과 같이 진행된다. 이 수식은 어디선가 매우 많이 본 형태이다. 왜냐하면 거의 모든 value based approach에서 TD loss를 적용하기 때문이다. 어찌 보면 당연한 것.

$$
V(s_t) ← V(s_t) + α · (y_t − V(s_t))
$$

이 방식은 빠르고 효율적이며, 오프폴리시 학습에도 적합하다. 하지만 타깃이 예측된 값이기 때문에, 잘못된 예측은 학습 전반에 영향을 줄 수 있어 bias가 존재한다. 

## 중간지점: n-step TD & GAE

MC와 TD 사이의 절충안으로는 n-step TD와 GAE (Generalized Advantage Estimation)이 있다. n-step TD는 일부 리턴을 관측한 뒤 이후는 추정값으로 대체하는 방식이다:

$$
y_t^{(n)} = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^n V(s_{t+n})
$$

이 방식은 variance와 bias 사이의 trade-off를 조절할 수 있으며, GAE는 이 아이디어를 Advantage 추정에 적용하여 PPO, TRPO 등의 최신 알고리즘에 사용된다.

<br>


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

이러한 value와 관련된 값을 직접 업데이트하는 방식을 사용하는 알고리즘에는 대표적으로 SARSA, DQN이 있다. 이러한 valuen based approach에서는 알고리즘이 직관적이고 안정적이며, action space가 discrete한 경우 매우 안정적으로 동작하지만 continuous action space에서는 동작이 어렵다는 단점이 있다. 또한 **policy $\pi$**에 간접적으로 접근한다는 점 또한 문제가 될 수 있다. 어디까지나 이러한 방법론은 점차 행동을 개선시켜가며 정책을 implicit하게 발전시키기 때문이다.

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

### target network의 추가, advantage 개념 - example.Dueling DQN


### Deterministic Policy Gradient - example.DDPG


### TRPO


<br>

# 2. 가치를 업데이트 하는 방식에 대한 구분: MC / TD





<br>


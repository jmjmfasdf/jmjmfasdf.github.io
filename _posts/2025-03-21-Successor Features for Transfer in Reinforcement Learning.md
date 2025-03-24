---
title: "Successor Features for Transfer in Reinforcement Learning"
date: 2025-03-21
tags:
    - Reinforcement Learning
    - Representation
categories: 
    - Paper Review
toc: true
toc_sticky:  true
---

이전의 'The hippocampus as a predictive map' 논문에서도 SR에 관련된 내용을 다루었고, 이 논문에서도 그럴 것이다. 하지만 두 논문이 Successor feature/representation을 다루는 방식에는 약간의 차이가 있기에, 이 부분을 먼저 짚고 넘어가려고 한다. Stachenfeld et al. (2017, 이전 논문)에서 상태 가치 함수 (State Value Function)는 다음과 같이 표현될 수 있다.

$$
V(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t) \mid s_0 = s \right]
$$

그리고 Barreto et al. (2017, 본 논문)에서의 행동-가치 함수 Q는 다음과 같이 표현할 수 있다. 

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t, s_{t+1}) \mid s_0 = s, a_0 = a \right]
$$

또한 Barreto et al.에서 r은 다음과 같이 표현할 수 있다고 가정하겠다.

$$
r(s, a, s') = \phi(s, a, s')^\top w
$$

그리고 Stachenfeld의 논문에서 정의한 **Successor Representation**의 정의, 그리고 상태 가치 함수 V는 다음과 같이 표현될 수 있다.

$$
M = \sum_{t=0}^{\infty} \gamma^t T^t = (I - \gamma T)^{-1}
$$

$$
V = M R
$$

또한 Barreto의 논문에서의 **Successor Feature**의 정의는 다음과 같이 표현할 수 있다.

$$
\psi^\pi(s,a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t \phi(s_t, a_t, s_{t+1}) \mid s_0 = s, a_0 = a \right]
$$

$$
\begin{aligned}
Q^\pi(s, a) 
&= \mathbb{E}_\pi \left[ r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots \mid S_t = s, A_t = a \right] \\
&= \mathbb{E}_\pi \left[ \phi_{t+1}^\top w + \gamma \phi_{t+2}^\top w + \gamma^2 \phi_{t+3}^\top w + \cdots \mid S_t = s, A_t = a \right] \\
&= \left( \mathbb{E}_\pi \left[ \sum_{i=t}^{\infty} \gamma^{i - t} \phi_{i+1} \mid S_t = s, A_t = a \right] \right)^\top w \\
&= \psi^\pi(s, a)^\top w
\end{aligned}
$$

ϕ(s, a, s')는 feature vector로 보통은 어떤 임의의 함수 ϕ = S × A × S → ℝⁿ 형태다. 그런데 지금 상황에서는 ϕ가 one-hot 벡터라고 가정해 보자. 이 말은 특정 (s', a')가 주어졌을 때 ϕ(s, a, s')는 벡터의 특정 위치에서만 1이고 나머지는 0이라는 뜻이다. 즉, 각 (s, a, s′) 쌍마다 feature vector ϕ(s, a, s′)는 다음과 같은 벡터다:

$$
\phi(s, a, s') = e_{(s', a')} \quad \text{(some fixed index depending on } s', a')
$$

여기서 $$e_{(s', a')}$$는 feature vector의 특정 위치가 1이고 나머지는 0인 standard basis vector다. 이 의미는 곧, ϕ가 어떤 **미래 state-action pair (s′, a′)**가 발생했는지를 인디케이터 벡터로 나타낸다는 뜻이다. Barreto et al.의 SF 정의를 다시 보자:

$$
\psi^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t \phi(s_t, a_t, s_{t+1}) \mid s_0 = s, a_0 = a \right]
$$

이 정의는, 시작점 (s, a)에서 출발해 policy π를 따라 행동할 때, 미래에 발생하는 (s′, a′) pair들의 ϕ값을 누적한 것이다. **그런데 만약 ϕ가 one-hot이라면?** 각 시점 t에서 ϕ(s_t, a_t, s_{t+1})는 다음과 같다:

$$
\phi(s_t, a_t, s_{t+1}) = e_{(s_{t+1}, a_{t+1})}
$$

여기서 a_{t+1}는 policy π에 따라 확률적으로 정해지므로, expectation 안에서는 e_{(s', a')}의 기대값으로 남는다. 중요한 것은, 이 기대값은 결국, 현재 (s, a)에서 시작했을 때, (s', a')가 미래에 몇 번 등장하는지의 할인된 기대값이다. 따라서, ψ^π(s, a)의 각 원소는 다음을 나타낸다:

$$
\psi^\pi(s, a)[(s', a')] = \mathbb{E}^\pi \left[ \sum_{t=0}^{\infty} \gamma^t \mathbb{I}[s_t = s', a_t = a'] \mid s_0 = s, a_0 = a \right]
$$

이는 정확히 (s, a)에서 시작했을 때 미래에 (s′, a′)를 방문할 할인된 기대 횟수를 뜻한다. 이 정의는 우리가 SR에서 사용한 정의와 구조적으로 동일하다. 그리고 Stachenfeld et al.에서의 Successor Representation은 다음과 같다:

$$
M(s, s') = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t \mathbb{I}[s_t = s'] \mid s_0 = s \right]
$$

이 정의는 미래에 상태 s′에 몇 번 방문하는지를 나타낸다. 여기서는 action은 명시적으로 고려되지 않지만, 만약 (s, a)-pair를 상태로 간주하면 구조는 완전히 같다. 이러한 논리는 Barreto et al. 논문에서도 명시적으로 언급되며, SR이 SF의 특수한 경우임을 보여준다. SF는 SR의 구조를 일반화해 feature space로 확장하며, 이로써 transfer learning에 더 적합한 구조를 갖추게 된다.

<br>

# 1. Introduction

이 논문은 강화학습(Reinforcement Learning, RL)의 전이 학습(transfer learning)에 초점을 맞추며, 특히 동일한 환경 내에서 보상 함수만 달라지는 다양한 하위 과제(subtask) 간의 전이를 다룬다. 논문에서 제안하는 방법은 다음 두 가지 개념적 기반 위에 세워져 있다:

Successor feature는 기존 Dayan의 Successor Representation(SR)을 일반화한 개념으로, 상태를 미래의 상태 방문 확률이 아닌 feature space에서의 예측으로 표현한다. 이는 연속적인 상태 공간(continuous state space)에도 적용 가능하며, 환경의 dynamics와 보상을 분리해서 표현할 수 있다는 점에서 전이 학습에 유리하다. 즉, 환경이 동일하고 보상 구조만 달라질 때, dynamics에 해당하는 SF만 재활용하면 새로운 가치 함수(Q)를 빠르게 계산할 수 있다. 또한 기존 Bellman 이론은 단일 정책에 대해 정의되었으나, 이 논문에서는 다수의 기존 정책들로부터 새로운 과제에서의 성능 보장을 제공하는 일반화된 이론을 제시한다. 이를 통해 새로운 과제에 대해 학습을 수행하기 전부터 어느 정도 성능을 예측할 수 있으며, 이를 기반으로 **재사용 가능한 스킬 라이브러리(skill library)**를 구축할 수 있다.

논문이 다루는 동일한 환경(즉, 동일한 transition dynamics)을 공유하면서 보상 구조만 다른 과제들 간의 전이 시나리오는 다음과 같다. 에를 들어서 운전 과제를 운전대 조작 → 우회전 → 목적지 도달의 계층적 하위 과제로 나눌 수 있다면, 이들 간의 지식 공유가 가능해야 한다고 말하고 있다. 이러한 전이 과정은 별도의 모듈이 아닌 강화학습 내부 구조에 자연스럽게 통합되어야 하며, 과제 간 정보를 유연하게 공유할 수 있어야 한다고 주장한다.

<br>

# 2. Background and problem formulation

MDP는 다음과 같은 5-튜플로 정의된다:

- $\mathcal{S}$: 상태(state)의 집합  
- $\mathcal{A}$: 행동(action)의 집합  
- $p(\cdot \mid s, a)$: 상태 $s$에서 행동 $a$를 취했을 때 다음 상태로의 확률 분포 (transition dynamics)  
- $R(s, a, s')$: 전이 $s \xrightarrow{a} s'$ 에서 얻는 보상 (보통 기대값 $r(s,a,s')$를 사용)  
- $\gamma \in [0,1)$: 미래 보상에 대한 할인 계수 (discount factor)  

또한 에이전트의 목표는 **정책(policy) $\pi$**를 찾아서 기대 누적 보상 (return)

$$
G_t = \sum_{i=0}^{\infty} \gamma^i R_{t+i+1}
$$

을 최대화하는 것이다. 이때 사용하는 핵심 개념은 **가치 함수(value function)**이며, 특히 **행동-가치 함수(action-value function)**는 다음과 같이 정의된다:

$$
Q^\pi(s,a) = \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right]
$$

$\pi$에 대한 $Q^\pi$가 주어지면, 이를 이용해 탐욕적(greedy) 정책 $\pi'$를 만들 수 있다:

$$
\pi'(s) \in \arg\max_a Q^\pi(s, a)
$$

이때 $\pi'$는 $\pi$보다 성능이 같거나 더 나은 정책임이 보장된다. 이와 같이 **정책 평가(policy evaluation)**와 **정책 개선(policy improvement)**을 반복함으로써 최적 정책 $\pi^*$에 수렴할 수 있다. 또한 이 논문에서 전이 transition은 다음과 같이 정의된다:

**작업 집합(task set) $\mathcal{T}$와 그 부분집합 $\mathcal{T}' \subset \mathcal{T}$가 있을 때, $\mathcal{T}$에서 학습한 후 어떤 과제 $t \in \mathcal{T}$에 대해 $\mathcal{T}'$에서만 학습했을 때보다 항상 더 나은(혹은 같은) 성능을 보인다면, 전이가 이루어진 것이다.** 논문에서는 하나의 과제(task)를 동일한 MDP 하에서의 보상 함수 $R(s,a,s')$의 구체적인 정의로 본다.

<br>

# 3. Successor features

Barreto는 보상 함수가 다음과 같이 표현될 수 있다고 가정한다:

$$
r(s, a, s') = \phi(s, a, s')^\top w
$$

여기서 $\phi(s, a, s')$는 $(s, a, s')$ transition의 feature vector이며, $w$는 각 feature에 대응하는 가중치 벡터이다. 이 표현은 일반성을 잃지 않으며, $\phi_i(s, a, s') = r(s, a, s')$인 단일 feature로도 임의의 보상 함수를 구성할 수 있다.

정의된 $r(s, a, s')$를 이용해 Q-함수를 다음과 같이 전개할 수 있다:

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ r_{t+1} + \gamma r_{t+2} + \cdots \mid S_t = s, A_t = a \right]
$$

보상을 $\phi^\top w$로 표현하면,

$$
\begin{aligned}
Q^\pi(s, a) &= \mathbb{E}_\pi \left[ \phi_{t+1}^\top w + \gamma \phi_{t+2}^\top w + \cdots \mid S_t = s, A_t = a \right] \\
&= \mathbb{E}_\pi \left[ \sum_{i=t}^\infty \gamma^{i-t} \phi_{i+1} \mid S_t = s, A_t = a \right]^\top w \\
&= \psi^\pi(s, a)^\top w
\end{aligned}
$$

여기서 $\psi^\pi(s, a)$는 **Successor Feature (SF)**이며, 정책 $\pi$ 하에서 상태-행동 쌍 $(s, a)$로부터 시작해 미래에 얼마나 많이 각 피처가 발생할지를 할인된 기대값으로 나타낸다. 그리고 Successor Feature는 아래와 같은 Bellman equation을 만족한다:

$$
\psi^\pi(s, a) = \phi(s, a, s') + \gamma \mathbb{E}_\pi \left[ \psi^\pi(s', \pi(s')) \right]
$$

이 식은 SR이 미래 상태 방문 횟수를 기대값으로 나타내듯이, SF는 미래 피처 발생을 기대값으로 나타낸다는 점에서 동일한 구조를 가진다. 특히 $\phi(s, a, s')$가 $(s', a')$에 대해 one-hot 인코딩이라면, 이 피처는 단순히 미래 특정 전이의 존재를 인디케이터로 나타낸다. 이때 SF는 다음과 같이 해석할 수 있다:

$$
\psi^\pi(s,a) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t \mathbb{I}[(s_t, a_t, s_{t+1}) = (s', a', s'')] \mid s_0 = s, a_0 = a \right]
$$

이는 바로 SR의 일반화로 볼 수 있으며, SR은 SF의 특수한 형태로 해석된다. 이런 방식으로 SF는 기존 SR 구조를 피처 공간으로 확장한 것으로, 보상 함수의 변경에도 빠르게 적응할 수 있는 유연한 표현을 제공한다. 

<br>

# 4. Transfer via successor features

먼저, 논문은 다음과 같은 가정 하에서 전이를 정의한다: 환경의 상태 공간 $\mathcal{S}$, 행동 공간 $\mathcal{A}$, 전이 함수 $p$, 할인 인자 $\gamma$는 고정되어 있고, 오직 보상 함수만이 달라진다. 보상 함수는 다음과 같이 피처 표현 $\phi(s, a, s')$와 가중치 벡터 $w$의 내적 형태로 표현된다:

$$
r(s,a,s') = \phi(s,a,s')^\top w
$$

따라서 $\phi$가 고정되어 있다면, **$w$를 바꾸는 것으로 서로 다른 MDP(과제, task)를 정의할 수 있다.** 이렇게 정의된 모든 MDP들의 집합을 다음과 같이 표현한다:

$$
\mathcal{M}_\phi(\mathcal{S}, \mathcal{A}, p, \gamma) \equiv \left\{ M(\mathcal{S}, \mathcal{A}, p, r, \gamma) \mid r(s,a,s') = \phi(s,a,s')^\top w \right\}
$$

이처럼 보상 가중치 $w$에 따라 달라지는 MDP를 "task"라고 간주하며, 이 집합 내에서의 전이 문제를 해결하는 것이 논문의 목표다.

실제 예시로는, 동물이 "배고픈 상태"에서는 음식에 보상을 주고, "목마른 상태"에서는 물에 보상을 주는 상황이 있다. 여기서 $w$는 해당 순간의 "선호(taste)"를 나타내는 벡터로 해석될 수 있다. 다른 예시로는, 상품의 생산 프로세스는 동일하지만, 시장 가격(w)이 변화하는 경우가 있다. 이런 경우 과거 경험으로부터 학습한 정책을 바탕으로 새로운 환경에서도 빠르게 적응할 수 있어야 한다.

본 논문에서는 이전 task 집합 $\mathcal{M} = {M_1, M_2, \dots, M_n}$에서 학습한 경험을 바탕으로, 새로운 과제 $M_{n+1}$에서 어떤 초기 정책 $\pi$를 생성할 수 있어야 하며, 이 정책은 이전 task의 일부 $M' \subset \mathcal{M}$에서만 학습했을 때 얻는 정책 $\pi'$보다 항상 같거나 더 높은 성능($Q^\pi(s, a) \geq Q^{\pi'}(s, a)$)을 보장해야 한다.

이를 해결하기 위해, 저자들은 먼저 기존 dynamic programming(DP)의 policy improvement 개념을 일반화하고, 이 일반화된 형태를 SF를 통해 효율적으로 구현할 수 있음을 보인다. 

## 4.1. Generalized policy improvement

우선, 강화학습에서 중심이 되는 이론 중 하나는 Bellman의 정책 개선 정리(Policy Improvement Theorem) 이다. 이 정리는 어떤 정책 $\pi$의 가치 함수 $Q^\pi$에 대해 greedy하게 행동을 선택하면 기존 정책보다 성능이 떨어지지 않는 새로운 정책을 얻을 수 있다는 것을 보장한다. 이러한 정리는 동적 프로그래밍(DP)의 근간이자 많은 RL 알고리즘이 따르는 원칙이다.

저자는 이러한 정리를 여러 개의 정책 $\pi_1, \pi_2, ..., \pi_n$이 존재할 때로 확장한다. 이 확장은 다음과 같은 상황을 다룬다: 여러 개의 정책들이 있고, 각각의 정책에 대한 Q-value 함수(혹은 그 근사값)가 존재한다고 하자. 이때 우리는 각 정책의 가치 함수 중에서 **가장 높은 값을 가지는 행동을 선택해 새로운 정책 $\pi$를 구성**할 수 있으며, 이 새로운 정책은 모든 기존 정책보다 나쁘지 않다는 보장을 갖는다. 이를 Generalized Policy Improvement (GPI) 라고 부른다.

각 정책 $\pi_i$의 action-value 함수 $Q^{\pi_i}(s,a)$는 근사값 $\tilde{Q}^{\pi_i}(s,a)$로 표현되며, 이때 오차는 최대 $\epsilon$으로 제한된다:

$$
\vert Q^{\pi_i}(s,a) - \tilde{Q}^{\pi_i}(s,a) \vert \leq \epsilon, \quad \forall s,a,i
$$

그리고 새로운 정책 $\pi$는 다음과 같이 정의된다:

$$
\pi(s) \in \arg\max_a \max_i \tilde{Q}^{\pi_i}(s,a)
$$

이때 새로운 정책 $\pi$의 진짜 Q-값은 다음의 하한을 만족한다:

$$
Q^{\pi}(s,a) \geq \max_i Q^{\pi_i}(s,a) - \frac{2\epsilon}{1 - \gamma}
$$

이 정리는 여러 정책 중 어느 하나의 Q-값도 완전히 신뢰하지 않지만, 모든 정책의 Q-값이 일정 수준 이하의 오차만을 포함한다면, 가장 높은 Q값을 기준으로 행동을 결정하는 새로운 정책도 충분히 좋은 성능을 낼 수 있다는 점을 보장한다.

이 결과는 특히 두 가지 상황에서 유용하게 적용될 수 있다. 첫째, 여러 정책이 병렬적으로 평가되는 상황에서, GPI는 이들을 조합하여 좋은 행동을 선택할 수 있는 체계적인 방법을 제공한다. 둘째, 환경이 바뀌는 경우(예: 보상 함수가 바뀌는 transfer learning 설정)에도 이전에 학습한 여러 정책의 Q-값들을 활용해 빠르게 좋은 초기 정책을 생성할 수 있다.

또한, 만약 오차가 없고 ($\epsilon = 0$), 기존의 어떤 정책이 모든 상태에서 항상 다른 정책보다 높은 Q-값을 가진다면, GPI는 기존의 정책 개선 정리와 동일하게 작동하며 더 이상 이전 정책들을 보존할 필요가 없어진다. 하지만, 여러 정책이 서로 다른 상태에서 강점을 가질 경우, GPI는 이들을 상호 보완적으로 활용해 더 강력한 정책을 만들어낸다.

## 4.2. Generalized policy improvement with successor features

이 부분에서는 Successor Features(SFs) 를 활용한 Generalized Policy Improvement(GPI) 기법을 통해, 어떻게 효율적인 transfer learning이 가능한지를 설명한다. 핵심 아이디어는 여러 과제(task)에 대한 최적 정책을 학습한 이후, 새로운 과제에 대한 정책을 이들로부터 유도할 수 있다는 것이다.

우선, 각 과제 $M_i \in \mathcal{M}\phi$는 고유한 weight vector $w_i \in \mathbb{R}^d$에 의해 정의된다. 이때, 과제 $M_i$의 최적 정책을 $\pi_i^*$라고 하고, 그에 따른 action-value function을 $Q_i^{pi^*}(s,a)$라고 하자. 새로운 과제 $M{n+1}$이 등장했을 때, 기존에 학습된 $\pi_i^*$들을 $M_{n+1}$에 대해 재사용하는 방식으로 정책을 구성할 수 있다. 단, reward가 $w_{n+1}$로 바뀌었다면, 기존 정책 $\pi_i^*$의 성능을 다시 계산해야 한다.

이때 SF가 유용하게 작동한다. 이미 학습된 $\psi^{\pi_i^*}(s,a)$가 존재하고, 새로운 reward vector $w_{n+1}$만 주어지면, 다음과 같이 즉시 Q-function을 계산할 수 있다:

$$
Q_{n+1}^{\pi_i^*}(s,a) = \psi^{\pi_i^*}(s,a)^\top w_{n+1}
$$

이렇게 계산된 여러 정책들의 Q-function 중 가장 높은 값을 기준으로 행동을 선택하면, GPI를 통해 기존 정책 중 어떤 것보다도 성능이 떨어지지 않는 새로운 정책 $\pi$를 생성할 수 있다. 이 방식은 새로운 task에 대해 RL을 처음부터 다시 수행하는 것보다 훨씬 효율적이다.

이 접근법의 성능은 Theorem 2를 통해 수학적으로 보장된다. 이 정리는 주어진 task $M_i$에 대해, 다른 과제들 $M_j$에서 유도된 최적 정책 $\pi_j^*$를 실행했을 때의 성능 $Q_i^{\pi_j^*}$와 SF 기반 근사값 $\tilde{Q}_i^{\pi_j^*}$ 간 오차 $\epsilon$이 작을 경우, $\pi$가 $M_i$에서 최적 정책 대비 얼마나 성능이 떨어질지를 다음과 같이 상한할 수 있다고 보장한다:

$$
Q_i^{\pi_i^*}(s,a) - Q_i^{\pi}(s,a) \leq \frac{2}{1 - \gamma} \left( \phi_{\text{max}} \min_j \| w_i - w_j \| + \epsilon \right)
$$

이 식에서 중요한 요소는 $\min_j \mid w_i - w_j |$이다. 즉, 현재 과제 $w_i$와 가장 가까운 과제 $w_j$ 간의 거리(distance)가 작을수록 성능 손실이 적다는 것이다. 이는 곧 **유사한 과제를 과거에 본 경험이 있다면, 새로운 과제도 잘 해결할 수 있다는 직관**을 수학적으로 뒷받침해 준다.

실제 구현에서 SF를 메모리에 저장해야 하므로, 어떤 SF를 유지할지 결정하는 기준도 이론적으로 도출된다. 예를 들어, 새 과제의 $w_i$가 기존의 어떤 $w_j$와 충분히 멀다면 새로운 SF를 생성하고, 아니라면 기존 것을 재활용하거나 교체할 수 있다.

<br>

# 5. Experiments



<br>

# 6. Related work



<br>

# 7. Conclusion









<figure class='align-center'>
    <img src = "/images/2025-03-21-Successor Features for Transfer in Reinforcement Learning/figure1.png" alt="">
    <figcaption>figure 1. caption</figcaption>
</figure>

<figure class='align-center'>
    <img src = "/images/2025-03-21-Successor Features for Transfer in Reinforcement Learning/figure2.png" alt="">
    <figcaption>figure 2. caption</figcaption>
</figure>

<figure class='align-center'>
    <img src = "/images/2025-03-21-Successor Features for Transfer in Reinforcement Learning/figure3.png" alt="">
    <figcaption>figure 3. caption</figcaption>
</figure>
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

# Preliminary

이전의 'The hippocampus as a predictive map' 논문에서도 SR에 관련된 내용을 다루었고, 이 논문에서도 그럴 것이다. 하지만 두 논문이 Successor feature/representation을 다루는 방식에는 약간의 차이가 있기에, 이 부분을 먼저 짚고 넘어가려고 한다. Stachenfeld et al. (2017, 이전 논문)에서 상태 가치 함수 (State Value Function)는 다음과 같이 표현될 수 있다.

$$
V(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t) \mid s_0 = s \right]
$$

그리고 Barreto et al. (2017, 본 논문)에서의 행동-가치 함수 Q는 다음과 같이 표현할 수 있다. 

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t, s_{t+1}) \mid s_0 = s, a_0 = a \right]
$$

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
Q^\pi(s,a) = \psi^\pi(s,a)^\top w
$$

ϕ(s, a, s')는 feature vector로 보통은 어떤 임의의 함수 ϕ = S × A × S → ℝⁿ 형태다. 그런데 지금 상황에서는 ϕ가 one-hot 벡터라고 가정해 보자.
만약 ϕ(s, a, s')가 (s, a)에 따라 one-hot 벡터라고 가정해 보자. 이 말은 특정 (s', a')가 주어졌을 때 ϕ(s, a, s')는 벡터의 특정 위치에서만 1이고 나머지는 0이라는 뜻이다. 즉, 각 (s, a, s′) 쌍마다 feature vector ϕ(s, a, s′)는 다음과 같은 벡터다:

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
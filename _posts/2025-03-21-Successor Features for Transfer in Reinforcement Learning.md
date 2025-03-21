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


만약 ϕ(s, a, s')가 (s, a)에 따라 one-hot 벡터라고 가정해 보자. 이 말은 특정 (s', a')가 주어졌을 때 ϕ(s, a, s')는 벡터의 특정 위치에서만 1이고 나머지는 0이라는 뜻이다. 이 경우 ϕ는 미래에 방문할 state (혹은 state-action) pair의 인디케이터 역할을 하게 된다. 이때 ψ^π(s,a)는 미래에 각 (s', a')에 도달할 할인된 기대 방문 횟수로 해석될 수 있으며, 이는 정확히 SR 행렬 M(s, s')의 정의와 동일하다:

$$
M(s, s') = \mathbb{E}^\pi \left[ \sum_{t=0}^{\infty} \gamma^t \mathbb{I}[s_t = s'] \mid s_0 = s \right]
$$

또는 행동을 고려하면 state-action SR은 다음과 같이 일반화될 수 있다. 

$$
\psi^\pi(s, a) = \mathbb{E}^\pi \left[ \sum_{t=0}^{\infty} \gamma^t \mathbb{I}[s_t = s', a_t = a'] \mid s_0 = s, a_0 = a \right]
$$

이는 SR이 상태 방문 횟수를 나타낸다면, SF는 feature 방문 횟수를 나타내는 것으로 볼 수 있다. 그런데 ϕ가 one-hot이면 feature와 state는 1:1 대응되므로 같은 의미가 된다. 즉 피처 함수 ϕ(s,a,s')가 one-hot이면, 각 피처는 특정 상태(또는 상태-행동 쌍)를 그대로 표현하므로, 이 경우 SF는 단순히 SR을 그대로 복제한 구조와 같아진다:

$$
\psi^\pi(s,a) \equiv M(s,s')
$$

이러한 논리는 Barreto et al. 논문에서도 명시적으로 언급되며, SR이 SF의 특수한 경우임을 보여준다. SF는 SR의 구조를 일반화해 feature space로 확장하며, 이로써 transfer learning에 더 적합한 구조를 갖추게 된다.

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
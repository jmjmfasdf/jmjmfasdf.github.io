---
title: "Linear reinforcement learning in planning, grid fields, and cognitive control"
date: 2025-03-25
tags:
    - Reinforcement Learning
categories: 
    - Paper Review
toc: true
toc_sticky:  true
---

이 논문에서는 인간의 유연한 계획 능력과 반대로 비유연적인 습관적 행동을 설명하는 데 있어서, **과거 연산의 재사용(reuse of previous computation)**이라는 개념을 중심으로 한 새로운 모델을 제안한다.

기존에는 model-based 방법론이 유연한 행동을 설명하고, model-free 방법론이 습관을 설명했으나, model-based 계획은 계산적으로 복잡하고, model-free는 유연한 재계획(replanning)을 설명하지 못하는 등 이 두 방식 모두 뇌의 행동을 완전히 설명하는 데에는 한계가 있었다. 

이에 저자들은 control engineering에서 영감을 받아, **미래 사건에 대한 시간적으로 추상화된 지도를 재사용(temporally abstracted map of future events)**하여 유연한 선택을 가능하게 하면서도 특정한, 정량화 가능한 편향(bias)을 유도하는 생물학적으로 그럴듯한 모델을 제안한다.

이 모델은 고전적인 비선형 최적화 대신, **default policy를 기준으로 하는 soft한 선형 근사(linear approximation)**를 사용해 최적화한다. 이 default policy에 약한 편향이 들어감으로써, 이 모델은 유연한 재계획과 인지 통제(cognitive control) 같은 행동신경과학 현상들을 연결지을 수 있게 된다.

# Introduction

전통적인 강화학습 모델은 유연한 행동(예: 재계획, goal 변경)에 대해서는 model-based 접근을, 비유연적인 행동(예: 습관, 중독, Pavlovian 편향)에 대해서는 model-free 접근으로 설명해 왔다. 그러나 이 모델들은 유연성과 비유연성 모두를 동시에 설명하지는 못한다는 한계가 있다.

Model-based 접근은 미래 행동을 계획하기 위해 Bellman 방정식을 반복적으로 풀어야 하며, 이는 계산량이 너무 커서 생물학적으로 비현실적이다. 반면 model-free 접근은 과거에 학습한 행동 선호도를 캐시해 두고 재사용하지만, 이는 정책 변경이나 보상의 변화에 적응하는 데에 한계가 있다. 이에 따라 기존 연구에서는 **successor representation (SR)**을 활용하여 일정 수준의 재계획 가능성을 확보했지만, SR 역시 기존 정책에 종속되어 있어 목표(goal)나 행동 경로가 크게 바뀌는 경우 적절한 재계산이 어렵다는 한계를 가진다.

이러한 한계를 극복하기 위해 저자들은 control engineering 분야에서 제안된 **linearly solvable MDP (LMDP)**에 기반하여, SR과 유사하지만 보다 안정적이고 범용적인 Default Representation (DR) 개념을 도입한다. DR은 default policy 하에서의 미래 상태 방문 기대치를 저장한 행렬로, 이는 정책이 변경되더라도 변하지 않는 안정적인 기저(reusable map) 역할을 하며, 목표 변경에 따른 유연한 계획도 가능하게 한다. 핵심 아이디어는 정확한 최적화를 대신해 default policy로부터의 편차를 비용으로 반영하는 soft한 최적화 방식을 도입함으로써, Bellman 방정식을 선형 형태로 근사하고 생물학적으로 plausibility를 유지한다는 점이다.

이러한 방식은 entorhinal grid cell과 border cell의 역할에 대한 새로운 해석도 가능하게 한다. Grid cell은 과제 공간의 Fourier-like 표현을 제공하는 것으로 해석될 수 있으며, DR은 이러한 반복적인 공간 구조를 기반으로 동작할 수 있다. 반면 환경에 경계가 생겼을 때는 DR이 새로운 기저 함수로 업데이트되며, 이는 border cell의 응답 특성과 대응된다.

또한 Linear RL은 유연한 계획만이 아니라, 왜 특정 상황에서는 비유연하거나 편향된 행동이 발생하는지도 설명한다. 예를 들어, Stroop task나 Pavlovian 편향 같은 현상은 default policy에서 벗어나는 데 드는 비용으로 해석되며, 이는 인지 통제(cognitive control)의 부담(cost of control)을 수치적으로 설명하는 근거가 된다. 따라서 Linear RL은 단순한 모델임에도 불구하고, 행동 신경과학에서 관찰되는 다양한 현상들을 하나의 이론적 틀로 통합적으로 설명할 수 있다.

<br>

# Results

## The model

이 논문에서는 고전적인 마르코프 결정 과정(MDP)에서 발생하는 순차적 의사결정 문제의 복잡성을 해결하기 위한 새로운 접근법으로 linear RL 모델을 제안한다. 일반적인 강화학습 문제에서는 에이전트가 상태 $s_t$에서 행동 $a_t$를 선택하고, 이에 따라 다음 상태로 전이하며 보상 $r_t$를 받는다. 에이전트의 목표는 미래 보상의 총합, 즉 value function $$v^*(s)$$를 최대화하는 것이다.

전통적인 방식에서는 최적 value 함수 $$v^*(s)$$는 Bellman 방정식을 통해 재귀적으로 정의되며, 각 상태에서의 최적 행동은 미래의 모든 선택에 의존하게 된다:

$$
v^*(s_t) = r(s_t) + \max_{a_t} \sum_{s_{t+1}} P(s_{t+1} | s_t, a_t) v^*(s_{t+1})
$$

하지만 이러한 계산은 큰 상태 공간에서는 매우 비효율적이며, 실제 생물학적 시스템이 이를 그대로 구현하기에는 부담이 크다. 이에 대한 한 가지 해결책은 정책 $\pi$가 고정되었다고 가정하고, 그에 따라 미래 상태 방문 기대치를 계산하는 **Successor Representation (SR)**이다. SR을 이용하면 다음과 같이 간단한 선형 방정식으로 표현된다:

$$
v^\pi = S^\pi r
$$

여기서 $S^\pi$는 정책 $\pi$ 하에서 시작 상태로부터 미래에 어떤 상태를 방문할 기대치를 담고 있는 행렬이다. 하지만 SR은 현재 정책 $\pi$ 하에서의 예측만 가능하므로, 목표가 바뀌어 최적 정책 $$\pi^*$$가 달라지면 정확한 의사결정을 할 수 없다.

이에 대해 linear RL은 최적 정책 $$\pi^*$$ 자체를 선형적으로 근사할 수 있는 모델을 제안한다. 핵심은 정확한 최적화를 하지 않고, default policy $\pi_d$로부터의 편차를 비용으로 간주하는 방식이다. 이렇게 하면 max 연산이 제거되며, 아래와 같이 지수 함수 형태의 선형 방정식으로 value function을 표현할 수 있다:

$$
\exp(v^*) = M P \exp(r)
$$

여기서

- $$v^*$$: 각 상태에서의 최적 가치 함수 (보상 - 제어 비용)  
- $r$: terminal state(목표 상태)에서의 보상 벡터  
- $P$: 각 nonterminal 상태에서 terminal 상태로 도달할 확률  
- $M$: Default Representation (DR) 행렬로, default policy $\pi_d$ 하에서 nonterminal 상태 간의 '가까움'을 측정하는 값이다.  

DR은 SR과 비슷하지만, SR이 특정 정책 $\pi$에 종속된 반면 DR은 default policy $\pi_d$ 하에서 계산되므로 다양한 목표 변화에 안정적으로 재사용 가능하다. DR은 특정 목표 $r$가 주어지면 그에 대응하는 최적 policy $$\pi^*$$를 쉽게 계산할 수 있도록 한다. 이는 특히 replanning, reward 변화, goal 전환 등의 상황에서도 유연하게 대응할 수 있게 해 준다.

또한 max 연산을 log-sum-exp로 근사하는 방식(log-average-exp)은 soft한 최적화를 가능하게 하고, 이는 Pavlovian 편향이나 인지 통제 비용 같은 행동 편향을 자연스럽게 설명하는 기반이 된다. DR은 grid cell 기반의 공간 표현, border cell의 응답 방식 등 뇌에서의 공간적/정신적 탐색에도 연관될 수 있는 잠재력을 지닌 표현 구조이다.

결론적으로, linear RL은 강화학습에서의 정책 간 상호의존성 문제를 부드럽게 근사하는 선형 모델로 볼 수 있다.

<br>

## Replanning



<br>

## Grid fields



<br>

## Border cells



<br>

## Planning in environments with stochastic transitions



<br>

## Habits and inflexible behavior



<br>

## Cognitive control



<br>

## Pavlovian-instrumental transfer



<br>

# Discussion




<figure class='align-center'>
    <img src = "/images/2025-03-25-Linear reinforcement learning in planning, grid fields, and cognitive control/figure1.jpg" alt="">
    <figcaption>figure 1. caption</figcaption>
</figure>

<figure class='align-center'>
    <img src = "/images/2025-03-25-Linear reinforcement learning in planning, grid fields, and cognitive control/figure2.jpg" alt="">
    <figcaption>figure 2. caption</figcaption>
</figure>

<figure class='align-center'>
    <img src = "/images/2025-03-25-Linear reinforcement learning in planning, grid fields, and cognitive control/figure3.jpg" alt="">
    <figcaption>figure 3. caption</figcaption>
</figure>

<figure class='align-center'>
    <img src = "/images/2025-03-25-Linear reinforcement learning in planning, grid fields, and cognitive control/figure4.jpg" alt="">
    <figcaption>figure 4. caption</figcaption>
</figure>

<figure class='align-center'>
    <img src = "/images/2025-03-25-Linear reinforcement learning in planning, grid fields, and cognitive control/figure5.jpg" alt="">
    <figcaption>figure 5. caption</figcaption>
</figure>

<figure class='align-center'>
    <img src = "/images/2025-03-25-Linear reinforcement learning in planning, grid fields, and cognitive control/figure6.jpg" alt="">
    <figcaption>figure 6. caption</figcaption>
</figure>

<figure class='align-center'>
    <img src = "/images/2025-03-25-Linear reinforcement learning in planning, grid fields, and cognitive control/figure7.jpg" alt="">
    <figcaption>figure 7. caption</figcaption>
</figure>

<figure class='align-center'>
    <img src = "/images/2025-03-25-Linear reinforcement learning in planning, grid fields, and cognitive control/figure8.jpg" alt="">
    <figcaption>figure 8. caption</figcaption>
</figure>

<figure class='align-center'>
    <img src = "/images/2025-03-25-Linear reinforcement learning in planning, grid fields, and cognitive control/figure9.jpg" alt="">
    <figcaption>figure 9. caption</figcaption>
</figure>

---
title: "The hippocampus as a predictive map"
date: 2025-03-14
tags:
    - Reinforcement Learning
    - Representation
categories: 
    - Paper Review
toc: true
toc_sticky:  true
---

이 논문의 해당 부분에서는 hippocampus(해마)의 기능을 기존의 인지 지도(cognitive map) 가설과 대비되는 예측 지도(predictive map) 가설을 강화학습(RL) 관점에서 설명하고 있다.

전통적으로 **hippocampus의 place cells(장소 세포)**는 공간을 명확하게 코딩하는 역할을 한다고 여겨져 왔다. 즉, place cells는 동물이 특정 위치에 있을 때 활성화되며, 이들이 모이면 공간의 구조를 나타내는 ‘인지 지도(cognitive map)’를 형성한다고 본다. 그러나 최근 연구에 따르면 place cells의 활성 패턴이 단순히 공간 정보만을 반영하는 것이 아니라는 점이 밝혀졌다. 예를 들어, **예측 부호화(predictive coding)**, **보상 민감성(reward sensitivity)**, **정책 의존성(policy dependence)** 등의 특성은 place cells가 단순한 공간 지도로서의 역할을 넘어서 더 복잡한 정보를 담고 있을 가능성을 시사한다.

# Introduction

이 논문에서는 hippocampus의 역할을 강화학습(RL) 관점에서 해석하며, 기존 cognitive map 이론을 확장하는 predictive map 이론을 제안한다. 강화학습에서 미래 보상을 예측하는 것이 중요하며, 이를 위해 전통적으로 두 가지 접근법이 존재한다. 첫 번째는 **모델 기반 RL(Model-Based RL)**,  두 번째는 **모델 프리 RL(Model-Free RL)**이다. 그러나 이 논문에서는 hippocampus가 세 번째 해결책을 제공한다고 주장한다. hippocampus는 각 상태를 단순한 공간 정보로 저장하는 것이 아니라, 그 상태에서 도달할 수 있는 **후속 상태(successor states)들의 가중 평균으로 표현하는 예측 지도(predictive map)**을 학습한다. 즉, hippocampus는 단순한 위치 정보가 아니라 미래 상태를 예측하는 구조를 학습하며, 이를 통해 장기적인 보상 예측이 가능하고, 보상이 변화해도 빠르게 적응할 수 있다. 이는 기존의 Successor Representation (SR) 개념과 직접 연결되며, 현재 상태를 단순한 위치 정보로 저장하는 것이 아니라, 미래 상태의 가중치로 표현하는 방식이다.

Predictive Map 이론이 place cells의 여러 특성을 더 잘 설명할 수 있다고 주장한다. 첫째, place cells의 활성 패턴이 장애물(obstacles), 환경의 지형(environment topology), 이동 방향(direction of travel)과 같은 변수들에 의해 조절되는 이유를 설명할 수 있다. 단순한 공간 정보가 아니라, 미래 상태를 예측하는 방식으로 동작하기 때문이다. 둘째, predictive map 이론은 공간적(spatial) 과제뿐만 아니라 비공간적(non-spatial) 과제에서도 hippocampus의 역할을 설명할 수 있다. 기존의 cognitive map 이론은 공간적 정보만 설명할 수 있지만, predictive map은 강화학습에서 일반적으로 활용될 수 있는 구조이기 때문이다. 셋째, 이 방식은 모델-프리 RL보다 보상 변화에 더 빠르게 적응할 수 있으며, 모델-기반 RL보다 계산량이 적다는 장점이 있다.

hippocampus와 함께 작용하는 **entorhinal cortex(내후각 피질)의 grid cells(격자 세포)**는 공간을 주기적인 패턴으로 표현한다. 이 논문에서는 grid cells가 predictive map을 더 효과적으로 활용하기 위해 중요한 역할을 한다고 주장한다. grid cells는 predictive representation의 저차원 표현(low-dimensional basis set)을 제공하며, hippocampus에서 학습한 predictive map이 grid cells의 역할을 통해 더 정교해지고, 노이즈가 감소하며, 다중 스케일의 구조(multiscale structure)를 추출하여 계층적 계획(hierarchical planning)에 유리해진다. 이러한 Predictive Map 이론은 기존 Cognitive Map 이론을 확장하며, 보상 학습과의 연결을 설명할 수 있는 강력한 개념이다.

<br>

# Results

## Successor Representation(SR)과 Hippocampus의 역할

전통적으로, hippocampus는 장소(place) 자체를 인코딩하는 역할을 한다고 알려져 있다. 즉, **place cells(장소 세포)**는 특정 장소에서 활성화되며, 동물이 해당 위치에 있을 때만 firing(발화)한다고 여겨져 왔다. 그러나 SR 모델에서는 place cells가 단순히 현재 위치를 인코딩하는 것이 아니라, 미래 상태(successor states)에 대한 예측적 표현(predictive representation)을 인코딩한다고 주장한다. 즉, 각 상태(state)는 미래에 방문할 상태들의 가중 합으로 표현되며, 미래 상태가 비슷한 두 지점은 유사한 표현을 갖는다.

강화학습(RL)에서 **가치 함수(value function) $$V(s)$$**는 특정 상태  $$s$$에서 시작하여 얻을 미래 보상의 기댓값으로 정의된다. 이는 할인 계수 $$\gamma$$를 포함하여 다음과 같이 표현된다.

$$
V(s) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t) \mid s_0 = s \right]
$$

여기서, $$s_t$$는 시간 $$t$$에서 방문한 상태(state), $$\gamma$$는 할인 계수 ($$0 \leq \gamma \leq 1$$), $$R(s_t)$$는 상태 $$s_t$$에서 얻는 보상을 의미한다. 그리고 SR은 현재 상태에서 시작했을 때 미래 상태들의 가중 평균을 저장하는 행렬 $$M$$로 정의된다.

$$
M(s, s') = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t \mathbf{I}(s_t = s') \mid s_0 = s \right]
$$

여기서 $$M(s, s')$$는 현재 상태 $$s$$에서 시작했을 때 미래 상태 $$s'$$를 방문할 확률의 가중치(Successor Representation), $$\mathbf{I}(s_t = s')$$는 특정 시간 $$t$$에서 $$s_t$$가 $$s'$$이면 1, 아니면 0으로 표현된다. 즉, SR 행렬 $$M(s, s')$$는 현재 상태에서 특정 미래 상태로의 이동 가능성을 할인 계수를 적용해 예측한 값이다. 이를 행렬 형태로 다시 표현하면

$$
M = \sum_{t=0}^{\infty} \gamma^t P^t
$$

이 식이 수렴하면 닫힌 형태로 다음과 같이 표현할 수 있다.

$$
M = (I - \gamma P)^{-1}
$$

즉, SR 행렬 $$M$$은 Bellman 방정식과 직접적으로 연결되며, 전이 확률 행렬 $$P$$을 기반으로 계산된다.

기존의 강화학습에서는 상태의 가치 $$V(s)$$를 보상 함수 $$R(s)$$를 이용하여 직접 계산해야 한다. 하지만 SR을 이용하면 가치 함수는 단순한 행렬 곱으로 표현될 수 있다.

$$
V(s) = \sum_{s'} M(s, s') R(s')
$$

이 식을 사용하면, 보상이 변경되었을 때도 SR 행렬 $$M$$을 유지한 채 새로운 보상 함수 $$R(s)$$만 바꾸어 즉시 새로운 가치 함수를 계산할 수 있다. 또한 SR 행렬 $$M(s, s')$$은 Temporal Difference (TD) Learning 알고리즘을 통해 점진적으로 업데이트될 수 있다.

$$
M(s, s') \leftarrow M(s, s') + \alpha \left[ \mathbf{I}(s = s') + \gamma M(s', s'') - M(s, s') \right]
$$

이 방정식은 기존의 TD 학습법과 유사하지만, 보상이 아닌 미래 상태를 예측하는 방식으로 학습이 진행된다는 점이 다르다. SR을 사용하면 보상이 변해도 $$M$$은 유지되며, $$R$$만 변경하면 즉시 $$V(s)$$를 재계산할 수 있다는 장점이 있다. 기존 모델-기반 RL에서는 전이 확률 $$P(s' | s)$$을 명시적으로 저장해야 하지만 SR은 전이 확률을 직접 저장하지 않고, 미래 상태의 가중 평균을 학습하므로 계산량이 상대적으로 적다. 또한 SR의 중요한 특징은 비슷한 미래 상태를 가지는 현재 상태는 유사한 SR 표현을 갖는다는 특징이 있는데, 이는 SR이 공간적인 과제 뿐만 아니라 비공간적 과제에도 적용 가능함을 의미한다.

<br>

## Hippocampus가 Successor Representation(SR)을 인코딩하는 방식

이 논문의 핵심 주장 중 하나는 hippocampus가 Successor Representation(SR)을 인코딩한다는 가설이다. 기존 연구에서는 hippocampus가 공간과 환경의 맥락(context)을 인코딩하며, 순차적 의사결정(sequential decision-making)에 기여한다고 알려져 있다. 이러한 역할은 hippocampus가 SR을 인코딩할 가능성을 시사한다. 이 논문에서 설명하는 predictive map 이론에서는 place cells가 단순히 현재 위치를 인코딩하는 것이 아니라, 미래 위치를 예측적으로 인코딩한다고 주장한다.

### SR 모델에서 Place Cells의 역할

SR 가설에서는 각 뉴런이 환경 내에서 미래 상태(successor state)를 인코딩한다고 본다. 현재 상태 $$s$$에서 hippocampus의 전체 뉴런 집단(population)은 **SR 행렬 $$M(s, :)$$의 한 행(row)**을 인코딩한다. 특정 뉴런이 인코딩하는 상태 $$s'$$의 발화율(firing rate)은 **현재 상태 $$s$$에서 시작했을 때 미래에 $$s'$$를 방문할 할인된 기대 횟수 (discounted expected number of visits)**와 비례한다. 이러한 SR 모델에서 각 뉴런의 발화 패턴을 **SR 장소 필드(SR place field)**라고 하며, 이는 SR 행렬 $$M(:, s')$$의 한 열(column)에 해당한다. 즉, hippocampus의 개별 뉴런은 특정 위치를 직접적으로 나타내는 것이 아니라, 해당 위치에서 시작했을 때 미래에 방문할 가능성이 높은 위치들을 반영하는 방식으로 발화한다.

또한 SR 모델은 다양한 공간적 환경에서 place cell의 발화 패턴을 예측할 수 있는데, 열린 2차원(2D) 환경에서는 전형적인 place cell은 점진적으로 감소하는 원형 발화 필드(circular firing field)를 가진다. SR 모델도 이와 유사한 예측을 하는데, 무작위 이동(random walk) 시, 동물은 현재 위치 및 주변 위치를 즉시 방문하고, 먼 위치는 더 나중에 방문하게 된다. 따라서 현재 위치와 가까운 상태일수록 미래 방문 가능성이 높으므로 인코딩 뉴런의 발화율이 더 높게 나타난다.

<figure class='align-center'>
    <img src = "/images/2025-03-14-The hippocampus as a predictive map/figure2.jpg" alt="">
    <figcaption>Figure 2. SR illustration and model comparison.</figcaption>
</figure>

하지만 SR 모델은 기존의 모델과 차별화되는 점도 존재한다. 기존의 유클리드 거리 기반 Gaussian 모델은 모든 방향에서 같은 거리만큼 발화율이 감소한다고 가정하여 (Figure 2a)발화 패턴이 장애물의 영향을 받지 않고 단순히 거리 기반으로 결정된다. 또한 지형학적(topologically-sensitive) 모델은 장애물을 고려하여, 특정 위치에서 출발해 장애물을 피해 도달하는 최단 거리(geodesic distance)를 기준으로 발화 패턴이 형성되는데, (Figure 2b) 장애물이 존재할 경우 반대편 위치로의 직접적인 경로가 없으므로, place field가 단절되어  이동 방향성(behavioral policy)의 영향을 반영하지 못한다. SR 모델(Figure 2c)은 장애물이 있으면, 반대편 공간은 방문할 가능성이 낮으므로 발화 필드가 끊어진다. 이는 topologically-sensitive한 공간 인코딩을 설명할 수 있는 강력한 특성이다.

일차원(1D) 트랙에서 동물이 일정한 방향으로 이동하는 경우(Figure 3), SR place fields는 이동 방향의 반대쪽으로 비대칭적으로 기울어진다(skewing). 동물이 일정한 방향으로 움직이면, 현재 위치에서 특정 미래 상태를 더 확실하게 예측할 수 있기 때문에 place cell의 receptive field는 예측적 특성을 가지며, 실제 상태보다 조금 앞서서 발화하기 시작한다. 이로 인해 각 개별 뉴런의 receptive field는 후방(backward) 방향으로 기울어진 패턴을 보인다. Figure 3a는 SR 모델이 예측하는 backward skewing 패턴을 보여주며, Figure 3b는 실제 동물의 hippocampus에서 기록된 place cell 발화 패턴을 나타낸다. 결과적으로, 실험에서 관찰된 발화 패턴이 SR 모델의 예측과 매우 유사하게 나타났다.

<figure class='align-center'>
    <img src = "/images/2025-03-14-The hippocampus as a predictive map/figure3.jpg" alt="">
    <figcaption>Figure 3 Behaviorally dependent changes in place fields.</figcaption>
</figure>










<figure class='align-center'>
    <img src = "/images/2025-03-14-The hippocampus as a predictive map/figure1.jpg" alt="">
    <figcaption>figure 1. caption</figcaption>
</figure>


<figure class='align-center'>
    <img src = "/images/2025-03-14-The hippocampus as a predictive map/figure3.jpg" alt="">
    <figcaption>Figure 3 Behaviorally dependent changes in place fields.</figcaption>
</figure>
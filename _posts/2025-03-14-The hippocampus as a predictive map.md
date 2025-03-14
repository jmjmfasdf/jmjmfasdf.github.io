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
    <figcaption>Figure 3. Behaviorally dependent changes in place fields.</figcaption>
</figure>

## Hippocampus에서 SR 모델의 예측과 실험적 검증

### 1. 장애물이 존재할 때 장소 필드의 변화 (Figure 3c–h)

장소 세포의 발화 패턴이 단순한 공간적 위치에 따라 고정되지 않고, 행동 정책(transition policy)에 의해 영향을 받는다는 점이 중요하다. 기존 연구에 따르면 장소 필드는 장애물 근처에서 더 크게 변형(local remapping)되며, 장애물에서 멀어질수록 변화가 적다 (Alvernhe et al., 2018). 이러한 실험 결과를 SR 모델을 통해 시뮬레이션하여 비교하였다.

SR 모델은 동물이 환경을 탐색하면서 학습한 이동 정책에 따라 장소 필드가 조정될 수 있음을 가정한다. 따라서 장애물이 추가되면 동물은 새로운 최적 경로를 따르게 되며, 이에 따라 장소 필드도 영향을 받는다. 특히, 장애물 가까이에 위치한 장소 필드는 변경된 이동 경로로 인해 발화율이 크게 변하는 반면, 장애물에서 멀리 떨어진 장소 필드는 상대적으로 적은 변화를 보인다.

Figure 3d에서는 Tolman detour maze(Figure 3c)에서 장애물이 추가되었을 때(Barrier Insertion), 장애물 근처에서 장소 필드(상단)가 급격히 변화하는 양상을 보여준다. 통계적으로도 유의미한 차이가 나타났으며(P < 0.001), 이는 장소 필드가 환경 내 장애물 배치에 따라 동적으로 조정될 수 있음을 시사한다.

Figure 3f–h에서는 SR 모델을 이용한 시뮬레이션 결과를 제시한다. Figure 3f는 사용한 미로의 모양을 제시하며, Figure 3g는 SR 모델로 시뮬레이션한 결과를 보여준다. SR 모델에서는 장애물 근처의 장소 필드(상단)가 크게 변형되었으며, 먼 지역(하단)에서는 상대적으로 변동이 적었다. 특히, 초기 장애물 삽입 위치에 따라 장소 필드의 변화 양상이 다르게 나타났으며, 이는 이동 정책이 반영된 결과임을 보여준다.

이러한 결과를 종합하면, SR 모델은 장애물이 환경에 삽입되었을 때 동물의 행동 정책 변화에 따라 장소 필드가 어떻게 조정되는지를 설명할 수 있는 강력한 모델임을 확인할 수 있다. 이는 기존 Gaussian 모델이 장애물을 고려하지 않는 한계를 극복하는 중요한 특징이다.

### 2. 보상의 영향을 받은 장소 필드의 변화 (Figure 3i–l)

장소 필드의 발화 패턴은 단순한 환경적 요소뿐만 아니라, 보상(reward)과 같은 행동적으로 중요한 요인에 의해서도 조정될 수 있다. Hollup et al.(2001)의 연구에서는 동물이 은폐된 보상(hidden reward)이 있는 위치에서 더 많은 시간을 머물렀으며, 이에 따라 보상 위치 주변에서 장소 필드가 더 밀집하는 경향이 관찰되었다. 

SR 모델이 이러한 보상의 영향을 어떻게 설명할 수 있는지 시뮬레이션을 통해 분석하였다. 이 실험은 **Hollup et al.(2001)**의 환형 수중 미로(Annular Water Maze) 실험을 기반으로 한다. 동물(주로 쥐 또는 랫)이 **수영해야만 이동할 수 있는 원형의 미로(링 모양의 수조)**에서 특정한 위치에 숨겨진 보상(플랫폼)이 있을 때, 장소 필드(place fields)가 어떻게 변하는지를 연구하였다. Figure 3ㅑ, j에서 SR 모델을 사용하여 보상 위치에서의 발화 패턴을 재현한 결과, **보상 근처에서 장소 필드가 더욱 밀집되는 경향이 나타났으며, 보상 직전 상태에서도 발화율이 증가하는 양상**이 확인되었다. 이는 SR 모델이 강화학습(reinforcement learning)의 특성을 반영하여, 보상이 있는 위치뿐만 아니라 그 직전 상태에서도 발화 패턴이 조정될 수 있음을 시사한다.

Figure 3k에서는 보상의 위치가 불확실할 때, SR 모델에서도 장소 필드가 부드럽게 변화(smoothing)하고, 발화 중심이 뒤쪽으로 이동하는 현상이 나타났다. 이는 실험에서 관찰된 asymmetric firing field를 잘 설명할 수 있는 중요한 특징이다. 그러나 SR 모델이 모든 실험 결과를 완벽하게 설명하는 것은 아니다. 예를 들어, 실험에서는 보상 위치에서 장소 필드의 크기가 일정하게 유지되었지만, SR 모델은 보상 근처에서 장소 필드의 크기가 커질 것이라고 예측하였다.(Figure 3l) 이는 SR 모델이 아직 보상에 대한 장소 필드의 변화를 완벽하게 반영하지 못하는 한계를 나타낸다.

### 3. 비공간적 환경에서의 SR 모델 적용 (Figure 4a–g)

SR 모델이 단순히 공간적(spatial) 환경에서만 유용한 것이 아니라, 비공간적(non-spatial) 태스크에서도 적용 가능할 수 있음을 검증하였다.

Schapiro et al.(2013)의 연구에서는 인간 피험자들에게 특정한 그래프(graph) 구조를 기반으로 한 프랙탈 자극(fractal stimuli)을 학습시키고, 실험 중 피험자들에게 랜덤 워크(random walk)를 통해 특정한 순서로 자극을 제시하고, 학습이 진행됨에 따라 hippocampus에서의 신경 신호 변화를 관찰하였다. Figure 4b의 결과에서, 특정 영역에서 같은 커뮤니티 내의 상태들이 강하게 연관된 신호 패턴을 보였다.

Figure 4a–d에서는 실제 실험에서 관찰된 hippocampal 신호의 패턴 유사성 결과를 보여준다. 

<figure class='align-center'>
    <img src = "/images/2025-03-14-The hippocampus as a predictive map/figure4.jpg" alt="">
    <figcaption>Figure 4. Hippocampal representations in nonspatial task.</figcaption>
</figure>

Figure 4e–g에서는 SR 모델을 적용하여 실험 결과를 재현한 결과, SR 행렬(SR matrix)에서도 동일한 커뮤니티 구조가 나타났으며, SR 기반의 다차원 스케일링(MDS, Multidimensional Scaling)을 통해 실험 결과와 유사한 구조가 형성됨이 확인되었다. 이러한 결과는 SR 모델이 공간적 정보뿐만 아니라, 일반적인 상태(state) 간의 관계를 반영하는 보편적인 기제로 작용할 수 있음을 시사한다.

### 4. 시공간적(spatiotemporal) 태스크에서의 SR 적용 (Figure 5a–f)

Hippocampus가 단순한 공간적 정보 저장소가 아니라, 공간적 정보와 시간적 정보를 통합적으로 반영하는 역할을 수행할 수 있는가? 이에 대한 검증을 위해 Deuker et al.(2016)의 연구를 기반으로 SR 모델을 적용하였다.

Figure 5a–c에서는 피험자들은 가상의 도시(virtual city)에서 특정한 물체를 탐색하는 태스크를 수행하였다. 실험 환경에서는 일반적인 이동 경로(winding path)뿐만 아니라 특정 장소 간 순간이동(teleportation)도 가능하도록 설정, 즉 공간적으로 먼 두 지점이라도 텔레포트 기능이 있으면 짧은 시간 내에 이동할 수 있는 환경이 조성되었다. 이는 공간적 근접성(spatial proximity)과 시간적 근접성(temporal proximity)을 분리할 수 있는 실험 조건을 제공한다. Figure 5b에서는 hippocampal representational similarity(해마의 신경 표현 유사성)를 분석하였다. 실험 결과, 단순히 공간적으로 가까운 장소들뿐만 아니라, 시간적으로 가까운 장소들도 hippocampus 내에서 유사한 방식으로 인코딩됨이 확인되었다.
이는 hippocampus가 순수한 공간적 정보만을 저장하는 것이 아니라, 이동 경로와 시간적 관계를 함께 반영하고 있음을 시사한다.

다만 공간적 근접성과 시간적 근접성이 서로 상관관계를 가질 수 있기 때문에, 연구자들은 공간적 요인을 통제한 상태에서 시간적 관계의 효과를 분석하였다(Figure 5c). 통계적으로 P<0.05 수준에서 유의미한 차이가 확인되었으며, 공간적 근접성과 독립적으로 시간적 근접성 또한 hippocampal 신경 표현을 결정하는 요인임을 보였다.

<figure class='align-center'>
    <img src = "/images/2025-03-14-The hippocampus as a predictive map/figure5.jpg" alt="">
    <figcaption>Figure 5. Hippocampal representations in spatiotemporal task.</figcaption>
</figure>

이후 SR 모델을 이용하여 동일한 실험 태스크를 시뮬레이션하고, 모델이 학습한 상태 표현(state representation)이 hippocampus의 신경 표현과 유사한 패턴을 나타내는지 검증하였다. Figure 5d–f에서는 SR 모델이 학습한 상태 표현을 분석하였으며, 실험 데이터에서 관찰된 공간적-시간적 관계가 SR에서도 반영됨을 확인하였다. SR 모델은 현재 상태에서 미래 상태를 예측하는 방식으로 학습되므로, 이동 경로와 소요 시간을 반영한 상태 표현을 학습하는 것이 가능하다. 이를 통해 hippocampus에서 관찰된 신경 패턴과 SR 모델의 예측이 일치한다는 점을 검증하였다.

## Entorhinal Grid Cells의 역할: Predictive Map의 차원 축소 (Dimensionality Reduction of the Predictive Map by Entorhinal Grid Cells)

Hippocampus의 공간적 표현이 SR(Successor Representation)이라는 예측 기반 지도(predictive map) 형식으로 구성된다고 가정할 때, entorhinal grid cells는 이 지도에서 차원 축소(dimensionality reduction) 역할을 수행할 수 있는가? 이 질문에 대한 답을 찾기 위해, 연구진은 grid cells가 SR의 저차원 고유벡터(eigendecomposition)를 반영한다는 가설을 제시하였다.

### 1. 기존 Grid Cell 이론과 SR 가설의 통합

Entorhinal grid cells는 격자 형태(hexagonal pattern)의 공간적 발화(spatial firing)를 가지며, 위치 추적(dead reckoning)에 기여한다는 가설이 제안된 바 있다(Hafting et al., 2005). 그러나, 다른 연구들은 grid cells의 발화 패턴이 해마(hippocampus)의 공간 지도(cognitive map)를 저차원으로 압축한 결과일 수 있다고 주장했다.

이에 따라, 본 연구에서는 grid cells가 SR의 고유벡터(eigenvectors)로 표현될 수 있으며, 이를 통해 환경의 전이(transitional dynamics) 정보를 내재적으로 반영할 수 있다는 가설을 제시하였다. 즉, grid fields는 SR의 저차원 구조를 나타내는 뉴런의 집합으로 볼 수 있으며, 이 가설의 주요 예측 중 하나는 환경의 경계(boundary conditions)에 따라 grid cell의 발화 패턴이 달라진다는 점이다.

### 2. Grid Cell의 Boundary Sensitivity 실험 (Krupic et al., 2015)

이 가설을 검증하기 위해, **Krupic et al. (2015)**는 환경의 기하학적 경계를 조작하여 grid cell의 반응이 어떻게 변화하는지 연구하였다. 연구진은 사각형(Square)과 원형(Circular) 환경을 비교하였으며, 각각의 환경에서 grid cell의 정렬(grid alignment) 패턴을 분석하였다. 그 결과, 사각형 환경에서는 grid field가 일정한 방향으로 정렬되었다.(Figure 6b) 이는  grid field의 **육각 대칭(hexagonal symmetry, 60° 간격 정렬)**으로 인해 발생할 가능성이 높다. 원형 환경에서는 Grid cells의 정렬이 불규칙적이고 가변적(variable alignment)이며, 특정 경계에 고정되지 않았다.(Figure 6c) 이는 원형 환경이 회전 대칭(rotational symmetry)을 가지며, grid cells가 특정 경계를 기준으로 정렬할 필요가 없기 때문으로 해석된다.

이러한 결과는 grid cells가 단순히 유클리드 거리(Euclidean distance)를 기반으로 하는 것이 아니라, 환경의 구조적 요소를 반영하여 작동할 가능성을 시사한다.

### 3. 사각형과 사다리꼴 환경 비교: Split-Halves 분석 (Figure 6d, 6e 참고)

<figure class='align-center'>
    <img src = "/images/2025-03-14-The hippocampus as a predictive map/figure6.jpg" alt="">
    <figcaption>Figure 6 Grid fields in geometric environments.</figcaption>
</figure>

Krupic et al. (2015)는 grid cell이 환경의 비대칭적 구조(asymmetrical boundaries)에 따라 다르게 반응하는지 확인하기 위해, 사각형(sq)과 사다리꼴(tr) 환경을 비교하였다. 사각형 환경에서는 grid field가 환경의 양쪽에서 유사한 패턴을 보였고, grid cell의 발화 패턴은 환경 좌우(side-to-side)에서 큰 차이가 없었다.(Figure 6d) 사다리꼴 환경에서는 grid field의 구조가 양쪽에서 다르게 나타났는데, 환경이 **비대칭적(asymmetrical)**으로 변경될 경우, grid cell의 정렬이 깨지고, 양쪽에서 서로 다른 발화 패턴을 형성하는 것이 관찰되었다. 특히, 자동 상관분석(autocorrelation analysis) 결과, trapezoidal 환경에서는 grid pattern의 반복 주기가 양쪽에서 달라지는 것으로 확인되었다(Figure 6e).

이러한 결과는 grid cell이 단순한 공간적 거리 측정 도구가 아니라, 환경의 전이 확률(transition probability)과 경계 구조를 반영하는 예측 기반 표현(predictive representation)을 구성할 가능성이 높다는 점을 시사한다.

### 4. SR Eigenvector 모델을 통한 Grid Field 시뮬레이션 (Figure 6f–h 참고)

연구진은 SR 모델을 기반으로 한 시뮬레이션을 수행하여, **grid field가 환경의 기하학적 구조에 따라 어떻게 변화하는지** 분석하였다. SR 모델에서 추출한 고유벡터(eigenvectors)를 이용하여 grid field를 시뮬레이션한 결과, 실험 결과와 유사한 패턴이 관찰되었다. (Figure 6f–h). 또한 원형 환경에서 eigenvector의 방향성이 불규칙적으로 변하는 반면, 사각형 환경에서는 경계를 따라 정렬되는 경향을 보였다. (Figure 6g, j). 사각형 환경에서 사다리꼴 환경으로 이동하면, SR의 고유벡터도 비대칭적으로 변화하며, grid field의 구조가 달라졌다. 이러한 결과는 grid cell이 SR의 저차원 고유벡터를 반영하며, 환경의 구조에 따라 그 표현이 변화한다는 가설을 뒷받침한다.

### 5. 


## Subgoal discovery using grid fields




<figure class='align-center'>
    <img src = "/images/2025-03-14-The hippocampus as a predictive map/figure6.jpg" alt="">
    <figcaption>Figure 6 Grid fields in geometric environments.</figcaption>
</figure>
---
title: "Controlling human causal inference through in silico task design"
date: 2025-03-10
tags:
    - Task Control
    - Reinforcement Learning
categories: 
    - Paper Review
toc: true
toc_sticky: false
---

인과 관계를 학습하는 능력은 생존에 필수적이다. 인간의 두뇌는 높은 기능적 유연성을 지니고 있어 효과적인 인과 추론(causal inference)이 가능하며, 이는 다양한 학습 과정의 근간을 이룬다. 기존 연구들은 환경적 요인이 인과 추론에 미치는 영향에 초점을 맞추어왔지만, 본 연구에서는 근본적인 질문을 제기한다: **이러한 환경적 요인을 전략적으로 조작하여 인과 추론을 통제할 수 있을까?**

이 논문은 "과제 제어(task control) 프레임워크" 를 제안하여 인간의 인과 학습을 조율하는 방법을 연구한다. 연구진은 2인 게임(two-player game) 구조 를 활용하여, 신경망(neural network)이 인간 인과 추론 모델과 상호작용하면서 실험 과제의 변수를 조작하는 방식을 학습하도록 설계하였다. 구체적으로, 태스크 컨트롤러(task controller) 가 실험 디자인을 최적화하면서 인과 구조의 복잡성을 반영할 수 있는지를 검증하였다. 126명의 인간 참가자를 대상으로 한 실험 결과, 태스크 컨트롤을 통해 인과 추론의 성과 및 학습 효율을 조정할 수 있음이 확인되었다. 특히, 태스크 컨트롤 정책은 인간의 인과 추론 과정에서 나타나는 "원샷 학습(one-shot learning)" 의 특성을 반영하는 것으로 나타났다. 이러한 연구 프레임워크는 인간 행동을 특정 방향으로 유도할 수 있는 응용 가능성을 제시 하며, 향후 교육, 치료, 그리고 인공지능을 활용한 학습 최적화 등의 분야에서 중요한 역할을 할 것으로 기대된다.

<br>

# Introduction

**인과 추론(causal inference)**은 관찰을 통해 원인과 결과의 관계를 학습하는 능력을 의미하며, 이는 강화 학습(reinforcement learning, RL)과 의사결정 과정에서 핵심적인 역할을 한다. 기존 연구에서는 동물 실험을 통해 최소한 두 가지 다른 방식의 인과 학습 패턴이 존재한다는 사실이 밝혀졌다. 첫 번째는 **점진적 학습(incremental inference)**으로, 충분한 경험을 쌓은 후 점진적으로 결과를 예측하는 방식이다. 반면, 두 번째는 **원샷 학습(one-shot inference)**으로, 단 한 번의 경험을 통해 강력한 신념을 형성하는 방식이다. 이러한 행동 패턴은 동물뿐만 아니라 인간에게도 동일하게 나타나며, 최신 연구에서는 인간의 뇌가 두 가지 학습 방식을 유연하게 조절할 수 있다는 신경학적 근거가 발견되었다. 이는 인간이 환경 속에서 복잡한 원인-결과 관계를 동적으로 해결하며 새로운 과제를 학습할 수 있음을 의미한다.

이전 연구들은 인간의 인과 추론이 환경적 요인에 의해 영향을 받는다는 점을 분석하는 데 집중해왔다. 예를 들어, 사람들이 새로운 과제를 학습할 때, 환경이 불확실성이 높은 경우에는 신뢰할 수 있는 성과를 보장하기 위해 점진적 학습을 선호하는 경향이 있다. 반면, 불확실성이 적은 환경에서는 보다 신속하고 효율적인 원샷 학습을 활용하는 것이 최적의 전략이 될 수 있다. 이러한 연구들은 인간의 인과 추론이 단일한 방식이 아니라 환경적 맥락에 따라 유연하게 조정된다는 사실을 보여준다.

이러한 맥락에서 연구진은 인간의 **인과 추론 과정이 환경적 요인에 의해 자연스럽게 조정**되는 것이 아니라, **전략적으로 조작**될 수 있는지에 대한 근본적인 질문을 던진다. 즉, 인간의 인지 과정을 특정 방향으로 유도할 수 있는지가 핵심적인 연구 질문이다. 이를 탐구하기 위해 연구진은 **"태스크 컨트롤(task control)"**이라는 개념을 제안하였다. 태스크 컨트롤은 인간의 인과 학습 과제를 설계하고 조율하는 프레임워크로, 딥러닝 기반의 태스크 컨트롤러(deep-RL-based task controller)를 활용하여 인간의 인과 추론 모델과 상호작용하면서 학습 변수를 조작하는 방식으로 설계되었다. 이 태스크 컨트롤러는 인간의 인과 추론 모델이 학습하는 과정을 탐색하고, 실험 변수를 조작하여 인간의 학습 성과와 효율성을 변화시키는 방식을 학습한다.

연구진은 태스크 컨트롤이 인간의 인과 추론을 조절할 수 있는지를 검증하기 위해, 세 가지 태스크 컨트롤 조건을 설정하여 실험을 진행하였다. 첫 번째 조건인 Bayesian+는 **점진적 학습 방식을 촉진하여 충분한 학습 시간이 주어졌을 때 신뢰성 높은 성과를 보장**하는 것을 목표로 한다. 두 번째 조건인 Oneshot+는 **신속한 학습을 유도하여 짧은 시간 내에 높은 학습 효율을 달성**하는 것을 목적으로 한다. 마지막으로 Oneshot– 조건은 **학습을 어렵게 만들어 인과 추론을 방해**하며, 이를 통해 동기 부여, 트라우마 억제, 나쁜 습관 제거 및 심리적 회복력 증진과 같은 응용 가능성을 탐색하는 데 초점을 맞춘다.

126명의 인간 참가자를 대상으로 한 실험 결과, 태스크 컨트롤을 통해 인과 추론의 성과 및 학습 효율을 조정할 수 있음이 실증적으로 확인되었다. 특히, 태스크 컨트롤의 효과는 개별 참가자의 학습 특성을 반영한 인지 모델을 사용할 때 더욱 극대화되는 것으로 나타났다. 이는 태스크 컨트롤이 단순히 실험 디자인을 최적화하는 것이 아니라, 인간 인과 추론의 본질적인 특성을 반영하는 방식으로 작동한다는 점을 시사한다.

<br>

# Results

연구진은 인간의 인과 추론 학습 과정에서 태스크 컨트롤이 어떻게 작용하는지를 분석하기 위해, 126명의 성인 참가자를 모집하여 실험을 수행했다. 총 126명(남녀 포함, 연령 18~56세) 참가
참가자들은 사전에 정신적·신경학적 장애가 없는지 확인, 모든 참가자는 실험 전에 서면 동의(Informed Consent)를 받았으며 세 가지 태스크 컨트롤 조건(Bayesian+, Oneshot+, Oneshot–) 중 하나에 무작위로 배정되었다.

## Task control: In silico task design for guiding human causal inference

이 연구에서 제안하는 태스크 컨트롤(Task Control) 프레임워크 는 인간의 인과 추론 과정을 조정하기 위한 컴퓨터 기반 실험 설계(in silico task design) 방식이다. 기존의 과제 학습(task learning)에서는 학습자가 환경과 상호작용하며 인과 관계를 학습하지만, 본 연구에서는 이를 역으로 조작하여 과제가 인간의 인지 모델을 조정하도록 설계 한다. 즉, 태스크 컨트롤러(task controller) 가 인간의 인과 추론 모델과 상호작용하며 실험 변수를 조정함으로써 특정한 학습 경로를 유도하는 방식이다.

- 1. 마르코프 결정 과정(MDP) 정의
태스크 컨트롤러 문제는 5-튜플 $$ (U, A, T, R, \gamma) $$ 로 구성된 MDP로 모델링된다.

$$ U $$: 상태 공간 (State space)
$$ A $$: 행동 공간 (Action space)
$$ T $$: 전이 함수 (Transition function)
$$ R $$: 보상 함수 (Reward function)
$$ \gamma $$: 할인 계수 (Discount factor)

- 2. MDP 구성 요소 설명

태스크 컨트롤러의 상태 $$ u_i $$ 는 특정 시점 $$ i $$ 에서 각 S-O 페어(Stimulus-Outcome Pair)의 인과적 불확실성(causal uncertainty) 을 포함하는 벡터로 정의된다.
$$
u_i = \{ u_1, u_2, ..., u_N \} \in U
$$

$$ u_k $$ 는 특정 S-O 페어 $$ p_k $$ 의 인과적 불확실성이며, 인과적 불확실성 $$ u_k $$ 는 학습자가 해당 S-O 관계를 얼마나 신뢰하는지에 대한 불확실성을 나타낸다. 연구에서는 베이지안 인과 추론 모델(Bayesian Inference Model) 또는 원샷 학습 모델(One-shot Learning Model) 을 사용하여 이를 정량화했하였다. 태스크 컨트롤러의 목표 상태(goal state) $$ u_{\text{goal}} $$ 은 모든 S-O 관계의 인과적 불확실성이 특정 임계값 이하로 낮아진 상태이다.

$$
u_{\text{goal}} = \{ u_1, u_2, ..., u_N \}, \quad \text{where } u_k \leq u_{\text{TH}}, \forall k
$$

즉, 태스크 컨트롤러는 인간 학습자가 가능한 한 빨리 인과적 불확실성을 줄일 수 있도록 실험 조건을 조정하는 역할을 한다.

태스크 컨트롤러가 특정 상태 $$ u_i $$ 에서 행동 $$ a_i $$ 를 선택하면, 인간 인지 모델이 다음 상태 $$ u_{i+1} $$ 로 전이되며, 전이 확률 $$ P(u_i, a_i, u_{i+1}) $$ 는 인지 모델 M에 의해 결정되는 함수 로 표현된다.

$$
T: U \times A \times U \rightarrow [0, 1]

P(u_i, a_i, u_{i+1}) = M(u_i, a_i, u_{i+1})
$$

여기서 $$ M $$ 은 인간의 인과 학습 과정을 설명하는 인지 모델이며 어떤 자극-결과 페어를 보여주었을 때, 인간 학습자가 해당 관계를 학습하는 정도를 반영한다. 또한 태스크 컨트롤러는 인간 인지 모델이 목표 상태(goal state)에 도달하도록 유도 해야하기 때문에 이를 위해 보상 함수 $$ R $$ 을 다음과 같이 정의한다. 목표 상태 $$ u_{\text{goal}} $$ 에 도달하면 +1의 보상 을 부여하고, 그렇지 않으면 -1의 보상 을 부여한다.

$$
R: U \times A \times R \rightarrow R
$$

$$
R(U) =
\begin{cases}
1, & \text{if } u_U \leq u_{\text{TH}} \\
-1, & \text{otherwise}
\end{cases}
$$

- 3. 태스크 컨트롤러의 학습 과정

태스크 컨트롤러의 학습 목표는 기대 보상(expected return)을 최대화하는 최적 정책 $$ \pi^* $$ 을 찾는 것이다. $$ V^{\pi}(u) $$ 는 특정 상태 $$ u $$ 에서 정책 $$ \pi $$ 를 따를 때 기대되는 총 보상 값이다.

$$
V^{\pi}(u) = \mathbb{E}_{\pi} \left[ \sum_{i=0}^{M-1} \gamma^i R_{i+1} | u_0 = u, \pi \right]
$$

또한 태스크 컨트롤러는 Dueling DDQN 구조를 활용하여 학습 성능을 개선하였다. $$ Q(s, a) $$ 는 현재 상태에서 행동 $$ a $$ 를 선택했을 때의 기대 보상을 나타낸다.

$$
Y^{DDQN}_t = R_{t+1} + \gamma Q(s_{t+1}, \arg\max_a Q(s_{t+1}, a; \theta); \theta^-)
$$

## Key concepts

- 1. 인과적 불확실성 (Causal Uncertainty)

주어진 자극(S, Stimulus) 과 결과(O, Outcome) 사이의 인과 관계는 확률 분포 로 정의된다. 이때, 확률 분포의 평균(mean)은 인과 강도(causal strength) 를 나타내고, 분산(variance)은 인과적 불확실성(causal uncertainty) 을 의미한다. 이 실험에서 인과적 불확실성은 특정 S-O 관계의 강도에 대한 불확실성 수준을 나타내게 된다.  인과적 불확실성 값이 크면, 해당 관계에 대한 신뢰도가 낮고 학습이 필요한 상태를 의미한다. 연구에서는 인간 인과 추론 모델(cognitive model of human causal inference) 을 활용하여 태스크 컨트롤러 학습 과정에서 인과적 불확실성을 추정하도록 한다.

- 2. 인과적 복잡성 (Causal Complexity)

인과적 복잡성은 하나의 지식 베이스(Knowledge Base, KB) 내에서 존재하는 S-O 관계의 수 를 의미한다. 인과적 복잡성이 높을수록, 더 많은 인과적 연관 관계(그래프 내의 간선 수)가 존재한다. 복잡한 환경에서는 학습자가 더 많은 S-O 페어를 탐색하고 학습해야 하므로 학습 난이도가 증가한다. 예를 들어 단순한 경우(Simple case)는 5개의 S-O 관계 (Figure 3B-left), 복잡한 경우(Complex case)는 17개의 S-O 관계 (Figure 3A-right)를 사용한다.

## Task control problem

태스크 컨트롤러는 기대 보상(expected return)을 최대화하는 방향으로 학습된다. 이를 위해, 다음과 같은 강화 학습 목표를 설정한다.

$$
V^{\pi}(u) = \mathbb{E}_{\pi} \left[ \sum_{i=0}^{M-1} \gamma^i R_{i+1} | u_0 = u, \pi \right]
$$

$$ V^{\pi}(u) $$ 는 특정 상태 $$ u $$ 에서 정책 $$ \pi $$ 를 따를 때 기대되는 총 보상 값이다. $$ \gamma $$ 는 할인 계수(discount factor)로, 미래 보상의 중요도를 조절하는 값이 된다. 즉, 태스크 컨트롤러는 학습자가 빠르고 효과적으로 인과적 불확실성을 줄이도록 유도하는 방식으로 학습된다. 상술하였듯이 태스크 컨트롤러의 상태 $$ u $$ 는 각 S-O 페어의 인과적 불확실성 값으로 표현되는데, 초기 상태는 학습자의 초기 인과적 불확실성 수준 에 의해 결정된다. 실험에서, 태스크 컨트롤러는 각 단계에서 학습자가 어떤 S-O 페어를 학습해야 하는지를 결정하는 역할을 한다. 결국 태스크 컨트롤러는 각 단계에서 학습자의 상태 $$ u $$ 를 분석하고, 최적의 행동 $$ a $$ 를 선택하여 보상을 최대화하는 방향으로 학습된다.

## Computational Model of Causal Human Inference

### 1. 인과 학습 모델의 유형

- 1. 베이지안 인과 추론 모델 (Bayesian Inference Model)

인간이 점진적으로(Step-by-Step) S-O 관계를 학습 한다는 가설을 기반으로 한다. 동일한 S-O 페어를 반복적으로 경험할수록 인과적 불확실성($$ u $$)이 감소 하며, 학습이 이루어진다. 태스크 컨트롤러 Bayesian+ 는 이 모델을 기반으로 학습자를 점진적 학습 패턴으로 유도한다.

$$
\text{If } p_k \text{ is observed multiple times, then } u_k \to 0
$$

즉, S-O 관계를 반복적으로 노출하면 학습자가 점진적으로 인과 관계를 신뢰하게 된다. 이는 Bayesian+ 컨트롤러의 원리 이며, 태스크 컨트롤러는 학습자가 점진적으로 불확실성을 줄이도록 최적의 실험 설계를 학습 한다.

- 2. 원샷 인과 학습 모델 (One-shot Causal Learning Model)

인간이 한 번의 경험만으로도 강한 인과적 신념을 형성할 수 있다 는 가설을 기반으로 한다. 베이지안 모델과 달리, 학습률(learning rate)이 동적으로 변하며, 인과적 불확실성에 따라 조정된다. 태스크 컨트롤러 Oneshot+ 및 Oneshot- 는 이 모델을 활용하여 원샷 학습을 촉진하거나 억제할 수 있다.

$$
\text{If } p_k \text{ is observed once, then } u_k \approx 0
$$

즉, Oneshot+ 컨트롤러는 특정 S-O 페어를 한 번만 경험해도 신뢰하도록 학습을 유도 한다. 반면, Oneshot- 컨트롤러는 원샷 학습 효과를 최소화하고, 점진적 학습을 유도 한다.

### 2. 인과 학습 모델의 수학적 표현

- 1. 잠재 클래스 모델

인간의 인과 추론을 모델링하기 위해 잠재 클래스 모델(Latent Class Model)을 적용 한다. 이 모델은 S-O 관계의 확률을 다음과 같이 정의한다.

$$
P(O | q) =
\begin{cases}
q_1, & \text{if } O = S_1 \\
q_2, & \text{if } O = S_2 \\
q_3, & \text{if } O = S_3
\end{cases}
$$

$$ q_i $$ 는 $$ S_i $$ 가 결과 $$ O $$ 를 유발할 확률을 나타내며, 확률 값 $$ q $$ 는 디리클레 분포(Dirichlet Distribution) 를 따른다.

$$
(q_1, q_2, q_3) \sim Dir(\lambda_1, \lambda_2, \lambda_3)
$$

사전 확률(prior)은 균등 분포(uniform)로 설정되며, 특정 S-O 페어 $$ p $$ 가 여러 번 제시되면, 사후 확률(posterior probability)이 업데이트된다.

- 2. 베이지안 인과 추론 모델 (Bayesian Inference Model)

베이지안 모델은 모든 S-O 페어가 동일한 학습 효과를 갖도록 가정 한다. 학습률 $$ g_i $$ 는 고정된 상수($$ C $$) 로 설정되며, 학습 과정에서 변하지 않는다. 즉, 특정 S-O 페어가 반복될수록 학습자가 해당 관계를 더 강하게 신뢰하게 된다.


$$\Delta a_i = g_i x_i, \quad \text{where } g_i = C$$

$$E(q_i | E) = \frac{a_i}{a_0}$$

$$Var(q_i | E) = \frac{a_i (a_0 - a_i)}{a_0^2 (a_0 + 1)}$$

평균 $$ E(q_i | E) $$ 는 인과적 강도를 나타내며, 반복 노출될수록 값이 증가 한다. 반대로 분산 $$ Var(q_i | E) $$ 는 인과적 불확실성을 나타내며, 반복 노출될수록 값이 감소 한다.

- 3. 원샷 인과 학습 모델 (One-shot Causal Learning Model)

원샷 모델은 학습률이 동적으로 조정 된다. 특정 S-O 페어의 불확실성이 크면, 해당 페어의 학습률이 증가하고, 반대로 불확실성이 낮으면 학습률이 감소한다.

$$
g_i = \frac{\exp(t \cdot Var(q_i | E))}{\sum_j \exp(t \cdot Var(q_j | E))}
$$

위 식에서 $$ t $$ 는 온도(temperature) 파라미터로, 학습률의 변화를 조정하는 역할을 한다. 결과적으로, 불확실성이 높은 S-O 페어일수록 더 빠르게 학습 된다.

$$
\text{If } Var(q_i | E) \text{ is high, then } g_i \text{ is large.}
$$

즉, Oneshot+ 컨트롤러는 특정 S-O 관계가 한 번만 제시되더라도 강한 신뢰를 형성하도록 유도 한다. 반면, Oneshot- 컨트롤러는 원샷 학습을 억제하고 점진적 학습을 유도하는 방식으로 작동한다. 이 컨트롤러는 학습자가 여러 번 반복적인 경험을 통해서만 인과 관계를 학습할 수 있도록 학습률을 낮추는 방향으로 실험을 조정한다. 이를 통해 학습자는 특정 S-O 관계를 여러 번 경험해야만 신뢰를 형성할 수 있으며, 인과적 불확실성이 점진적으로 감소하도록 설계된다.

## 태스크 컨트롤러의 강화 학습 과정

태스크 컨트롤러의 학습 과정은 강화 학습 기반의 반복적인 상호작용 을 통해 이루어진다. 이 과정에서 태스크 컨트롤러는 인간 인지 모델(cognitive model)과 반복적으로 상호작용하며 최적의 정책을 학습 한다. 연구진은 이러한 상호작용을 이중 플레이어 확률 게임(Two-player Stochastic Game) 으로 모델링하였다. 이는 마르코프 결정 과정(MDP, Markov Decision Process) 및 반복 게임(Repeated Games)을 확장한 개념으로, 태스크 컨트롤러와 인간 인지 모델이 서로 적응하며 학습하는 구조를 반영 한다.

### 1. 확률 게임(Stochastic Game) 모델링

태스크 컨트롤러의 학습은 6-튜플(6-Tuple) 확률 게임 으로 정의된다.

$$
(U, N, A, T, r, \gamma)
$$

이 게임에서는 태스크 컨트롤러와 인간 인지 모델이 상호작용하면서 학습이 진행 된다. 각 플레이어(태스크 컨트롤러, 인간 인지 모델)는 자신의 전략을 조정하며 최적의 학습 조건을 찾아가는 과정 을 거친다.
각 플레이어는 자신의 행동 이력(history)에 따라 정책(policy)을 선택하며, 이러한 정책의 조합을 공동 정책(Joint Policy) $$ p = (p_1, p_2, ..., p_N) $$ 라고 한다. 공동 정책을 통해 게임이 반복적으로 진행되며, 각 플레이어는 자신에게 최대한 유리한 학습 경로를 찾아가도록 학습 한다.

### 2. 최적 정책 학습 과정

확률 게임에서 각 플레이어는 최대 누적 보상(Expected Cumulative Reward) 을 얻기 위한 전략을 세운다. 각 플레이어 $$ i $$ 의 보상은 다음과 같이 정의된다.

$$
V^{p, i}(u) = \mathbb{E}_p \left[ \sum_{t=0}^{M-1} \gamma^t r_i (u_t, a_t) | u_0 = u, p \right]
$$

여기서, $$ V^{p, i}(u) $$ 는 특정 상태 $$ u $$ 에서 플레이어 $$ i $$ 가 정책 $$ p $$ 를 따를 때 기대되는 누적 보상 값이다. $$ \gamma $$ 는 할인 계수로, 미래 보상의 중요도를 조절하는 값이다.
각 플레이어는 자신의 보상을 최대화하는 정책을 찾는 과정에서 최적 전략을 학습 하게 된다. 이 과정에서 최적 응답(Best Response, BR) 정책을 찾는 것이 핵심 목표가 된다.

$$
p^*_i \in BR(p^{-i}) = \arg\max_{p_i} V^{p, i}(u) \quad \text{(given that other players' policies are fixed)}
$$

즉, 각 플레이어는 상대방의 전략이 고정된 상태에서 자신이 최대한 높은 보상을 얻을 수 있는 최적 정책을 찾는다.

### 3. 내쉬 균형 (Nash Equilibrium) 적용

이 게임에서 내쉬 균형(Nash Equilibrium) 이 존재한다는 것은, 모든 플레이어가 자신의 전략을 바꿔도 추가적인 이득을 얻을 수 없는 상태에 도달할 수 있다는 것을 의미 한다. 즉, $$p^* = (p^*_1, p^*_2, ..., p^*_N)$$ 일 때, 어느 플레이어도 자신의 정책을 단독으로 변경하여 추가적인 보상을 얻을 수 없다.

내쉬 균형을 통해, 태스크 컨트롤러와 인간 인지 모델은 각각의 학습 전략이 수렴하는 최적 지점 을 찾게 된다.

### 4. 태스크 컨트롤러의 강화 학습 적용

태스크 컨트롤러는 Dueling Double Deep Q-Network (Dueling DDQN) 을 활용하여 학습된다. DDQN은 정책(policy)과 가치(value) 사이의 일관성을 유지하면서도 최적 학습 경로를 찾을 수 있도록 설계된 강화 학습 알고리즘 이다. DDQN의 업데이트 식은 다음과 같이 정의된다.

$$
Y^{DDQN}_t = R_{t+1} + \gamma Q(s_{t+1}, \arg\max_a Q(s_{t+1}, a; \theta); \theta^-)
$$

여기서,

$$ Q(s, a) $$`` 는 현재 상태에서 행동 $$a$$를 선택했을 때의 기대 보상을 의미한다. $$ \theta^- $$ 은 타겟 네트워크(target network) 파라미터이다.

### 5. 모델 학습 설정 및 하이퍼파라미터
태스크 컨트롤러는 DDQN을 사용하여 다양한 실험 조건에서 강화 학습을 수행 한다. 실험에서는 두 가지 수준의 지식 베이스(KB) 복잡도를 고려하였으며, 각 실험 조건에 따라 6개의 DDQN 에이전트를 학습 시켰다.

할인 계수($$ \gamma $$) = 0.99
학습률($$ \alpha $$) = 0.0001
타겟 네트워크 업데이트 빈도($$ \tau $$) = 0.001
배치 크기($$ \text{batch size} $$) = 32로,

각 에이전트는 100만 개의 에피소드(1M episodes) 를 경험하며 학습되었으며 각 에피소드에서 최대 100회의 상호작용을 수행하여 목표 상태에 도달하도록 학습 되었다.





<figure class='align-center'>
    <img src = "/images/2025-03-10-Controlling human causal inference through in silico task design/figure1.jpg" alt="">
    <figcaption>figure 1. In silico task design for controlling human inference processes</figcaption>
</figure>

<figure class='align-center'>
    <img src = "/images/2025-03-10-Controlling human causal inference through in silico task design/figure2.jpg" alt="">
    <figcaption>figure 2. Behavioral analyses of the trained task controller</figcaption>
</figure>

<figure class='align-center'>
    <img src = "/images/2025-03-10-Controlling human causal inference through in silico task design/figure3.jpg" alt="">
    <figcaption>figure 3. Experimental design – causal learning tasks</figcaption>
</figure>

<figure class='align-center'>
    <img src = "/images/2025-03-10-Controlling human causal inference through in silico task design/figure4.jpg" alt="">
    <figcaption>figure 4. Effect of task control on subjects’ behavior and performance</figcaption>
</figure>

<figure class='align-center'>
    <img src = "/images/2025-03-10-Controlling human causal inference through in silico task design/figure5.jpg" alt="">
    <figcaption>figure 5. Comparison of task control effect between the subject-cognitive model compatible and incompatible case (post hoc analysis)</figcaption>
</figure>

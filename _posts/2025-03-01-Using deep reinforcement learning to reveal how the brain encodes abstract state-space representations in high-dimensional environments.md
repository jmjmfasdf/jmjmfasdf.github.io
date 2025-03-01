---
layout: post
title: "Using deep reinforcement learning to reveal how the brain encodes abstract state-space representations in high-dimensional environments"
author: jeongmin seo
tags: [Reinforcement Learning, fMRI, Task Representation]
categories: Paper Review
color: rgb(25, 25, 112)
feature-img: 
thumbnail: 
---

- 논문의 이해를 위해 SARSA, Q-learning, DQN 등의 기본적인 강화학습 알고리즘, 그리고 뇌에서 일어나는 시각 처리 과정(Visual pathway)에 대한 이해를 권장합니다.

인간은 고차원적인 감각 정보를 이용하여 주어진 상황에서 결정을 내리는 데에 탁월한 강점을 가지고 있다. 그럼에도 불구하고 뇌에서 어떻게 주변의 상태를 뇌 내에서 간결하게(compactly) 표현(represent)하는지에 대해서는 명확하게 밝혀진 바가 없다. Deep reinforcement learning model 중 DQN은 고차원 입력을 nonlinear activation을 통해 action value의 형태로 반환한다. 논문의 저자는 상당히 재미있는 시도를 하였는데, DQN의 nonlinear activation이 '사람과 유사하게' 일어나는지 알아보았다.

<br>

## Figure 1. explanation

저자는 실험 디자인에 아타리 게임 중 Pong, Enduro, Space invaders를 사용하였다. Pong은 양 쪽에서 판들이 왔다갔다 하며 움직이는 공을 넘기는 게임, Enduro는 3인칭 시점에서 차를 좌우로 운전하며 차를 피하는 게임, 그리고 Space Invaders는 비행기를 조종하며 적을 맞추는 슈팅 게임이다. 게임의 플레이 화면은 아래의 그림에 나타나 있다.

{% include aligner.html images="images/2025-03-01-Using deep reinforcement learning to reveal how the brain encodes abstract state-space representations in high-dimensional environments/figure1.jpg" caption="figure 1. atari game setup and DQN" %}

위의 두 번째 그림은 연구에서 사용한 DQN의 구조이다. DQN은 4개의 Conv layer, 1개의 FC layer, 그리고 Q-value output layer로 구성되어 있다. 

앞으로 실험 결과 설명에 앞서서, 연구 디자인에 대해서 간단하게 설명하겠다. 총 6명의 참가자(이하 sub001 ~ sub006)가 존재하며, 이들이 3종류의 게임을 플레이하는 동시에 fMRI데이터를 측정하였다. 이외에 사용한 자원과 모델은 뒤에서 설명하도록 하겠다.

<br>

## Figure 2. Basic concepts

{% include aligner.html images="images/2025-03-01-Using deep reinforcement learning to reveal how the brain encodes abstract state-space representations in high-dimensional environments/figure2.jpg" caption="figure 2. basic concepts" %}

DQN 모델은 6명의 사용자와 관련없이 독립적으로 학습되었고, 각각 게임에 대해 개별적으로 학습되었다. figure 2에서는 사용자의 프레임을 주고 pretrained DQN에게 행동을 선택하도록 했고, 그 상황에서의 action value를 표시한 그림이다. 가로축은 DQN이 선택한 action, 그리고 그 상황에서 사람이 어떤 행동을 취했는지 별도로 표시하였다. 세로축은 이 행동에 대해 DQN이 평가한 action value이다. 결과를 보면 사람이 장애물을 피하기 위해 왼쪽으로 움직인 선택에 대하여 DQN도 왼쪽으로 선택하며 action value를 더 높게 평가하는 것을 볼 수 있다. 이를 통해 DQN이 인간과 유사한 방식으로 상태를 평가하고, 이를 행동에 반영할 수 있음을 알 수 있다.

아래 그림은 사용자의 게임 플레이 데이터를 DQN에 통과시켜 hidden layer의 값을 저장한 후, 이 값을을 PCA로 압축 후 100PC를 사용해 logistic regression으로 사람의 행동을 예측한 결과이다. 즉 이는 DQN으로 사람의 게임 플레이 데이터를 처리하였을 때, 이 데이터를 행동 예측에 사용할 수 있는가? 를 평가하기 위해 진행하였다고 생각하면 된다. 그 결과는 그림과 같이 6명의 참가자 모두 유의미한 정도로 행동 예측이 가능하다는 결론이 나왔다. 여기까지의 내용을 간단하게 정리하면

> - DQN의 내부 representation이 인간의 행동을 근사할 수 있으며, 이는 게임에 따라 예측 성능이 다르게 나타난다.

로 정리할 수 있다.

<br>

## Figure 3. Encoding model

이제 DQN이 사람의 행동을 근사할 수 있다는 것 까지는 알았다. 그렇다면 '어떻게' 가 밝혀져야 한다. figure 3은 DQN의 hidden layer가 사람의 뇌의 state-space representation이 어떻게 나타나는지 알아보기 위한 그림이다. 

{% include aligner.html images="images/2025-03-01-Using deep reinforcement learning to reveal how the brain encodes abstract state-space representations in high-dimensional environments/figure3.jpg" caption="figure 3. encoding model" %}

연구자들은 6명이 실험자의 게임 플레이 데이터를 DQN에 통과시킨 후, 각 4개의 hidden layer에서 PCA를 수행, 100개의 PC를 추출했다. 이 PC를 가지고 ridge regression을 수행하여 6명의 실험자의 fMRI 데이터와 대조, 각 voxel의 반응도를 예측하는 모델을 구축했다. 예측도는 원래 fMRI데이터와 대조하여 pearson correlation으로 평가되었다. 그렇게 분석을 한 결과가 위와 같은데, 결과가 꽤 인상적이다. 각 게임마다 활성화되는 부위가 조금씩 다른데, 상관계수 r가 높은 것이 유의미하게 드러나는 것을 볼 수 있다. B-D까지의 그림을 보면 공통적으로 visual cortex - posterior parietal cortex까지의 활성이 두드러지는 것을 알 수 있다.

그림 E에서는 DQN의 hidden layer에서 추출한 정보의 유의미한 voxel의 비율을 voxel을 ROI별로 분류하여 나타낸 것인데, 확실히 **시각 영역(v1~v4), 공간정보처리 영역(spl, sg), 운동 영역(mc, sma, sfg)**에서 유의미한 voxel 수가 많은 것을 알 수 있다. 비율이 그렇게 높지 않아 이상하게 보일 수 있지만, fMRI의 해상도가 보통 2-3세제곱밀리미터이고, 뇌에는 수천-수만개의 voxel이 존재한다는 것을 감안하면 매우 보수적인 보정을 거쳐 나온 결과이기 때문에(여기서는 자세한 보정 알고리즘은 설명하지 않음) 충분히 유의미한 결과라 해석할 수 있다. 다르게 말하면 절대적인 비율이 낮더라도 특정 voxel과 확실히 DQN의 representation이 연관된다는 점을 집중해서 해석하면 되겠다.

## Figure 4. Control model

{% include aligner.html images="images/2025-03-01-Using deep reinforcement learning to reveal how the brain encodes abstract state-space representations in high-dimensional environments/figure4.jpg" caption="figure 4. control model" %}



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

## Figure 1. explanation

저자는 실험 디자인에 아타리 게임 중 Pong, Enduro, Space invaders를 사용하였다. Pong은 양 쪽에서 판들이 왔다갔다 하며 움직이는 공을 넘기는 게임, Enduro는 3인칭 시점에서 차를 좌우로 운전하며 차를 피하는 게임, 그리고 Space Invaders는 비행기를 조종하며 적을 맞추는 슈팅 게임이다. 게임의 플레이 화면은 아래의 그림에 나타나 있다.

![figure 1. atari game setup and DQN](images\2025-03-01-Using deep reinforcement learning to reveal how the brain encodes abstract state-space representations in high-dimensional environments\figure1.jpg)

위의 두 번째 그림은 연구에서 사용한 DQN의 구조이다. DQN은 4개의 Conv layer, 1개의 FC layer, 그리고 Q-value output layer로 구성되어 있다. 

앞으로 실험 결과 설명에 앞서서, 연구 디자인에 대해서 간단하게 설명하겠다. 총 6명의 참가자(이하 sub001 ~ sub006)가 존재하며, 이들이 3종류의 게임을 플레이하는 동시에 fMRI데이터를 측정하였다. 이외에 사용한 자원과 모델은 뒤에서 설명하도록 하겠다.

## Figure 2. Basic concepts

![figure 1. atari game setup and DQN](images\2025-03-01-Using deep reinforcement learning to reveal how the brain encodes abstract state-space representations in high-dimensional environments\figure2.jpg)


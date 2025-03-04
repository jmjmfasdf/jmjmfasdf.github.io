---
layout: post
title: "Neural computations underlying arbitration between model-based and model-free learning"
author: jeongmin seo
tags: [Arbitration, fMRI, Reinforcement Learning]
categories: Paper Review
color: rgb(25, 25, 112)
feature-img:
thumbnail: "images/2025-03-04-Neural computations underlying arbitration between model-based and model-free learning/figure1.jpg"
---

우리의 뇌에는 행동 선택을 조절하는 두 가지 시스템, model-based와 model-free 시스템이 존재한다는 신경학적인 증거가 존재한다. 하지만 특정한 순간에, 어떤 시스템이 행동을 주도하는지에 대한 메커니즘은 정확하게 밝혀진 바가 없다. 본 논문에서는 두 모델 사이의 중재(arbitration) 메커니즘이 존재하며, 각 시스템의 신뢰도(reliability)에 따라 행동 제어 **비율**을 할당한다는 증거를 제시하고 있다. 이러한 arbitration system은 model-based system의 제어 정도에 따라 model-free system과의 connectivity를 negative direction으로 조절하는 것으로 밝혀졌는데, 즉 arbitrator가 model-based system을 신뢰할 경우 model-free system의 영향력을 감소시키는 방향으로 행동을 조절할 수 있다는 것이다.

## Introduction

개체의 행동을 조절하는 두 가지 경쟁적 시스템이 존재하는 것은 오래 전부터 알려진 사실이지만 **두 시스템 간 제어가 어떻게 전환되는지에 대한 연구는 부족하다.** 두 시스템은 goal-directed, habitual으로, 각각 prefrontal cortex, anterior striatum/posterior lateral striatum과 연관되어 있다고 알려져 있다. 가령 약물 중독같은 경우 goal-directed system과 habitual system 같의 균형이 무너져 **약물과 관련된 stimulus-response habit이 억제되지 못해** 일어나는 현상일 수 있다는 것이다. 저자들은 RL framework를 활용해서 goal-directed system의 경우 model-based RL(환경에 대한 내부 모델을 활용해 행동의 가치를 실시간으로 계산), habitual system의 경우 model-free RL(시행착오를 통해 행동의 보상을 학습)을 활용하여 디자인하고, 인간의 뇌에서 arbitration process가 어떻게 작동하는지 기제를 규명하는 것을 목표로 한다.

## Computational model of arbitration

본 논문에서 제안하는 arbitration 시스템은 세 가지 layer 1)model-based/model-free RL, 2)reliability estimation, 3)reliability competition 으로 구성된다. 첫 번째 단계에서는 MB, MF agent가 각각 state-prediction error(SPE), reward-prediction error(RPE)를 계산하여 학습한다. 이후 각 시스템이 환경을 얼마나 정확하게 예측하는지 계산한다. MB system의 SPE의 경우 값이 0에 가까울수록 정확하며, MF system의 RPE의 경우 값이 작을수록 보상 예측이 정확하다. MB system의 경우 bayesian framework를 이용하여 추론하나, MF system의 경우 RPE를 추정하기 위해 더 단순한 시스템을 사용할 수도 있다. 이후 두 신뢰도 지표가 경쟁하여 PMB(MB system이 선택될 확률)를 결정하며, PMB가 높으면 MB system이 행동을 주도하고 낮으면 그 반대가 되는 방식이다. 두 시스템 간 제어는 이분법적이 아닌 동적 가중치로 조절된다.

## Markov Decision Task

저자는 arbitration model의 행동 조절 메커니즘을 검증하기 위해서 decision task를 설계하였다. 이 실험은 MDP를 기반으로 하며 공통적인 컨셉은 **참가자는 binary choice를 통해 토큰을 선택하며, 이 토큰은 금전적 보상으로 교환이 가능**하다는 것이다.

#### 실험 디자인 1. 조건

이 실험은 두 가지의 조건으로 나뉜다. 즉 서로 다른 통제 조건의 실험 디자인이라고 할 수 있다. 하나는 특정 목표 조건, 다른 하나는 유연한 목표 조건이다. 특정 목표 조건에서는 한 가지 색상의 토큰만 보상이 주어지며, 다른 토큰은 금전적으로 무효한 것이 된다. 보상이 가능한 색상은 매 라운드마다 바뀌며, 특정 색상의 토큰만 유효하기 때문에 실험은 환경의 변화를 실시간으로 고려해야 하는 참가자를 MB system으로 행동하게끔 유도한다. 이 통제 조건에서는 MF system의 RPE가 높아지는 경향성을 보인다.

다른 디자인인 유연한 목표 조건에서는 어떤 색상 토큰이라도 금전적 보상을 취할 수 있으며, 참가자에게 MF system으로의 의존을 유도한다. 특정 목표가 변하지 않기 때문에 **경험에 의존하는 습관적 행동**을 유도하는 것이 핵심이다.

{% include aligner.html images="images/2025-03-04-Neural computations underlying arbitration between model-based and model-free learning/figure1.jpg" caption="figure 1. task design" %}

#### 실험 디자인 2. 전이 확률

실험에서는 상태 전이 확률을 조작하여 시스템간의 전환을 유도한다. 예를 들어 전이 확률이 0.5라면 어떤 선택지를 선택해도 상태가 바뀔 확률은 반반이다. 즉 참가자로 하여금 예측이 어려운 상황을 유도하며, MF system이 유리하게 된다. 반대로 전이 확률이 0.1이라면 예측이 쉽기 때문에 MB system이 유리하다.

실험 과정 동안 총 22명의 성인 참가자가 참여하였으며, 실험동안 fMRI로 뇌 활동을 측정하였다.




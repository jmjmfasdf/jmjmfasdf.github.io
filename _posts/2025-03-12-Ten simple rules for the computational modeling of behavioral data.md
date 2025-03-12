---
title: "Ten simple rules for the computational modeling of behavioral data"
date: 2025-03-12
tags:
    - Modeling
    - fMRI
    - EEG
    - Behavioral Data
categories: 
    - Paper Review
toc: true
toc_sticky: true
---

컴퓨테이셔널 모델링(computational modeling)은 심리학과 신경과학 연구에서 중요한 혁신을 가져왔다. 실험 데이터를 모델에 적합하게 피팅(fitting)하면 행동의 근본적인 알고리즘을 탐구하고, 계산적 변수의 신경학적 상관(neural correlates)을 찾을 수 있으며, 약물, 질병, 중재(intervention)의 영향을 더 깊이 이해할 수 있다. 본 논문에서는 컴퓨테이셔널 모델링을 신중하게 사용하고 의미 있는 통찰을 얻기 위한 10가지 간단한 규칙을 제시한다. 특히, 초보 연구자들이 쉽게 접근할 수 있도록 모델과 데이터를 연결하는 방법을 실용적이고 세부적인 관점에서 설명하고 있다.

**"모델이 우리에게 정신(mind)에 대해 정확히 무엇을 말해줄 수 있는가?"**라는 핵심 질문을 다루기 위해, 저자들은 가장 기본적인 모델링 기법을 중심으로 설명하며, 실험 코드와 예제를 통해 개념을 구체적으로 보여준다. 그러나 이 논문에서 제시하는 대부분의 규칙은 더 발전된 모델링 기법에도 적용할 수 있다.

# 1. What is computational modeling of behavioral data?

행동 데이터의 컴퓨테이셔널 모델링(computational modeling of behavioral data)이란 수학적 모델을 이용해 행동 데이터를 보다 체계적으로 이해하는 방법을 의미한다. 행동 데이터는 주로 **선택(choice)** 형태로 나타나지만, **반응 시간(reaction time), 시선 움직임(eye movement), 신경 활동(neural data)** 등의 다양한 관찰 가능한 변수들도 포함될 수 있다. 이 모델들은 실험에서 관찰할 수 있는 자극(stimuli), 결과(outcomes), 과거 경험(past experiences) 등의 변수와 미래 행동을 수학적으로 연결하는 방정식의 형태를 갖는다. 즉, 컴퓨테이셔널 모델은 행동이 생성되는 과정에 대한 ‘알고리즘적 가설(algorithmic hypothesis)’을 구체적으로 구현하는 역할을 한다.

행동 데이터를 '이해한다'는 의미는 연구자의 목표에 따라 다를 수 있다. 어떤 경우에는 데이터를 설명할 수 있는 단순한 모델로도 충분하지만, 보다 정량적인 예측을 제공하는 복잡한 모델이 필요한 경우도 있다. 연구자들이 컴퓨테이셔널 모델을 사용하는 주요 목적은 크게 네 가지로 나눌 수 있다.

**시뮬레이션(Simulation)**

모델의 특정한 매개변수(parameter) 설정하에 가상의 ‘행동 데이터’를 생성하는 과정이다. 이러한 시뮬레이션 데이터를 실제 데이터처럼 분석하면, 행동의 질적 및 양적 패턴을 예측하고 검증할 수 있다. 즉, 시뮬레이션을 통해 이론적 예측을 보다 정밀하게 만들고 실험적으로 검증할 수 있도록 한다.

**매개변수 추정(Parameter Estimation)**

주어진 모델이 실험 데이터를 가장 잘 설명할 수 있도록 매개변수 값을 찾는 과정이다. 이렇게 얻어진 매개변수는 데이터의 요약 정보로 활용될 수 있으며, 개인차(individual differences)를 연구하거나 약물, 병리적 상태, 실험적 조작(intervention) 등의 영향을 정량화하는 데 사용될 수 있다.

**모델 비교(Model Comparison)**

여러 개의 가능한 모델 중 어떤 모델이 행동 데이터를 가장 잘 설명하는지를 평가하는 과정이다. 특히, 서로 유사한 질적 예측을 하지만 정량적 차이가 있는 모델들을 비교할 때 유용하다. 모델 비교를 통해 행동을 생성하는 기저 메커니즘을 더 잘 이해할 수 있다.

**잠재 변수 추론(Latent Variable Inference)**

행동 데이터에서 직접 관찰할 수 없는 숨겨진 변수(latent variable)의 값을 추정하는 과정이다. 예를 들어, 어떤 선택이 주어진 상황에서 얼마나 가치가 있는지(value of choices)와 같은 정보는 직접 측정할 수 없지만, 모델을 통해 추론할 수 있다. 이러한 기법은 특히 신경영상(neuroimaging) 연구에서 활용되며, EEG, ECOG, 전기생리학(electrophysiology), 동공 측정(pupillometry) 등의 다양한 데이터와 결합되어 신경 메커니즘을 밝히는 데 기여한다.

<br>

컴퓨테이셔널 모델링은 강력한 도구이지만, 잘못 사용될 경우 잘못된 결론을 도출할 위험이 있다. 시뮬레이션, 매개변수 추정, 모델 비교, 잠재 변수 추론 각각은 특정한 강점과 약점을 가지고 있으며, 부주의하게 다루면 오해를 유발할 수 있다. 따라서 논문에서는 초보자도 이해할 수 있도록 실용적이고 세부적인 접근 방식을 제시하고, 모델을 데이터와 어떻게 연결해야 하는지, 그리고 모델링 과정에서 발생할 수 있는 일반적인 실수를 어떻게 피할 수 있는지를 설명하고자 한다.

이 논문의 목표는 단순히 모델을 구현하는 기술적 측면을 다루는 것이 아니다. 대신, 모델이 인간의 인지 과정과 행동을 어떻게 설명하는지를 보다 심층적으로 탐구하는 데 중점을 둔다. 이를 위해 가장 기초적인 모델링 기법을 중심으로 설명하지만, 논문에서 제시하는 원칙들은 보다 복잡한 모델에도 적용될 수 있다. 또한, 보다 심화된 모델링 기법을 다루는 다양한 튜토리얼, 예제, 그리고 교재들을 참고할 것을 권장한다.

논문에서는 설명의 명확성을 위해 강화학습(reinforcement learning) 모델을 선택하고, 이를 선택 행동(choice data)에 적용하는 예제를 중심으로 설명한다. 이 특정한 도메인을 선택한 이유는 다음과 같다.

**1. 강화학습 모델은 학습 과정에 대한 연구에서 특히 인기가 많다.**

행동 데이터에서 학습의 특성을 분석할 때, 강화학습 모델은 중요한 도구가 될 수 있다. 특히, 행동 데이터에서 개별적인 시도(trial)는 모든 과거 경험에 영향을 받으며, 이는 고전적인 조건별 데이터 분석(aggregation across conditions)을 어렵게 만든다. 따라서, 컴퓨테이셔널 모델링이 이러한 데이터의 특성을 포착하는 데 유리하다.

**2. 학습 과정에서의 연속적 의존성(sequential dependency)은 모델 피팅(fitting) 과정에서 독특한 기술적 문제를 유발한다.**

비학습(non-learning) 과제에서는 존재하지 않는 이러한 문제를 다루는 방법을 학습할 필요가 있다.

이 논문에서 다루는 모델링 기법들은 강화학습뿐만 아니라 다른 행동 데이터에도 광범위하게 적용될 수 있다. 예를 들어,

- 반응 시간(reaction time) 모델링 (Ratcliff & Rouder, 1998; Viejo et al., 2015)  
- 지각(perception) 및 지각적 의사결정(perceptual decision-making) (Sims, 2018; Drugowitsch et al., 2016)  
- 경제적 의사결정(economic decision-making) (van Ravenzwaaij et al., 2011; Nilsson et al., 2011)  
- 단기 기억(visual short-term memory) (Donkin et al., 2016)  
- 장기 기억(long-term memory) (Batchelder & Riefer, 1990)  
- 범주 학습(category learning) (Lee & Webb, 2005)  
- 집행 기능(executive functions) (Haaf & Rouder, 2017)  

등 다양한 분야에서도 동일한 모델링 원칙이 적용될 수 있다.


<figure class='align-center'>
    <img src = "image path" alt="">
    <figcaption>figure 1. caption</figcaption>
</figure>

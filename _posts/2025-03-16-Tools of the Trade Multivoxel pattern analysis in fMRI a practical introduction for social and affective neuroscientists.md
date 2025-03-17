---
title: "Tools of the Trade Multivoxel pattern analysis in fMRI: a practical  introduction for social and affective neuroscientists"
date: 2025-03-16
tags:
    - fMRI
    - Machine Learning
    - MVPA
    - RSA
categories: 
    - Paper Review
toc: true
toc_sticky:  true
---

기능적 자기공명영상(fMRI) 데이터는 뇌의 각 지점(즉, 복셀)에서 시간별 혈중 산소 농도 의존 신호(BOLD signal) 값을 제공한다. 전통적인 단변량(univariate) 분석에서는 개별 복셀을 독립적으로 분석하거나 특정 뇌 영역 내 복셀의 신호를 평균하여 비교한다. 반면, 다중복셀 패턴 분석(MVPA)은 이러한 개별 복셀 값을 고려하는 것이 아니라, 여러 복셀에 걸친 신경 반응 패턴을 분석하여 정보를 추출한다.

MVPA는 다양한 방식으로 수행될 수 있으나, 가장 널리 사용되는 두 가지 분석 기법은 디코딩 분석(decoding analysis) 과 표상 유사성 분석(Representational Similarity Analysis, RSA) 이다. 디코딩 분석은 특정 조건에서 나타나는 신경 반응 패턴을 기반으로 조건을 예측하는 것을 목표로 하며, RSA는 서로 다른 조건 간 신경 반응 패턴의 유사성을 평가한다.

<br>

# Introduction

이 논문은 다중복셀 패턴 분석(Multivoxel Pattern Analysis, MVPA) 기법을 사회 및 감정 신경과학(Social and Affective Neuroscience) 연구자들에게 실용적이고 이해하기 쉬운 방식으로 소개하는 것을 목표로 한다. 기존의 fMRI 데이터 분석에서는 단변량(univariate) 또는 질량 단변량(mass-univariate) 접근법이 일반적으로 사용되었다. 이러한 방법들은 특정 실험 조건에서 개별 복셀(voxel)이나 특정 뇌 영역의 평균 신호 강도를 분석하는 방식이다. 예를 들어, 공포 유발 자극(fear-inducing stimuli) 이 중립 자극보다 편도체(amygdala) 의 활성화를 더 크게 유발하는지를 평가하는 것이 대표적인 단변량 접근법이다.

단변량 분석의 특징은 각 조건당 단 하나의 값을 고려한다는 점이다. 즉, 한 실험 조건에서 특정 복셀 또는 뇌 영역의 평균 신호 강도만 분석 대상으로 삼는다. 이 방법은 신경 활성의 전반적인 강도 변화를 평가하는 데 적절하지만, 다중 복셀에 걸친 패턴 정보는 포착하지 못하는 한계가 있다. 반면, 최근 들어 연구자들은 개별 복셀이 아니라 여러 복셀에 걸친 신경 반응 패턴(response patterns) 을 분석하는 MVPA 기법을 점점 더 많이 활용하고 있다. MVPA는 특정 조건에서 뇌가 정보를 어떻게 표상하는지를 이해하는 데 도움을 주며, 단순한 활성화 정도뿐만 아니라 공간적 패턴(Spatial Pattern)과 정보 내용(Information Content) 을 분석할 수 있다는 강점이 있다.

<br>

# What is MVPA?

기능적 자기공명영상(fMRI) 데이터는 뇌의 각 지점(복셀, voxel)에서 매 순간(시간 반복, TR)마다 혈중 산소 농도 의존 신호(BOLD signal) 값을 기록한다. 전통적인 단변량(univariate) 분석은 이러한 데이터를 개별 복셀 단위로 분석하거나, 특정 뇌 영역 내 복셀들의 신호를 평균 내어 비교하는 방식이다. 하지만 이러한 접근법은 개별 복셀의 신호 크기만을 고려할 뿐, 복셀 간의 공간적 관계나 신호 패턴을 반영하지 못한다는 한계를 가진다.

MVPA(Multivoxel Pattern Analysis, 다중복셀 패턴 분석) 는 단변량 분석과 달리, 단순히 복셀별 신호 강도를 평가하는 것이 아니라 복셀들 사이의 반응 패턴(response pattern) 에서 정보를 찾아낸다. 즉, 특정 조건에서 뇌의 공간적 반응 패턴을 통해 어떤 정보가 표상되고 있는지를 분석하는 방법이다.

## Decoding analyses

디코딩 분석은 특정 신경 반응 패턴을 기반으로 어떤 조건(자극이나 인지 상태 등)이 이를 유발했는지를 예측하는 분석 기법이다. 이는 전통적인 단변량 분석과는 추론의 방향이 반대라는 점에서 차이가 있다. 전통적인 단변량 분석은 특정 조건에서 뇌의 특정 영역이 활성화되는지 여부를 평가하는 방식으로, **$$P(\text{뇌 활성} \mid \text{조건})$$** 을 분석한다. 반면, 디코딩 분석은 주어진 신경 반응 패턴이 어떤 조건에서 유발되었는지를 추론하는 것으로, **$$P(\text{조건} \mid \text{뇌 활성})$$** 을 분석하는 접근법이다.

디코딩 분석에는 크게 두 가지 주요 방법이 있다. 첫 번째는 **분류(Classification)** 로, 신경 반응 패턴을 바탕으로 자극이 특정 범주(category)에 속하는지를 예측하는 방법이다. 예를 들어, 특정 신경 반응이 분노한 얼굴(angry face) 과 놀란 얼굴(surprised face) 중 어느 자극에서 유발되었는지를 예측할 수 있다. 주로 서포트 벡터 머신(SVM), 로지스틱 회귀(Logistic Regression) 등의 기계 학습 알고리즘이 사용된다. 두 번째 방법은 **회귀(Regression)** 로, 신경 반응 패턴을 바탕으로 연속적인 값(continuous value)을 예측하는 방식이다. 예를 들어, 주어진 신경 반응이 유발된 얼굴이 얼마나 화가 난 얼굴인지(0~100의 연속 값으로 평가)를 예측할 수 있다. 선형 회귀(Linear Regression), 랜덤 포레스트 회귀(Random Forest Regression)와 같은 알고리즘이 주로 활용된다. 디코딩 분석의 과정은 데이터 분할-학습-평가-결과 해석의 절차를 따르며, 성능은 Accuracy, Specificity, Sensitivity 등의 지표를 사용할 수 있다.

디코딩 분석은 전통적인 단변량 분석이 간과할 수 있는 정보를 효과적으로 탐지할 수 있다. Haxby et al. (2001) 의 연구는 이를 대표적으로 보여주는 사례다. 연구 참가자들에게 얼굴, 집, 의자, 신발 등의 다양한 시각 자극을 제시한 후 fMRI 데이터를 수집한 결과, 전통적인 단변량 분석으로는 특정 범주의 자극(예: 의자, 신발, 집)에 대해 명확한 신경 활성 차이를 확인할 수 없었다. 그러나 MVPA를 적용한 디코딩 분석을 수행한 결과, 복셀 패턴을 통해 각 범주를 명확히 구별할 수 있음이 밝혀졌다. 이러한 연구 결과는 이후 MVPA를 활용한 다양한 디코딩 연구로 확장되었으며, 인간이 꿈을 꾸는 동안 나타나는 신경 패턴(Horikawa et al., 2013), 듣고 있는 소리(Giordano et al., 2013), 인식하는 얼굴(Goesaert & Op de Beeck, 2013), 상상하는 인물(Hassabis et al., 2014) 등을 해독하는 연구로 이어졌다.

## Representational similarity analysis

표상 유사성 분석(RSA)은 특정 자극이 유발한 신경 반응 패턴의 유사성을 분석하는 기법이다. 이는 단순히 개별 신경 반응을 비교하는 것이 아니라, 자극 간의 상대적인 관계를 분석하는 데 초점을 맞춘다. 즉, RSA는 특정 신경 반응 패턴이 나타나는지를 평가하는 것이 아니라, 서로 다른 자극들이 신경 반응 패턴에서 어떻게 구별되는지를 측정하는 방식이다.

### First- vs. Second-order Isomorphisms

대부분의 신경과학 연구는 일차적 동형성(first-order isomorphism) 에 초점을 맞춘다. 이는 특정 자극이 특정한 신경 반응을 직접적으로 유발하는 관계를 의미한다. 예를 들어, 얼굴을 보면 방추회 얼굴 영역(Fusiform Face Area, FFA) 이 집을 볼 때보다 더 활발하게 반응하는 경우가 이에 해당한다. 이러한 접근법은 단순한 신경 활성화 패턴과 자극 간의 관계를 연구하는 전통적인 방식이다. 반면, RSA는 이차적 동형성(second-order isomorphism) 을 분석한다. 이는 개별 신경 반응 패턴이 아니라 자극들 간의 유사성과 차이를 비교하는 것에 초점을 맞춘다. 

예를 들어 광고판에 있는 커다란 얼굴 사진과 종이에 그린 작은 웃는 얼굴을 비교해 보자. 두 이미지의 세부적인 얼굴 특징(눈, 코, 입의 모양 등)은 매우 다를 수 있다(즉, 일차적 동형성이 없음). 그러나 눈은 코보다 가깝고, 코는 입보다 위에 위치하는 등 **얼굴 구성 요소들 간의 관계는 두 이미지에서 동일하다.** 이것이 바로 이차적 동형성이다. 즉, 개별 요소들이 어떻게 배열되는지가 아니라 자극들 간의 상대적인 관계가 중요한 것이다.

<figure class='align-center'>
    <img src = "/images/2025-03-16-Tools of the Trade Multivoxel pattern analysis in fMRI a practical introduction for social and affective neuroscientists/figure2.jpg" alt="">
    <figcaption>Fig 2. First- and second-order isomorphisms.</figcaption>
</figure>

같은 방식으로, 특정 뇌 영역이 어떤 속성을 인코딩한다면, 해당 속성과 신경 반응 크기 간의 직접적인 상관관계는 보이지 않을 수 있으나 자극들 간의 유사성(예: 두 사람의 얼굴은 서로 유사하지만, 기린의 얼굴과는 다름)과 신경 반응 패턴의 유사성이 일치한다면, 그 뇌 영역이 해당 속성을 처리하고 있을 가능성이 높다고 볼 수 있다. RSA는 이러한 원리를 활용하여, **신경 반응 패턴 간의 유사성**을 비교함으로써 자극 간의 상대적 관계를 분석한다. 즉, 두 개의 서로 다른 사람이 동일한 자극을 보았을 때, **개별 신경 반응 패턴 자체는 다를 수 있지만(Figure 2B), 그 신경 반응들 간의 관계는 일정하게 유지될 수 있다.(Figure 2D)** 이와 같이 RSA는 개인 간 비교, 다른 측정 방식 간 비교, 모델과 신경 반응 간 비교 등을 가능하게 한다.

### Comparison Across Modalities

RSA에서 가장 중요한 도구는 표상 비유사성 행렬(Representational Dissimilarity Matrix, RDM) 이다. 이는 각 자극이 유발한 신경 반응 패턴 간의 유사성을 수량화한 행렬이다. 즉, 특정 뇌 영역이 두 개의 자극을 얼마나 다르게 처리하는지를 측정하는 역할을 한다.(Figure 2C) 신경 RDM 외에도 여러 종류의 RDM을 만들 수 있다. 예를 들어,

- 행동 데이터 기반 RDM: 참가자들이 평가한 두 자극 간의 유사성(예: 두 얼굴이 얼마나 비슷한가?)을 바탕으로 생성.  
- 객관적 속성 기반 RDM: 이미지의 픽셀 차이, 색상 차이, 구조적 차이를 바탕으로 생성.  
- 신경망 모델 기반 RDM: 딥러닝 모델이 예측한 자극 간 유사도를 바탕으로 생성.  
  
이러한 다양한 RDM은 신경 RDM과 비교하여 특정 뇌 영역이 어떤 속성을 중요하게 처리하는지를 분석하는 데 사용된다. 예를 들어, 특정 뇌 영역의 신경 RDM이 행동 데이터 기반 RDM과 높은 상관관계를 보인다면, 해당 뇌 영역은 사람들이 주관적으로 판단하는 자극 간의 유사성을 반영하고 있을 가능성이 높다.

### Data-driven exploration of representational structure.

표상 유사성 분석(Representational Similarity Analysis, RSA)은 단순히 자극 간의 신경 반응 패턴을 비교하는 것이 아니라, 뇌가 정보를 어떻게 구조화하는지(data-driven way) 를 탐색하는 데에도 활용될 수 있다. 이를 위해 다차원 척도법(Multidimensional Scaling, MDS), 군집 분석(Clustering), 차원 축소(Dimensionality Reduction) 와 같은 데이터 분석 기법이 사용된다.

예를 들어, 연구자가 특정 뇌 영역에서 얼굴이 어떻게 표상되는지를 알고 싶다고 가정해 보자. 단순히 얼굴의 감정 표현(예: 행복, 슬픔, 분노)을 기준으로 신경 반응을 비교하는 대신, MDS를 사용하여 데이터가 자연스럽게 형성하는 군집을 시각화할 수 있다. 이를 통해, 얼굴이 감정 표현이 아니라 연령(age)에 따라 클러스터링(clustering) 되는 패턴을 발견할 수도 있다. 만약 연구자가 감정 표현만을 기준으로 분석했다면, 이러한 정보를 놓쳤을 가능성이 크다.

이는 사전에 특정 가설을 강제하지 않고 데이터가 나타내는 자연스러운 구조를 탐색할 수 있다는 점에서 중요한 장점이 있다. 즉, 데이터가 특정 속성(예: 연령, 성별, 인종 등)에 따라 구조화될 가능성을 연구자가 사전에 예측하지 않아도, RSA를 통해 자연스럽게 나타나는 구조를 발견할 수 있다.

### Comparison across individuals

표상 유사성 분석(Representational Similarity Analysis, RSA)은 개별 참가자의 신경 데이터를 비교하여 특정 자극이 유발하는 신경 반응 패턴이 개인마다 어떻게 다르거나 유사한지를 분석하는 데 활용될 수 있다. 특히, RSA는 개별 참가자의 신경 반응 패턴 자체가 아니라, 자극 간 신경 반응 패턴의 유사성(Relational Similarity)을 비교하기 때문에 개인 간 차이에 덜 민감하다는 장점이 있다.

fMRI 연구에서는 일반적으로 참가자들의 데이터를 공통된 해부학적 좌표계(Talairach space, MNI space 등)로 변환하여 비교한다. 그러나 이러한 공간 정렬(spatial alignment) 방식은 개별 참가자 간의 미세한(fine-scale) 신경 반응 패턴 차이를 제대로 정렬하지 못하는 경우가 많다. 예를 들어, 방추회 얼굴 영역(Fusiform Face Area, FFA) 은 모든 참가자에서 비슷한 위치에 존재하지만, 세부적인 신경 반응 패턴은 개인마다 다를 수 있다. 이는 뇌의 대규모 기능적 구조(coarse-scale functional organization) 는 비교적 일정한 반면, 세부적인 신경 패턴(fine-scale spatial pattern) 은 개인마다 상당한 차이가 있기 때문이다. 이는 RSA를 이용하면 개별 참가자 간 신경 반응 패턴이 다를지라도, 자극 간의 신경 반응 관계(RDM, Representational Dissimilarity Matrix)가 유사하면, 두 사람이 동일한 방식으로 자극을 표상(represent)하고 있다고 해석할 수 있다. 

RSA는 신경 반응 패턴 자체가 아니라, 신경 반응 간의 관계를 비교하기 때문에 개별 참가자의 신경 반응 차이에 덜 민감하다. 다시 말해, 신경 반응 패턴 자체는 다르더라도, 동일한 자극을 받았을 때의 신경 반응 관계가 유지된다면, 두 사람이 자극을 비슷하게 표상하고 있다고 해석할 수 있다.

이러한 접근법은 개인 간 신경 데이터의 정렬(alignment) 문제가 있는 경우에도 신뢰할 수 있는 비교를 가능하게 한다. 예를 들어, 딥러닝 모델이 두 개의 신경 데이터셋을 비교하려고 할 때, 직접적인 신경 반응 패턴을 정렬하는 것은 어렵지만, 신경 반응 패턴의 관계(RDM)를 비교하면 보다 일관된 결과를 얻을 수 있다.

## Spatially mapping effects

신경영상 연구에서 중요한 목표 중 하나는 어떤 신경 반응이 뇌의 어느 위치에서 발생하는지를 파악하는 것이다. 이는 크게 두 가지 방법을 통해 수행될 수 있다. **영역 기반 분석** (Region of Interest, ROI-based analysis, 연구자가 사전에 특정 뇌 영역을 정의한 후, 해당 영역에서 신경 활성 패턴을 분석하는 방법이다.), **점별 분석** (Point-by-point analysis, 연구자가 사전 정의한 특정 영역이 아니라, 뇌 전체에서 개별 복셀(voxel)을 기준으로 분석하는 방법이다. 단변량 분석에서는 복셀 단위(voxel-wise) 로 분석하며, MVPA에서는 서치라이트 분석(searchlight analysis) 을 사용한다.)이다. 

<figure class='align-center'>
    <img src = "/images/2025-03-16-Tools of the Trade Multivoxel pattern analysis in fMRI a practical introduction for social and affective neuroscientists/figure1.jpg" alt="">
    <figcaption>Fig. 1. Comparing data in univariate analyses and MVPA.</figcaption>
</figure>

서치라이트 분석은 MVPA에서 공간적 매핑(spatial mapping) 을 수행하는 중요한 기법이다. 이는 뇌의 특정 위치에서 신경 반응 패턴이 실험 조건 간 차이를 보이는지 평가하는 방법이며, 기본적인 원리는 다음과 같다. 각 복셀을 중심으로 일정 반경(보통 8–12mm)의 구(sphere)를 설정하고, 해당 구 내의 복셀들을 이용해 MVPA를 수행하여 조건 간 차이를 평가한다. 이후 분석 결과를 구의 중심 복셀에 할당하고 이 과정을 모든 복셀에 반복하여, 뇌 전체에서 어떤 영역이 특정 조건을 구별하는 정보를 포함하는지 확인한다. 이렇게 얻어진 결과는 각 복셀을 중심으로 한 구(sphere)의 정보 판별력(distinctiveness)에 대한 맵을 생성한다. 연구자는 이 맵을 통해 특정 조건을 구별하는 데 중요한 뇌 영역이 어디인지 파악할 수 있다.

다만 서치라이트 분석은 복셀 간 공간적 관계를 반영할 수 있지만, 개별 복셀의 정보가 아니라 주변 복셀들의 패턴 차이에 의해 결과가 결정된다는 점을 고려해야 한다. 따라서 결과 해석 시, 단순히 특정 복셀에서 유의미한 차이가 나타났다고 해석하기보다는, 해당 복셀을 중심으로 한 구 내의 신경 패턴 차이가 특정 조건을 구별하는지를 평가하는 방식임을 이해하는 것이 중요하다.

## Benefits of MVPA

다중복셀 패턴 분석(MVPA)은 뇌의 특정 영역이 정보를 처리하는 방식을 더 깊이 이해할 수 있도록 하는 강력한 도구이다. 단순히 뇌의 특정 영역에서 반응 강도(response magnitude) 가 증가했는지를 분석하는 단변량(univariate) 분석과 달리, MVPA는 복셀(voxel) 간의 신경 반응 패턴을 분석하여 정보 처리를 탐색한다. 이를 통해 단순한 활성화 강도를 넘어, 뇌가 특정 자극을 어떻게 구별하는지를 밝힐 수 있다.

### 분포된 신경 반응 패턴에서 정보를 추출하는 민감성 (Sensitivity to Information in Distributed Response Patterns)

MVPA는 개별 복셀의 활성 강도가 아니라, 복셀 간의 분포된 신경 반응 패턴에서 정보를 추출하는 방식이다. 단변량 분석에서는 개별 복셀의 신호 변화가 통계적으로 유의하지 않으면 해당 복셀은 분석에서 제외되지만, MVPA는 이러한 복셀을 포함하여 전체적인 신경 반응 패턴을 분석한다. 이로 인해, 신경 반응이 미세하게 변화하는 영역에서도 특정 조건 간의 차이를 감지할 수 있다.

뉴런 활동을 직접 측정한 연구에 따르면, 뇌는 다양한 유형의 정보를 뉴런 집단 코드(distributed neuronal population code) 로 인코딩한다. 감각 정보(Uchida et al., 2000), 운동 계획(Georgopoulos et al., 1988), 고차원 범주 정보(Kiani et al., 2007), 주관적 의사 결정(Kiani et al., 2014) 등 여러 인지 기능이 이러한 방식으로 처리된다.

비록 fMRI가 개별 뉴런 활동을 직접 측정하는 것은 아니지만, MVPA를 통해 다중복셀 패턴을 분석하면 뉴런 집단 코드에서 다루는 정보와 유사한 수준의 정보를 추출할 수 있음이 밝혀졌다(Kriegeskorte et al., 2008b; Dubois et al., 2015). 최근 연구에서는 디코딩 분석을 활용하여 특정 조건을 잘 구별하는 신경 반응 패턴을 학습한 후, 이 패턴을 RSA와 비교하는 '혼합 RSA(mixed RSA)' 기법이 개발되었다. 이는 특히 저차원 시각 정보 처리 영역에서 적용되었으나, 향후 사회적 및 감정적 정보 처리 연구에서도 활용될 가능성이 크다(Khaligh-Razavi et al., 2017).

### 특정 뇌 영역이 정보를 조직하는 방식 분석 (Uncovering the Information Content of Brain Regions)

MVPA는 특정 뇌 영역이 다양한 자극을 어떻게 조직하고 구별하는지를 탐색할 수 있는 강력한 도구이다. 예를 들어, Peelen et al. (2010)의 연구에서는 내측 전전두엽(mPFC)과 좌측 상측두구(left superior temporal sulcus, STS) 이 얼굴, 몸짓, 목소리와 같은 다양한 감정 단서에 동일하게 반응하지만, 감정이 무엇인지(분노, 혐오, 두려움, 행복, 슬픔)를 구별하는 방식은 다중복셀 패턴에 의해 반영됨을 발견했다. 이는 STS와 mPFC가 감정을 특정한 감각 방식(얼굴, 목소리, 몸짓 등)이 아니라, 추상적인 감정적 가치(abstract emotional value)에 따라 조직하고 있음을 시사한다.

또한, MVPA는 신경 반응 패턴이 자극의 다양한 특성을 어떻게 구별하는지를 분석하여 정보가 뇌에서 어떻게 변환되는지를 탐색할 수 있다. 예를 들어, 초기 감각 피질에서 저차원 시각적 속성(예: 색상, 모양)을 처리하는 방식과, 이후 단계에서 고차원적인 의미적 범주(예: 감정, 사회적 의미)를 인코딩하는 방식 간 차이를 분석할 수 있다. 이를 통해 정보가 뇌에서 처리되는 여러 단계들을 탐색할 수 있으며, 특정 뇌 영역이 인지 과정에서 수행하는 기능적 역할을 더 정확하게 이해할 수 있다.

### 유사한 활성 패턴이 동일한 인지 과정을 의미하는지 검증 (Testing the Significance of Overlapping Activations Across Tasks, Domains, and Contexts)

MVPA는 동일한 뇌 영역이 서로 다른 유형의 정보(예: 사회적 보상 vs 금전적 보상)를 동일한 방식으로 인코딩하는지 여부를 탐색하는 데 유용하다. 예를 들어, 많은 연구에서 복측 선조체(ventral striatum) 가 사회적 보상(예: 칭찬, 행복한 얼굴)과 비사회적 보상(예: 돈, 주스)에 모두 반응하는 것으로 나타났다(Lin et al., 2012; Bhanji & Delgado, 2014). 하지만, 이 영역이 두 유형의 보상을 동일한 방식으로 처리하는지는 단변량 분석으로는 확인하기 어렵다.

Wake & Izuma (2017)의 연구에서는 MVPA를 활용하여 사회적 보상과 금전적 보상이 복측 선조체에서 유사한 다중복셀 반응 패턴을 유발하는지 분석하였다. 그 결과, 두 보상 유형이 유사한 패턴을 보였으며, 이는 두 보상이 공통적인 신경 기제를 공유할 가능성을 시사한다. 반면, 다른 연구에서는 특정 뇌 영역이 동일한 자극에 반응하더라도, MVPA를 수행하면 서로 다른 신경 반응 패턴을 보이는 경우도 발견되었다(Peelen et al., 2006; Downing et al., 2007).

이러한 결과는 단순히 동일한 뇌 영역이 활성화된다는 것이 반드시 동일한 정보 처리를 의미하는 것은 아니며, 반대로 다른 신경 반응 패턴을 보인다고 해서 두 과정이 전혀 관련이 없는 것도 아님을 시사한다. MVPA를 통해 상대적인 유사성과 차이를 평가함으로써, 특정 뇌 영역이 어떻게 정보를 조직하고 처리하는지를 보다 정밀하게 이해할 수 있다.

또한, MVPA는 공간적으로 정렬된 데이터에서 공통적인 '신경 표식(biomarkers)'을 식별하는 데에도 활용될 수 있다(Wager et al., 2013; Chang et al., 2015; Woo et al., 2017). 예를 들어, 특정 감정 상태(예: 불안, 스트레스)와 관련된 다중복셀 패턴을 찾아 개인을 넘어서 일반화 가능한 신경 지표를 정의하는 것이 가능하다. 이러한 접근법은 임상 신경과학 및 정신의학 연구에서도 점점 더 중요해지고 있다.

## a note on terminology

이 논문에서 다루고 있는 MVPA(Multivoxel Pattern Analysis) 기법들은 fMRI 데이터 분석을 위해 새롭게 개발된 것이 아니며, fMRI 분석에만 국한된 방법도 아니다. MVPA에서 사용하는 분석 기법들은 이미 다양한 학문 분야와 산업에서 널리 활용되는 데이터 분석 기법이며, 신경과학뿐만 아니라 머신러닝, 통계학, 심리학에서도 오랫동안 연구되어 왔다.

MVPA에서 사용하는 디코딩 분석(decoding analysis) 은 일반적으로 (지도형) 기계 학습(supervised machine learning) 또는 통계적 학습(statistical learning) 으로도 불린다(Hastie et al., 2017). 또한 유사성 구조(similarity structure) 분석 은 1960년대부터 심리학자들이 심적 표상(mental representations) 을 연구하는 데 사용해 왔다(Shepard, 1963, 1964; Shepard & Chipman, 1970; Shepard & Cooper, 1992).

그리고 MVPA에서 사용하는 디코딩 및 유사성 분석 기법들은 fMRI 데이터뿐만 아니라, 다양한 유형의 신경영상(neuroimaging) 데이터에도 적용될 수 있다.즉, MVPA는 특정한 분석 기법 자체가 아니라, fMRI 데이터에서 다중복셀 반응 패턴을 분석하는 방법을 의미한다. 같은 데이터 분석 기법이더라도, fMRI뿐만 아니라 다양한 신경영상 데이터에도 적용될 수 있으며, 연구 목적에 따라 다양한 방식으로 활용될 수 있다. 

<br>

# Practical implementation

MVPA를 연구에 적용할 때는 실험 설계와 데이터 분석의 여러 요소를 신중하게 고려해야 한다. 실험 자극을 어떻게 제시할 것인지, 데이터를 어떻게 전처리할 것인지, 어떤 알고리즘을 사용할 것인지, 하이퍼파라미터를 어떻게 조정할 것인지, 어떤 특성을 선택할 것인지 등 여러 결정들이 연구 결과에 큰 영향을 미칠 수 있다. 이러한 결정들은 단순히 분석 기법에 따라 정해지는 것이 아니라, 연구 질문과 실험 패러다임에 따라 달라져야 한다. 따라서 연구자는 자신의 연구 주제를 깊이 이해하고, 적절한 분석 방법을 선택해야 한다.

## Representational spatial scale

### 표상의 공간적 규모 (Representational Spatial Scale)

뇌가 정보를 표상하는 공간적 규모(Spatial Scale)는 실험 설계 및 데이터 분석의 여러 결정에 중요한 영향을 미친다. 연구자가 분석하려는 정보가 세밀한 수준(fine-scale)에서 표상되는지, 혹은 더 넓은 공간적 패턴(coarse-scale)에서 표상되는지를 이해하는 것이 중요하다.

예를 들어서 어떤 정보가 세밀한 공간 패턴에서 인코딩된다면(예: 얼굴 개별성을 구별하는 패턴, Kriegeskorte et al., 2007; Nestor et al., 2011), 참가자 개개인의 뇌 공간에서 분석하는 것이 적절하다. 즉, 데이터를 표준 해부학적 템플릿(MNI, Talairach 등)에 정렬하지 않고, 개별 참가자의 뇌 공간에서 분석하는 것이 바람직하다. 이러한 접근 방식은 공간 정렬 과정에서 미세한 신경 반응 패턴이 왜곡되거나 평균화되는 것을 방지할 수 있다. 따라서 디코딩 분석(Decoding Analysis)과 신경 RDM(Neural Representational Dissimilarity Matrix, Neural RDM) 계산은 개별 참가자의 원래 뇌 공간에서 수행하는 것이 적절하다. 이 경우, 공간적 스무딩(spatial smoothing)을 최소화하거나 수행하지 않는 것이 일반적이다.

반면, 감정 상태와 같은 상대적으로 넓은 영역에서 표상되는 정보를 분석하는 경우에는 더 광범위한 공간 정렬과 데이터 처리 기법을 적용하는 것이 바람직하다. 감정 상태나 사회적 정보 처리는 개별 복셀보다는 여러 뇌 영역의 상대적인 활성도 차이를 반영하는 패턴에서 표상될 가능성이 크기 때문에, 공간적 스무딩을 더 넓은 범위에서 적용하면 이러한 패턴을 보다 안정적으로 추출할 수 있다. 또한, 연구자가 참가자 간 비교(Between-subject analysis)를 수행하려면, 데이터를 표준 해부학적 템플릿(MNI, Talairach 등)에 맞추는 것이 필요하다.

### 실험 설계에서 공간적 규모의 영향

공간적 규모는 단순히 데이터 분석 방법뿐만 아니라, 실험 설계에도 영향을 미친다. 연구자가 분석하려는 정보가 **세밀한 패턴에서 표상된다면, 개별 참가자 내에서 분석할 데이터가 충분히 확보**되어야 한다. 반면, **더 넓은 패턴에서 표상되는 정보라면, 더 많은 참가자를 모집하는 것이 중요**하다. 세밀한 공간적 패턴을 연구할 경우, 개별 참가자가 충분한 반복 실험(trials)을 수행해야 한다. 참가자 간 데이터 비교보다는 참가자 내 데이터 분석을 우선으로 하며, 한 명의 참가자가 수행하는 데이터량을 늘리는 것이 중요하다. 반대로 넓은 공간적 패턴을 연구할 경우, 더 많은 참가자를 모집하는 것이 중요하며, 개별 참가자당 적은 수의 반복 실험(trials)도 가능하다.

## Design considerations

MVPA는 조건 간 신경 활성 패턴이 신뢰할 수 있고 체계적으로 차이가 나는지를 평가하는 분석 기법이므로, 실험 설계가 신경 활성 패턴을 안정적으로 측정할 수 있도록 최적화되어야 한다. 이를 위해 가장 중요한 요소는 잡음(noise)을 최소화하고, 조건 간 균등하게 분포되도록 하는 것이다. 실험 데이터에서 신호 대 잡음 비(signal-to-noise ratio, SNR)를 최대화하는 것이 신뢰할 수 있는 결과를 얻는 핵심이 된다.

### Minimizing noise

#### 1. 충분한 실험 반복 횟수(trials) 확보

MVPA 연구에서는 특정 조건에서 나타나는 신경 활성 패턴을 안정적으로 측정하기 위해, 각 조건에 대한 충분한 실험 반복 횟수(trials)가 필요하다. 단변량 분석에서도 소규모 표본 연구에서 발생하는 잡음을 줄이기 위해 참가자 수를 늘리는 것이 중요하듯, MVPA에서도 각 조건의 샘플 수(trials)를 충분히 확보하는 것이 중요하다.

각 참가자에게 가능한 한 많은 실험 반복을 제공하는 것은 참가자 내 분석(within-subject analysis) 이 수행되는 경우 특히 중요하다. 참가자별로 독립적인 신경 반응 패턴을 학습하는 분석에서는 충분한 반복 실험이 필요하며, 그렇지 않으면 신경 패턴의 안정성이 떨어질 수 있다.

그러나 실험 반복 횟수의 최적 개수는 연구의 신호 대 잡음 비(SNR)에 따라 달라진다. 일반적으로 더 많은 반복이 바람직하지만, 신경 반응이 충분히 모델링될 수 있도록 자극 간 간격(interstimulus interval, ISI)을 확보하는 것이 중요하다. 이러한 이유로, MVPA에서 필요한 반복 횟수를 일반적으로 정해진 기준으로 제안하는 것은 어렵다. 마치 t-검정을 수행할 때, 효과 크기를 모른 채 무조건 20개의 데이터 포인트가 충분하다고 말할 수 없는 것과 같다. 다만, 더 세밀한 차이를 탐색하는 분석일수록 더 많은 반복 실험이 필요 하며, 실험 조건 간 신경 패턴이 명확히 구별될 경우 상대적으로 적은 반복 실험으로도 구분이 가능하다. 

#### 2. 샘플 수와 특성(feature) 수의 균형 유지

MVPA에서 디코딩 분석을 수행할 때는, 샘플 수(반복 실험 횟수, trials)와 특성(feature) 수(보통은 복셀 개수)의 비율이 중요한 역할을 한다. 일반적으로 머신러닝 알고리즘에서 훈련 데이터의 샘플 수는 특성(feature) 수보다 충분히 많아야 한다. Jain & Chandrasekaran (1982)에 따르면, 샘플 수는 특성(feature) 수보다 최소 5~10배 많아야 한다는 것이 일반적인 경험 법칙이다.

그러나 fMRI 연구에서는 현실적으로 이러한 조건을 충족하기 어려운 경우가 많다. 예를 들어, 100개의 복셀을 포함하는 ROI(뇌 영역)를 분석할 경우, 500~1000개의 반복 실험이 필요하다. 하지만, 참가자를 이렇게 오랜 시간 동안 스캔하는 것은 현실적으로 어렵다. 이 문제를 해결하는 방법 중 하나는 가능한 한 많은 반복 실험을 포함하는 것이며, 다른 방법으로는 샘플 수를 유지하면서 특성(feature) 수를 줄이는 것이 있다. 특성 수를 줄이는 방법에는 특성 선택(feature selection), 차원 축소(dimensionality reduction) 등이 있다.

또한, 참가자 간 세밀한 신경 반응 패턴을 일치시키는 하이퍼얼라인먼트(Hyperalignment, Haxby et al., 2011), 또는 다수의 참가자로부터 공통적인 신경 반응 패턴을 추출하는 공유 반응 모델(Shared Response Model, Chen et al., 2015) 등의 방법도 있다. 일부 알고리즘(SVM 등)은 상대적으로 많은 특성(feature)을 포함하더라도 잘 작동하지만, 데이터를 훈련 및 테스트 세트로 나누어야 하는 디코딩 분석에서는 충분한 반복 실험이 필요하다. 반면, RSA(표상 유사성 분석)와 같은 방법은 데이터를 훈련 및 테스트 세트로 나눌 필요가 없으므로 상대적으로 적은 샘플로도 분석이 가능하다.

#### 3. 기타 잡음 완화 전략 (Other Ways to Mitigate Noise)

충분한 반복 실험을 포함하는 것 외에도, 자극과 관련 없는 잡음을 최소화하는 또 다른 방법은 짧은 스캔 세션(runs)을 여러 번 수행하는 것이다. 짧은 세션을 여러 번 수행하면, 실험 세션마다 독립적인 잡음이 발생하기 때문에 여러 세션을 평균내면 신경 RDM(Representational Dissimilarity Matrix)의 신뢰도가 높아지고, 패턴 분류기(classifier)의 성능이 향상될 수 있다(Coutanche & Thompson-Schill, 2012). 또한, 단일 세션 내에서 조건을 비교할 경우 발생할 수 있는 편향(bias) 을 줄이는 데도 도움이 된다(Mumford et al., 2014).

### Even sampling of noise

fMRI 연구에서 발생할 수 있는 중요한 문제 중 하나는 특정 조건과 체계적으로 공변하는 잡음(noise) 이 존재할 가능성이다. 이는 MVPA 분석에서 특히 중요한데, MVPA는 조건 간 신경 반응 패턴을 구별하는 데 민감하기 때문에, 특정 조건과 잡음이 함께 변하면 연구 결과가 왜곡될 가능성이 크다. 잡음의 원인은 외부 요인일 수도 있고 참가자의 행동이나 인지적 차이때문일 수도 있지만 이러한 요인들은 잘못된 결과나 해석 오류를 초래할 수도 있으므로 최대한 통제해야 한다.(Todd et al., 2013). 

MVPA 연구에서는 모든 조건에서 잡음이 균등하게 분포하도록 실험을 설계해야 한다. 실험을 여러 번 반복하는 경우, 특정 조건이 특정 run에만 존재하면 해당 조건이 시행에 의한 잡음과 함께 변할 가능성이 있다. 따라서 모든 시행에서 조건을 균등하게 포함해 신호 변화를 균등하게 샘플링하는 것이 중요하다. 또한 각 시행에서 모든 조건의 반복 횟수를 동일하게 유지해야 하며 순서 효과를 최소화할 수 있도록 순서를 최적화해야 한다.(앞의 사항들이 조건이나 결과와 공변할 수 있기 때문이다) 그리고 특정 조건이 인지적 난이도와 반응 시간 차이를 유발하는 경우 이는 결과 해석에 어려움을 겪을 수 있다.

#### MVPA가 단변량 분석보다 잡음에 민감한 이유

MVPA는 모든 정보를 학습하여 조건을 구별하려고 하므로, 연구자가 의도하지 않은 잡음 정보도 학습할 가능성이 있다. 단변량 분석에서는 일반적으로 특정 뇌 영역에서 신호의 강도 변화 만을 평가하지만, MVPA에서는 신경 반응 패턴의 차이를 분석하므로 신호 강도뿐만 아니라 미묘한 패턴 변화도 분석 대상이 된다. 이러한 문제를 방지하기 위해, 전통적인 잡음 샘플링 방법 외에도 MVPA 연구에 특화된 방법을 추가로 적용하는 것이 필요하다(Görgen et al., 2018).

#### 장비 관련 잡음과 순서 효과 최소화 전략

MRI 실험에서 흔히 발생하는 잡음 중 하나는 스캐너 드리프트(scanner drift) 이다. 이는 스캔 세션 동안 신호가 서서히 변화하는 현상으로, 특정 조건이 특정 시간대에 집중적으로 등장하면 신호 변화가 실험 조건과 함께 변할 수 있다. 이를 해결하는 가장 일반적인 방법은 실험 내에서 자극 순서를 무작위화(randomization)하는 것이다(Mumford et al., 2014).

일부 연구에서는 실험 순서를 최적화하는 데 수학적 알고리즘을 활용하여 최적의 실험 설계를 도출하는 방법을 사용한다. 대표적인 기법은 다음과 같다. de Bruijn cycles (Aguirre et al., 2011)은 일정한 규칙을 유지하면서 모든 가능한 자극 조합이 고르게 나타나도록 설계하는 알고리즘으로, 특정 조건이 특정 다른 조건과 연속적으로 등장하는 빈도를 최소화할 수 있다. m-sequences (Buračas & Boynton, 2002)는 자극의 순서를 결정할 때, 특정 패턴이 반복되지 않도록 최적화하는 방법으로 주어진 실험 조건에서 가능한 모든 조합을 고르게 포함하도록 설계할 수 있다.

## Analytical considerations

### Analytical considerations specific to decoding analyses

#### Types of algorithms

#### Overfitting

#### Hyperparameter tuning

### Analytical considerations specific to RSA.

### General analytical considerations: processing and selecting features

### How and when to smooth

### Feature selection

### Dimension reduction

## Analytical steps

### First steps

### Classification analysis

### Representational similarity analysis

### Statistical testing


<br>

# What questions can we ask with MVPA?

## Brain-reading

## Stages of neural processing

## Underlying neurocognitive mechanisms

## Individual differences


<br>

# Issues in MVPA

## What are we measuring, content or process?

## Beyond static multivoxel response patterns

## Within- vs between-subject decoding

## Imaging resolution

### Examining multivoxel, rather than multi-neuron, patterns can systematically produce both false positives and false negatives

## Uncertainty about the timing of social and affective processes


<br>

# Conclusion



<figure class='align-center'>
    <img src = "/images/2025-03-16-Tools of the Trade Multivoxel pattern analysis in fMRI a practical introduction for social and affective neuroscientists/figure1.jpg" alt="">
    <figcaption>figure 1. caption</figcaption>
</figure>


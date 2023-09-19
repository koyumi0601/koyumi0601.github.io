---
layout: single
title: "Feature-Adaptive Motion Tracking of Ultrasound Image Sequences Using A Deformable Mesh"
categories: physics
tags: [Physics, Ultrasound, Motion Tracking, Signal Processing, Paper]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

*Feature-Adaptive Motion Tracking of Ultrasound Image Sequences Using A Deformable Mesh, Fai Yeung*

변형 가능한 메시(Deformable Mesh)를 사용한 초음파 이미지 시퀀스의 특징 적응형 모션 추적(Feature-Adaptive Motion Tracking)

Abstract—By exploiting the correlation of ultrasound speckle patterns that result from scattering by underlying tissue elements, two-dimensional tissue motion can theoretically be recovered by tracking the apparent movement of the associated speckle patterns. However, speckle tracking is an ill-posed inverse problem due to temporal decorrelation of the speckle patterns and the inherent low signal-to-noise ratio of medical ultrasonic images.

This paper investigates the use of an adaptive deformable mesh for nonrigid tissue motion recovery from ultrasound images. The nodes connecting the mesh elements are allocated adaptively to stable speckle patterns that are less susceptible to temporal decorrelation. We employ the approach of finite element analysis in manipulating the irregular mesh elements. A novel deformable block matching algorithm, which makes use of a Lagrange element for a higher-order description of local motion, is proposed to estimate a nonrigid motion vector at each node.

To ensure that the motion estimates are physically admissible and yield a plausible solution, the nodal displacements are regularized by minimizing the associated strain energy of the deformable mesh. Experiments based on ultrasound images of a muscle undergoing contraction, and on a phantom mimicking the tissue, as well as computer simulations, have shown that the proposed algorithm is effective.

요약—초음파 스페클(ultrasound speckle) 패턴의 상관관계를 활용해 기저 조직의 산란(scattering) 결과로 나타나는 이러한 패턴의 명백한 움직임을 추적(track)함으로써, 이론적으로 두 차원 조직의 움직임을 복구(recover)할 수 있습니다. 그러나 스페클 추적(speckle tracking)은 스페클 패턴의 시간적 분리(temporal decorrelation)와 의료 초음파 영상의 본래 낮은 신호 대 잡음 비율(signal-to-noise ratio) 때문에 문제가 있는 역문제(ill-posed inverse problem)입니다.

이 논문은 초음파 영상에서 비강성(nonrigid) 조직 움직임을 복구하기 위한 적응형 변형 가능한 메시(adaptive deformable mesh)의 사용에 대해 조사합니다. 메시(mesh) 요소를 연결하는 노드(node)는 시간적 분리에 덜 취약한 안정된 스페클 패턴에 적응적으로 할당됩니다. 불규칙한 메시 요소를 조작(manipulate)하기 위해 유한 요소 분석(finite element analysis)의 접근법을 사용합니다. 로컬 움직임의 고차원 설명을 위한 라그랑주 요소(Lagrange element)를 사용하는 새로운 변형 가능한 블록 매칭(deformable block matching) 알고리즘이 각 노드에서 비강성 운동 벡터(nonrigid motion vector)를 추정하기 위해 제안됩니다.

움직임 추정치가 물리적으로 허용 가능하고 타당한 해결책을 제공하기 위해, 노드의 변위(displacement)는 변형 가능한 메시의 연관된 변형 에너지(strain energy)를 최소화함으로써 정규화(regularized)됩니다. 근육이 수축(contraction) 중인 초음파 영상과 조직을 흉내 내는 팬텀(phantom), 그리고 컴퓨터 시뮬레이션(computer simulations)을 기반으로 한 실험은 제안된 알고리즘이 효과적이라는 것을 보여줍니다.

# 요약

- 초음파 스페클 패턴: 이 패턴은 조직의 움직임을 이론적으로 추적하고 복구하는 데 사용됩니다.

- 문제점: 스페클 패턴의 시간적 분리와 의료 초음파 이미지의 낮은 신호 대 잡음 비율로 인해, 스페클 추적은 복잡한 문제입니다.

- 적응형 변형 가능한 메시: 이를 사용하여 비강성 조직 움직임을 복구합니다. 메시의 노드는 시간적 분리에 덜 취약한 스페클 패턴에 할당됩니다.

- 유한 요소 분석: 이를 통해 불규칙한 메시 요소를 조작하고 관리합니다.

- 새로운 알고리즘: 로컬 움직임을 고차원으로 설명하기 위한 새로운 변형 가능한 블록 매칭 알고리즘이 제안됩니다.

- 물리적 타당성: 움직임 추정치가 물리적으로 허용 가능하도록, 노드의 변위는 메시의 연관된 변형 에너지를 최소화하여 정규화됩니다.

- 실험 결과: 근육 수축, 조직을 흉내 내는 팬텀, 컴퓨터 시뮬레이션을 기반으로 한 실험은 제안된 알고리즘이 효과적임을 보여줍니다.
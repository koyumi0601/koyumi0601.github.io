import matplotlib.pyplot as plt
import numpy as np

# 활성화 함수 그래프 그리기 함수
def plot_activation_functions():
    x = np.linspace(-5, 5, 1000)
    
    # 활성화 함수 정의
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    leaky_relu = np.where(x > 0, x, 0.01 * x)
    swish = x * sigmoid
    hard_sigmoid = np.clip(0.2 * x + 0.5, 0, 1)
    hard_swish = x * hard_sigmoid
    elu = np.where(x > 0, x, 1.0 * (np.exp(x) - 1))
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    # 활성화 함수와 특장점 및 단점을 포함한 정의
    # Activation functions with their descriptions
    activation_functions = [
        (sigmoid, 'Sigmoid', '$\\frac{1}{1 + e^{-x}}$', 'Advantages: Bounded between 0 and 1\nDrawbacks: Gradient vanishing'),
        (tanh, 'Tanh', '$\\tanh(x)$', 'Advantages: Bounded between -1 and 1\nDrawbacks: Gradient vanishing'),
        (relu, 'ReLU', '$\\max(0, x)$', 'Advantages: Fast computation, easy optimization\nDrawbacks: Dead neurons'),
        (leaky_relu, 'Leaky ReLU', '$x$ if $x > 0$ else $0.01x$', 'Advantages: Mitigates dead neurons\nDrawbacks: Varying performance'),
        (swish, 'Swish', '$x \\cdot \\frac{1}{1 + e^{-x}}$', 'Advantages: Smooth non-linearity\nDrawbacks: Slow computation'),
        (hard_sigmoid, 'Hard Sigmoid', '$0$ if $x < -2.5$ else $0.2x + 0.5$ if $-2.5 \\leq x \\leq 2.5$ else $1$', 'Advantages: Fast, simple computation\nDrawbacks: Reduced accuracy'),
        (hard_swish, 'Hard Swish', '$x \\cdot (1 / (1 + e^{-x}))$', 'Advantages: Easily optimized\nDrawbacks: Reduced accuracy'),
        (elu, 'ELU', '$x$ if $x > 0$ else $(e^x - 1)$', 'Advantages: Mitigates dead neurons, smooth non-linearity\nDrawbacks: Slower computation'),
        (gelu, 'GELU', '$0.5x (1 + tanh(sqrt(2/π)(x + 0.044715 * x^3)))$', 'Advantages: Smooth non-linearity\nDrawbacks: Slower computation')
    ]
    

    # 유사한 도메인에서 활성화 함수 색상 설정
    colors = ['b', 'b', 'g', 'g', 'r', 'c', 'r', 'c', 'r']

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))  # 더 큰 크기로 조절
    plt.subplots_adjust(wspace=0.4)  # 서브플롯 간격 조정
    for i, (activation_function, function_name, equation, description) in enumerate(activation_functions):
        ax = axes[i // 3, i % 3]
        y = activation_function
        ax.plot(x, y, label=f"{equation}", color=colors[i])
        ax.legend(loc='upper left')
        ax.grid(True)
        # ax.text(0.5, -0.25, description, transform=ax.transAxes, fontsize=10, va='top')
        # ax.text(0.5, 1.1, description, transform=ax.transAxes, fontsize=10, va='center', ha='center')
        ax.text(0.5, -0.25, description, transform=ax.transAxes, fontsize=10, va='bottom', ha='center')
        
        # 각 subplot의 타이틀에 활성화 함수 이름 설정
        ax.set_title(function_name)
        
    # plt.tight_layout()
    plt.tight_layout(pad=2.0)
    # plt.subplots_adjust(wspace=0.4, hspace=0.6)  # wspace와 hspace 값을 조정하여 간격 설정
    plt.show()

# 모든 활성화 함수 그래프 그리기
plot_activation_functions()
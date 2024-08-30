在 SVM 中有很多輸入，然後乘上許多係數，再加上偏誤，最後得到一個數字，然後再丟進一些函數如 Logistic 函數，然後使用該函數的輸出。所以 SVM 只有輸入向量資料，然後乘上一個向量形式的係數，經由分類器得到一個數字，這在神經網絡(NN)中稱為一個神經元。NN 是由許多 SVM 或是其他分類方法連接而成，輸入仍是相同的資料，但根據不同的分類器可以選用不同的係數。SVM 的數學形式為 $$\ y = g\sum_{i=1}^{n}(w_{i}x_{i} + b) $$，其結構可以被畫成如下左圖，若是有兩個 SVM 則可以畫成下右圖，每個 SVM 都使用相同的輸入\
![img](https://github.com/JrPhy/MachineLearning/blob/master/Neural%20Network/img/Two_SVM.jpg)\
所以 SVM 被稱為神經元，許多 SVM 連接起來即稱為神經網路。輸入曾是資料，輸出曾是結果，其他中間層稱為隱藏層，所以 g 與 G 為隱藏層。
![img](https://github.com/JrPhy/MachineLearning/blob/master/Neural%20Network/img/hidden_layer.jpg)

## 1. 數學形式
一般使用 w<sub>ij</sub><sup>l</sup> 來表示 NN 的係數，分數 $$\ s = \sum_{j=0}^{l-1}w_{ij}^{l}x_{j} $$
| 層 | 輸入 | 輸出 |
|--- | --- | ---  |
|1 ≤ l ≤ L| 0 ≤ i ≤ d<sup>l-1</sup> | 1 ≤ j ≤ d<sup>l</sup> |

目前的輸出為下一層隱藏層的輸入，用內積來得到分數，表示每一層找到輸入資料最像的圖案。假設有兩個輸入，使用向量 x 表示，並且有兩個輸出，所以可以寫成聯立方程式

$$\begin{equation}
    \begin{matrix}
        w_{11}x_{1} + w_{12}x_{2} + b_{1} &=& s_{1}\\
        w_{21}x_{2} + w_{22}x_{2} + b_{2} &=& s_{2}
    \end{matrix}
    \rightarrow
    \begin{bmatrix}
        w_{11}&w_{12} \\
        w_{21}&w_{22} \\
    \end{bmatrix}
    \begin{bmatrix}
        x_{1} \\
        x_{2} \\
    \end{bmatrix}
    +
    \begin{bmatrix}
        b_{1} \\
        b_{2} \\
    \end{bmatrix}
    =
    \begin{bmatrix}
        s_{1} \\
        s_{2} \\
    \end{bmatrix}
\end{equation}$$

![img](https://github.com/JrPhy/MachineLearning/blob/master/Neural%20Network/img/NN.jpg)\
全部的輸入跟神經元都是往前傳播，所以這個 NN 結構又稱為完全連接的神經元。實際上可以根據問題建構自己的神經元，在圖像辨識中，CNN 是很流行的，因為可以消除一些不重要的權重，效能很高。遞迴神經網路則是常用在 NLP，所以根據問題選擇網路是很重要的。而此數學式可寫成

y = f(x) = σ(w<sup>T(L)</sup>σ(w<sup>T(L-1)</sup>σ(w<sup>T(L-2)</sup>...σ(w<sup>T(1)</sup>x<sup>(0)</sup>)...)

舉例來說想要辨認一張 16*16 的圖片，那麼輸入向量長度為 256。

## 2. 背向傳播
與機器學習的方法相同，我們想要最小化誤差，並且仍使用平方誤差。從最後一層開始，誤差為 e<sub>n</sub> = (y<sub>n</sub> - s<sub>n</sub><sup>(L)</sup>)<sup>2</sup>

$$\ \frac{\partial e_{n}}{\partial w_{il}^{(L)}} = \frac{\partial e_{n}}{\partial s_{1}^{(L)}}\frac{\partial s_{1}^{(L)}}{\partial w_{il}^{(L)}} = -2(y_{n} - s_{1}^{(L)})x_{j}^{(L-1)}, \frac{\partial e_{n}}{\partial s_{1}^{(L)}} = \delta_{1}^{(L)} $$

對於隱藏層來說，假設我們使用 σ 來得到分數，第一層到第 L-1 層為隱藏層，那麼最終分數為

y<sub>n</sub> = s<sub>n</sub><sup>(L)</sup> = σ(w<sup>T(L)</sup>s<sup>(L-1)</sup>) = 
σ(w<sup>T(L)</sup>σ(w<sup>T(L-1)</sup>s<sup>(L-2)</sup>)) = ... = σ(w<sup>T(L)</sup>σ(w<sup>T(L-1)</sup>σ(w<sup>T(L-2)</sup>...σ(w<sup>T(1)</sup>x<sup>(0)</sup>)...)

假設有三層，那麼

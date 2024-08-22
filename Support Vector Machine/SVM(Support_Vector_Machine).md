在機器學習中的基本問題就是分類，若是二元分類的話有很多種方法，PLA 就是其中一種解法，但不是唯一解，所以會加入一些限制條件來得到一條線或是超平面，下圖就是將 O 與 X 用一條線分開的例子
![img](https://github.com/JrPhy/MachineLearning/blob/master/Support%20Vector%20Machine/img/classfy.jpg)

## 1. 限制條件
在此的限制條件就是在 O X 的距離為最小下找到一條線，所以我們就先將距離最近的 O X 兩點找出來，如此一來該線就會通過 O X 的中點，所以

$$\begin{equation}
    \begin{bmatrix}
        w_{1} \\ w_{2} \\ w_{0}
    \end{bmatrix}
    \begin{bmatrix}
        x_{1} \\
        x_{2} \\
        x_{0} \\
    \end{bmatrix}
    = w_{1}x_{1} + w_{2}x_{2} + w_{0}x_{0} = 0
\end{equation}$$

## 2. 推導
利用點到線的距離來推導，此距離意思是在歐氏幾何中與線的垂直距離，也是「最短距離」，所以是唯一存在的。令 P(x<sub>0</sub>, y<sub>0</sub>) 到直線 L = ax + by + c = 0 的距離為 d(p, L)，則

$$d(p, L) = \dfrac{|ax_{0} + by_{0} + c|}{\sqrt{a^2 + b^2}}$$

#### 證明
假設 L' 通過 P 點且垂直 L，則 
L' = bx – ay = (bx<sub>0</sub> – ay<sub>0</sub>
)\
L = ax + by + c = 0\
求解 x, y 可得

$$B(x, y) = (\dfrac{b^2x_{0} - aby_{0} - ac}{a^2 + b^2}, \dfrac{a^2y_{0} - abx_{0} - bc}{a^2 + b^2})$$

$$d(p, L) = \sqrt{(x_{0} - \dfrac{b^2x_{0} - aby_{0} - ac}{a^2 + b^2})^2 + (y_{0} - \dfrac{a^2y_{0} - abx_{0} - bc}{a^2 + b^2})^2}$$

$$ = \dfrac{1}{a^2 + b^2}\sqrt{((a^2x_{0} + aby_{0} + ac)^2 + (b^2y_{0} + abx_{0} + bc)^2)}$$

$$ = \dfrac{1}{a^2 + b^2}\sqrt{((a^2 + b^2)^2c^2 + (a^2 + b^2)^2a^2x_{0}^2 + (a^2 + b^2)^2b^2y_{0}^2)}$$

$$ = \dfrac{|ax_{0} + by_{0} + c|}{\sqrt{a^2 + b^2}}$$

得證


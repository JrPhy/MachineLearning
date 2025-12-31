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
L' = bx – ay = (bx<sub>0</sub> – ay<sub>0</sub>)\
L = ax + by + c = 0\
求解 x, y 可得

$$B(x, y) = (\dfrac{b^2x_{0} - aby_{0} - ac}{a^2 + b^2}, \dfrac{a^2y_{0} - abx_{0} - bc}{a^2 + b^2})$$

$$d(p, L) = \sqrt{(x_{0} - \dfrac{b^2x_{0} - aby_{0} - ac}{a^2 + b^2})^2 + (y_{0} - \dfrac{a^2y_{0} - abx_{0} - bc}{a^2 + b^2})^2}$$

$$ = \dfrac{1}{a^2 + b^2}\sqrt{((a^2x_{0} + aby_{0} + ac)^2 + (b^2y_{0} + abx_{0} + bc)^2)}$$

$$ = \dfrac{1}{a^2 + b^2}\sqrt{((a^2 + b^2)^2c^2 + (a^2 + b^2)^2a^2x_{0}^2 + (a^2 + b^2)^2b^2y_{0}^2)}$$

$$ = \dfrac{|ax_{0} + by_{0} + c|}{\sqrt{a^2 + b^2}}$$

得證#

假設要在二為平面找一條線，那麼 w = (a, b) 與, x = (x<sub>0</sub>, y<sub>0</sub>) 皆為向量，||w|| = $$\sqrt{a^2 + b^2}$$，w<sub>0</sub>x<sub>0</sub> = c = b

$$d(p, L) = \dfrac{|ax_{0} + by_{0} + c|}{\sqrt{a^2 + b^2}} = \dfrac{|w^Tx +b|}{||w||}$$

剩下的就如同 PLA，y = 1 or -1 $$\ \rightarrow $$ y(w<sup>T</sup>x +b) > 0 為一邊，< 0 為另外一邊。

## 3. 最佳化
結合標籤後只須加上 y(w<sup>T</sup>x +b) ≥ 1，接著要來找最大的距離，根據前面的條件，最大距離為 max (||w||<sup>-1</sup>)，根據我們的限制條件可以寫成

$$\ \max_{b, w} \dfrac{1}{||w||} = \min_{b, w} ||w|| = \min_{b, w} \sqrt{{w^2}} \rightarrow \min_{b, w} \dfrac{1}{2}||w||^2  $$

在此想要在 $$\ y = <w^Tx + b> ≥ 1 $$ 的條件下最小化 $$\ <w^T, w> $$，使用 [Lagrange undetermined multiplier](https://en.wikipedia.org/wiki/Lagrange_multiplier) 來解

$$\ L(\alpha, w, b) = \dfrac{1}{2}||w||^2 - \sum_{i}\alpha_{i}[y_{i}w^Tx + b-1] $$

$$\\alpha$$ 為未定乘子。若第二項比較大，那麼第一項就比較小，所以就是要解

$$\ \max_{b, w} \min_{\forall \alpha_{i} ≥ 0} L(\alpha, w, b) = \min_{b, w} \max_{\forall \alpha_{i} ≥ 0} L(\alpha, w, b) $$

此處利用 [KKT 條件](https://ccjou.wordpress.com/2017/02/07/karush-kuhn-tucker-kkt-%E6%A2%9D%E4%BB%B6/)來交換順序，再來就分別對變數 b, w 微分

$$\ \frac{\partial L(\alpha, w, b)}{\partial w} = ||w|| - \sum_{i}\alpha_{i}y_{i}x = 0, \frac{\partial L(\alpha, w, b)}{\partial b} = \sum_{i}\alpha_{i}y_{i} = 0 $$ 

再帶回 L 可得 

$$\ L(\alpha, w, b) = \dfrac{1}{2}||w||^2 - \sum_{i}\alpha_{i}[y_{i}w^Tx + b-1]$$

$$\ = \dfrac{1}{2}||\sum_{i}\alpha_{i}y_{i}x||^2 - \sum_{i}\alpha_{i}[y_{i}w^Tx + b-1] - \sum_{i}(\alpha_{i}y_{i}w^T + \alpha_{i}y_{i}b - \alpha_{i} $$

$$\ = -\dfrac{1}{2}||\sum_{i}\sum_{j}\alpha_{i}y_{i}x^T\alpha_{j}y_{j}x||^2 + \sum_{i}\alpha_{i} $$

其中第二個等號中的第二個累加前兩項會消成 0，最後就如上式。

在此我們已經決定最佳化的 w，下一步就是要解最佳化的 b。在第二項中如果 $$\alpha$$ = 0，則 b 可為任意數。所以讓 $$\alpha$$ > 0，則這些點就都在邊界上，即稱為支持向量(Support Vector)，故所求為

$$\ \min_{\forall \alpha_{i} ≥ 0} \dfrac{1}{2}||\sum_{i}\sum_{j}\alpha_{i}y_{i}x^T\alpha_{j}y_{j}x||^2 - \sum_{i}\alpha_{i} $$

## 3. 範例
有三個點 A(3, 3, 1), B(4, 3, 1), C(1, 1, -1)，最後一位為標籤。首先 $$\ \sum_{i}\alpha_{i}y_{i} = 0 \rightarrow \alpha_{1} + \alpha_{2} = \alpha_{i}y_{3} $$

$$\ \min_{\forall \alpha_{i} ≥ 0} \dfrac{1}{2}||\sum_{i}\sum_{j}\alpha_{i}y_{i}x^T\alpha_{j}y_{j}x||^2 - \sum_{i}\alpha_{i} $$

$$\ = \min_{\forall \alpha_{i} ≥ 0} \dfrac{1}{2}(8\alpha_{1}^2 + 20\alpha_{1}\alpha_{2} + 13\alpha_{2}^2) - 2\alpha_{1} - 2\alpha_{2} $$

$$\ \dfrac{1}{\partial \alpha_{1}} \dfrac{1}{2}(8\alpha_{1}^2 + 20\alpha_{1}\alpha_{2} + 13\alpha_{2}^2) - 2\alpha_{1} - 2\alpha_{2} = 8\alpha_{1} + 10\alpha_{2} - 2 = 0$$

$$\ \dfrac{1}{\partial \alpha_{2}} \dfrac{1}{2}(8\alpha_{1}^2 + 20\alpha_{1}\alpha_{2} + 13\alpha_{2}^2) - 2\alpha_{1} - 2\alpha_{2} = 10\alpha_{1} + 13\alpha_{2} - 2 = 0$$

由上兩式可得 $$\ \alpha_{1} = 1.5, \alpha_{2} = -1 $$，但此解並不滿足限制條件，故最小值應於邊界，將 $$\ \alpha_{1} = \alpha_{2} = 0 $$ 帶入可得 

$$\ \alpha_{1} = 0 \rightarrow \alpha_{3} = \alpha_{2} = \dfrac{2}{13}, min = \dfrac{-2}{13} $$

$$\ \alpha_{2} = 0 \rightarrow \alpha_{3} = \alpha_{1} = \dfrac{1}{4}, min = \dfrac{-1}{4} $$

故 $$\ \dfrac{-1}{4} $$ 為最小值，帶入 w 與 b 可得\
w = (-1)* (3, 3)/4 + (-1)* (4, 3)/4 = (0.5, 0.5)\
b = y - w<sup>T</sup>x = 1 - (-1)* 18/4 + (-1)*20/4 = 2\
所得分隔線為 0.5x + 0.5y = 2。所以在 SVM 中只關心在邊界上的點，在邊界內的就不用做計算

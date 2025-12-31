雖然我們可以接受錯誤，但還是希望錯誤盡量少，這就叫做正則化。現在我們已經找到一個超平面，並且加上一個條件 $$\ \min_{w}E_{in}(w) $$ 使得 $$\ \sum_{i}w_{i}^2 ≤ C $$。這問題可以使用拉格朗日未定乘子法來解，在此我們使用平方誤，令 z<sub>i</sub> = f(x<sub>i</sub>) 為我們的轉換，那麼誤差即為

$$\ E(w_{i}) = (z_{i}w^T - y_{i})^2, E = \sum_{i}E(w_{i}) = (Zw^T - Y)(Zw^T - Y)^T $$

## 1. 最佳化
在此我們要解的問題為 $$\ \min_{w}((Zw^T - Y)(Zw^T - Y)^T + \frac{\lambda}{N}ww^T)\lambda > 0 $$

對 w 微分求極值可得

$$\ \frac{\partial}{\partial w}((Zw^T - Y)(Zw^T - Y)^T + \frac{\lambda}{N}ww^T) = 0 = \frac{2}{N}(ZZ^Tw^T - Z^TY + \frac{\lambda w^T}{N})$$

可得 $$\ w = (ZZ^T+\lambda I)^{-1} ZY $$

當 x 增加時，高次項會增加很快，所以高次項的係數通常很小。所以 $$\ \lambda $$ 可以壓過高次項。
![img](https://github.com/JrPhy/MachineLearning/blob/master/Support%20Vector%20Machine/img/Regularization.jpg)\
在選擇多項式時可以選擇 Legendre 函數，此函數有較好的數值特性

$$\ -1 ≤ L(x_{i}) ≤ 1, \int_{-1}^{1} L(x_{i})L(x_{j}), dx = \delta_{ij} $$

目前使用的平方誤稱為 L2 正則化，使用絕對值則稱為 L1 正則化，兩者都可以避免雜訊。一般會使用 L2 是因為比較好求最小值，因為處處可微分。但若是 w 項數很少時就可以用 L1 正則化，因為細數較少，所以求值較快。下方給出了一些選擇的條件
1. 與目標相關
2. 使用者想要
3. 優化較簡單

我們已知正則化 (Regularization) 為 $$\ \min_{w} \dfrac{\lambda}{N}ww^T + \dfrac{1}{N}\sum_{i}err_{i} $$ ，使用表現理論，則其可被表示為

$$\ \min_{w} \sum_{i=1}^{N}\sum_{j=1}^{N}\beta_{i}\beta_{j}K(x_{i}, x_{j}) + \dfrac{1}{N}\sum_{i=1}^{N}log(1+exp(-y_{n}\sum_{j=1}^{N}\beta_{j}K(x_{i}, x_{j}))) $$ 

接著把平方誤 $$\ (y_{n} - w^Tz_{n})^2 $$ 帶進 err<sub>i</sub>，然後來導出解析解

## 最佳化
我們想要最佳化的函數為

$$\ \min_{w} \dfrac{\lambda}{N}w_{opt}w_{opt}^T +  \sum_{i=1}^{N}(y_{n} - w_{opt}^Tz_{n})^2 $$ 

其中 $$\ w_{opt} = \sum_{i=1}^{N}\beta_{i}z_{i}, z_{i} = \Phi(x_{n})$$ ，故上式可寫成

$$\ \dfrac{\lambda}{N}\sum_{i=1}^{n}\sum_{j=1}^{m}\beta_{i}\beta_{j}K(x_{i}, x_{j}) + \dfrac{1}{N}\sum_{i=1}^{n}(y_{i} - \sum_{j=1}^{m}\beta_{i}K(x_{i}, x_{j}))^2 $$ 

$$ \dfrac{1}{N}\sum_{i=1}^{n}(y_{i} - \sum_{j=1}^{m}\beta_{i}K(x_{i}, x_{j}))^2 = ||Y - \beta K||^2 = YY^T + \beta KK^T \beta^T - 2\beta^TKY $$ 

$$\ \sum_{i=1}^{n}\sum_{j=1}^{m}\beta_{i}\beta_{j}K(x_{i}, x_{j}) = \beta K\beta^T$$

在此我們想要最小化的量為
$$\ E(\beta) = \dfrac{\lambda}{N}\beta K\beta^T + \dfrac{1}{N}(YY^T + \beta KK^T \beta^T - 2\beta^TKY)$$

$$\ \nabla E(\beta) = 2\dfrac{\lambda}{N}\beta K + \dfrac{1}{N}(2\beta KK^T \beta^T - 2\beta^TKY) = 2\dfrac{K}{N}(\lambda\beta + \beta K^T - Y) $$

$$\ = 2\dfrac{K}{N}(\beta (\lambda I + K^T) - Y)$$

所以解析解為

$$\ \beta = (\lambda I + K)^{-1} Y$$

由 [Mercer's theorem](https://en.wikipedia.org/wiki/Mercer%27s_theorem)，因為 K 為半正定，對所有 $$\ \lambda $$ 來說， $$\ \lambda IK$$ 都存在。所以到目前為止，除了使用線性回歸外，也可以使用非線性回歸版本的 SVM。如果誤差為平方誤，那麼此種 SVM 被稱為 LSSVM(Least-Square SVM)，或稱 kernel rigde regression。
![IMG](https://github.com/JrPhy/MachineLearning/blob/master/Support%20Vector%20Machine/img/LSSVM.jpg)
## 3. 結合回歸與軟邊界
軟邊界可容許些許錯誤，所以支持向量就會比較少而且也比較快，這就是為什麼我們想結合兩者。
![IMG](https://github.com/JrPhy/MachineLearning/blob/master/Support%20Vector%20Machine/img/Gauss-LSSVM.jpg)
![IMG](https://github.com/JrPhy/MachineLearning/blob/master/Support%20Vector%20Machine/img/regre_soft_margin.jpg)\
假設上途中紅線是由回歸決定的，而虛線所為出的邊界寬度為 $$\ \epsilon $$。因為我們只關心該區域內的誤差，所以 err(y, s) = max(0, |y - s| - $$\ \epsilon $$)\
|y - s| - $$\ \epsilon $$ < 0: err(y, s) = 0\
|y - s| - $$\ \epsilon $$ > 0: err(y, s) = |y - s| - $$\ \epsilon $$\
稱為管回歸。因為誤差為線性的，所以增加的速度比平方誤還小，且當 s 很小時兩者會很接近。現在我們想要最小化下式

$$\ \min_{w} \dfrac{\lambda}{N}ww^T + \dfrac{1}{N}\sum_{i}max(0, |wz_{i} + b - y| - \epsilon) $$ 

但其中有絕對值，有不可微分點，所以仿照標準的 SVM 作法改寫成下式

$$\ \min_{b, w} \dfrac{1}{2}ww^T + C\sum_{i}max(0, |wz_{i} + b - y| - \epsilon) $$ 

其中符號與 SVM 推導相同，所以就變成要求

$$\ \min_{b, w, \xi} (\dfrac{1}{2}ww^T + C\sum_{i}^{n}\xi_{i}) $$ 

其中 $$\ \xi_{i} $$ 為虛線到資料點的鉛直距離。此時我們的限制條件為 $$\ |wz_{i} + b - y| ≤ \epsilon + \xi_{n}, \xi_{n} ≥ 0 $$ 。但因為有絕對值，就不是 QP 問題，所以需要將絕對值拆解掉，分成以下部分討論

$$\ -\epsilon - \xi_{n}^{-} ≤ y - (wz_{i} + b) ≤ +\epsilon + \xi_{n}^{+}, \xi_{n}^{-} ≥ 0, \xi_{n}^{+} ≥ 0, $$ 

所以要最小化的目標為

$$\ \min_{b, w, \xi} (\dfrac{1}{2}ww^T + C\sum_{i}^{n}(\xi_{i}^{-} + \xi_{i}^{-})) $$ 

此時 Lagrange 為

$$\ L(\alpha^+, \alpha^-, w, b, \xi^-, \xi^+, \beta^-, \beta^+) = $$ 

$$\ \dfrac{1}{2}||w||^2 + C\sum_{i}(\xi_{i}^{-} + \xi_{i}^{-}) - \sum_{i}(\beta_{i}^{-} \xi_{i}^{-} + \beta_{i}^{+} \xi_{i}^{+}) + $$ 

$$\ \sum_{i}\alpha_{i}^+[(y - wz_{i} - b) - (\epsilon + \xi_{i}^{+})] + $$ 

$$\ \sum_{i}\alpha_{i}^-[-(y - wz_{i} - b) - (\epsilon + \xi_{i}^{+})]$$ 

$$\ \frac{\partial L}{\partial w_{i}} = w - \sum_{i}^{n}\alpha_{i}^+z_{i} + \sum_{i}^{n}\alpha_{i}^-z_{i} = 0, w = \sum_{i}^{n}(\alpha_{i}^- + \alpha_{i}^+)z_{i}$$ 

$$\ \frac{\partial L}{\partial w_{i}} = w - \sum_{i}^{n}\alpha_{i}^+z_{i} + \sum_{i}^{n}\alpha_{i}^-z_{i} = 0, w = \sum_{i}^{n}(\alpha_{i}^- + \alpha_{i}^+)z_{i}$$ 

$$\ \frac{\partial L}{\partial \xi_{i}^{+}} = 0 = C - \beta_{i}^+ - \alpha_{i}^+, \frac{\partial L}{\partial \xi_{i}^{-}} = 0 = C - \beta_{i}^- - \alpha_{i}^-$$ 

$$\ \frac{\partial L}{\partial b} = 0 = \sum_{i}^{n}(\alpha_{i}^+ - \alpha_{i}^-)$$ 

將以上條件帶入 Lagrange 原式可得

$$\ L(\alpha^+, \alpha^-, w, b, \xi^-, \xi^+, \beta^-, \beta^+) = $$ 

$$\ \sum_{i}^{n}\xi_{i}^{+}(C - \alpha_{i}^+ - \beta_{i}^+) + \sum_{i}^{n}\xi_{i}^{-}(C - \alpha_{i}^- - \beta_{i}^-)$$ 

$$\ - \epsilon\sum_{i}^{n}(\alpha_{i}^+ - \alpha_{i}^-) + \sum_{i}^{n}(\alpha_{i}^+ - \alpha_{i}^-)(y - wz_{i} - b)$$ 

$$\ = \sum_{i}^{n}(\alpha_{i}^+ - \alpha_{i}^-)(y - wz_{i} - b - \epsilon)$$ 

由 KKT 條件可得\
$$\ \alpha_{i}^+[(y - wz_{i} - b) - (\epsilon + \xi_{i}^{+})], (C - \alpha_{i}^+)\xi_{i}^{+} = 0$$\
$$\ \alpha_{i}^-[(y - wz_{i} - b) - (\epsilon + \xi_{i}^{-})], (C - \alpha_{i}^-)\xi_{i}^{-} = 0$$

若 $$\ \xi_{i}^{+} > 0 $$，則 $$\ C = \alpha_{i}^+$$。
$$\ \xi_{i}^{-} > 0 $$，則 $$\ C = \alpha_{i}^-$$，所以 $$\ 0 ≤ \alpha_{i}^-, \alpha_{i}^+ ≤ C$$
![IMG](https://github.com/JrPhy/MachineLearning/blob/master/Support%20Vector%20Machine/img/SVM_SVR.jpg)
## 稀疏的 SVR 解
在軟邊界 SVM，我們只關心區域為內的錯誤，SVR 也是如此。若分類正確，那麼大部分的錯誤會在區域外，所以 $$\ \xi_{n}^{-} = \xi_{n}^{+} = 0 $$，但是
$$\ \alpha_{i}^+[(y - wz_{i} - b) - \epsilon] = 0 $$ 且 $$\ [(y - wz_{i} - b) - \epsilon] ≠ 0 $$，所以 $$\ \alpha_{i}^+ > 0, \alpha_{i}^- > 0 $$，此即為當 SVR 為稀疏時的解，如同 SVM。

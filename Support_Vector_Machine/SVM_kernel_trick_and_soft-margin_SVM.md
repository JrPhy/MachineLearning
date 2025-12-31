到目前為止 SVM 只能解決線性可分的資料，那對於非線性可分的資料呢?一種是接受一些錯誤，稱為軟邊界 SVM，另外一種則是使用非線性函數，又稱為核 SVM。在此來使用多項式核與高斯核。

在 SVM 中我們要算 $$\ \min_{\forall \alpha_{i} ≥ 0} \dfrac{1}{2}||\sum_{i}\sum_{j}\alpha_{i}y_{i}x^T\alpha_{j}y_{j}x||^2 - \sum_{i}\alpha_{i} $$

## 1. 多項式核
x<sup>T</sup>x 為內基，可看成是線性的。我們使用二次多項式為例，假設 f: R<sup>2</sup>->R<sup>3</sup> 且 dim(x) = 2，所以有以下項：(1, x<sub>1</sub>, x<sub>2</sub>, x<sub>1</sub><sup>2</sup>, x<sub>1</sub><sup>2</sup>, x<sub>1</sub>x<sub>2</sub>)，內積為

$$\ x^Tx' ⮕ f(x)^Tf(x) = 1 + \sum_{i=1}^{2}x_{i}x_{i}'+ \sum_{i=1}^{2}\sum_{j=1}^{2}x_{i}x_{i}'x_{j}x_{j}' $$

$$\ = 1 + \sum_{i=1}^{2}x_{i}x_{i}'+ \sum_{i=1}^{2}x_{i}x_{i}'\sum_{j=1}^{2}x_{j}x_{j}' = 1+x^Tx'+(x^Tx')(x^Tx') $$

轉到二次多項式後仍然可以算內積，所以此核為**轉換 + 內積**，在此我們使用 Φ 來表示。對於所有的 x->Φ(x)，我們想要計算的是

$$\ \min_{\forall \alpha_{i} ≥ 0} \dfrac{1}{2}||\sum_{i}\sum_{j}\alpha_{i}y_{i}\Phi(x)^T\alpha_{j}y_{j}\Phi(x)||^2 - \sum_{i}\alpha_{i}, g_{SVM}(x) = sign(w^T\Phi(x) + b) $$

一般來說此多項式核有以下形式

$$\ K_{Q}(x, x') = (\zeta + \gamma x^Tx)^Q, \gamma > 0, \zeta ≥ 0 $$

若將以上參數皆設為 1，那就是線性核。

## 2. 高斯核
使用高斯分布當作基底
$$\ x^Tx' = exp(-(x-x')^2) = exp(-x^2)exp(-x'^2)exp(-2xx') = exp(-x^2)exp(-x'^2)\sum_{i=1}^{\infty}\frac{(2xx')^i}{i!} $$

$$\ = \sum_{i=1}^{\infty} exp(-x^2) \sqrt{\frac{2^i}{i!}}x^i exp(-x'^2) \sqrt{\frac{2^i}{i!}}x'^i$$

$$\ exp(-x^2) \sqrt{\frac{2^i}{i!}}x^i$$ 即為高斯核。一般來說高斯核會寫成 K(x, x') = exp(-γ||x - x'||<sup>2</sup>), γ > 0，表示平均為 x' 標準差為 1/(2γ)，也就是中心是在 SV 上，往左右兩邊的距離為 1/(2γ)。也就是 γ 越大邊界就越窄，如下圖所示。
![IMG](https://ucc.alicdn.com/pic/developer-ecology/fh4lmf6lmlo7m_67c938f54d444c57890141bbce96a81f.png)\
[來源](https://developer.aliyun.com/article/1250144)\
[此影片可以看如何將資料轉到高維度平面上](https://www.youtube.com/watch?v=3liCbRZPrZA)

|     | 線性 | 多項式 | 高斯 |
| --- | --- | --- | --- |
| 優點| 易解釋 | 可用於 | 夠強 |
|     | 快速 | 非線性可分 | |
|     | 不易過擬合 | | |
| 缺點| 只適用於 | 較線性慢 | 非常慢 |
|     | 線性可分 | 非數值穩定 | 易過擬合 |
|     | 不易過擬合 | | 難解釋 |

非數值穩定是說 $$\ ||\zeta + \gamma x^Tx|| $$ 容易太大或太小。當然你也可以選擇自己的核來解決你的問題，只要滿足 K(x, y) = K(y, x) 即可。

## 3. 軟邊界 SVM
除了使用核轉換，接受錯誤或雜訓也是個方法，又稱為軟邊界 SVM，原始的則稱為(硬邊界) SVM。在此我們將錯誤標記為 ξ，稱為懲罰，所以要最佳化的函數為

$$\ min_{b, w, \xi}(\frac{||w||^2}{2} + C\sum_{i=1}^{N}\xi_{i}) $$

此處的 C 為取捨的邊界大小，C 越大表示月少雜訊，邊界就越窄，反之則越多，邊界越寬。所以此時的條件變為

$$\ y(w^Tx + b) ≥ 1 - \xi_{i}), xi_{i} ≥ 0 $$

$$\ L(\alpha, w, b, \xi, \beta) = $$ 

$$\ \dfrac{1}{2}||w||^2 + C\sum_{i}\xi_{i} + \sum_{i}\alpha_{i}(1 - \xi_{i} + y_{i}(w^Tx + b)) +  \sum_{i}-\xi_{i}\beta_{i} $$ 

所以 KKT 條件變為

$$\ \max_{\forall \alpha_{i} ≥ 0, \forall \beta_{i} ≥ 0} \min_{b, w, \xi} L(\alpha, w, b, \xi, \beta) $$

$$\ = \max_{\forall \alpha_{i} ≥ 0, \forall \beta_{i} ≥ 0} \min_{b, w, \xi} \frac{||w||^2}{2}+ C\sum_{i}\xi_{i} + \sum_{i}\alpha_{i}(1 - \xi_{i} + y_{i}(w^Tx + b)) +  \sum_{i}-\xi_{i}\beta_{i} $$

對 ξ 偏微可得 $$\ C - \alpha_{i} ≥ 0 ⮕ C ≥ \alpha_{i} ≥ 0, \forall i \in N $$。在帶回 L 可得

$$\ L(\alpha, w, b, \xi, \beta) = \dfrac{1}{2}||w||^2 + \sum_{i}\alpha_{i}(1 - y_{i}(w^Tx + b))$$ 

與原始的 SVM 完全相同，但條件多了一些。所以軟邊界 SVM = SVM + 其他限制條件。根據條件的不同，有以下四種情況
1. α<sub>i</sub> < C, ξ = 0 --> 即為原始的 SVM，點在邊界上
2. α<sub>i</sub> = C, 1 > ξ > 0 --> 分類正確，但有些點在邊界內
3. α<sub>i</sub> = C, ξ = 1 --> 點在超平面上
4. α<sub>i</sub> = C, ξ > 1 --> 分類錯誤

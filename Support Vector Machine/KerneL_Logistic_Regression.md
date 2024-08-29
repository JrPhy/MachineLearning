現在我們將錯誤的資料記錄在 $$\ \xi $$ 中，然後隨的限制條件最小化 w

$$\ min\_{w}(\frac{||w||^2}{2} + C \sum_{i}\xi_{i}) = min\_{w}(\frac{ww^T}{2} + C \sum_{i}err_{i}) $$

上式類似於我們正則化的目標 $$\ min\_{w}(\frac{\lambda}{N} ww^T + \frac{1}{N}\sum_{i}err_{i}) $$

現在我們要將其連結軟邊界 SVM 與 Logistic 回歸。軟邊界 SVM 的條件為

$$\ y(w^Tx + b) ≥ 1 - \xi_{i}), xi_{i} ≥ 0 ⭢ xi_{i} ≥ 1 - y(w^Tx + b) $$

等效於 max(1-y(w<sup>T</sup>x + b), 0)，所以軟邊界 SVM 的分數為 err<sub>SVM</sub> = max(1-y(w<sup>T</sup>x + b), 0)，Logistic 回歸 的分數為 err<sub>Logistic</sub> = log<sub>2</sub>(1-y(w<sup>T</sup>x + b), 0)。所以

$$\ ys ⭢ \infty, err_{SVM}, err_{Logistic} ⭢ 0 $$

$$\ ys ⭢ -\infty, err_{SVM}, err_{Logistic} ⭢ \infty $$

所以我們可以執行 SVM 來得到 w<sub>SVM</sub> 與 b<sub>SVM</sub>，然後再丟進 Logistic 回歸中，但是我們還無法得到伴隨機率分布的目標函數。或者我們可以設定 b, w 當作初始條件，然後使用梯度下降得到 w<sub>opt</sub> 與 b<sub>opt</sub>，但我們就不能使用非線性轉換的 kernel trick，所以我們必須修改分數的計算。想法是先執行 SVM 伴隨著 kernel，然後將分數帶成 $$\ z = w_{SVM}^T\Phi(x) + b_{SVM} $$，再乘 A 然後 +B 就可得到 Az+B，那麼此形式就類似於 Logistic 回歸。

$$\ g(x) = \theta(Az+B) = \theta(A(w_{SVM}^T\Phi(x) + b_{SVM}) + B) $$

$$\ min_{A, B}\frac{1}{N}\sum_{i=1}^{N}log(1+exp(-y_{n} (A(w_{SVM}^T\Phi(x) + b_{SVM}) + B) $$

在此需要 A > 0, B ~ 0，否則 SVM 的解會非常糟。此稱為**機率 SVM**，由 Platt 提出，所以又稱為 Platt 模型。先執行 SVM 再做 Logistic 回歸，但這只能得到近似解，下一步我們想要藉由 Logistic 回歸找到一個解。

回憶一下為什麼要做 Kernel，w<sub>opt</sub> 為 z 的線性組合

$$\ w_{opt} = \sum_{i=1}^{N}\beta_{n} z_{n} ⭢ w_{opt}^T z_{n} = \sum_{i=1}^{N}\beta_{n} z_{n}z_{n}^T = \sum_{i=1}^{N}\beta_{n} K(x_{n}, x) $$

所以此時我們就需要將 w<sub>opt</sub> 用 z 表示

## 表現理論
宣稱：對於任意 L2 正則化的線性模型有
$$\ min\_{w}(\frac{\lambda}{N}ww^T + \frac{1}{N} \sum_{i}err_{i}), w_{opt} = \sum_{i=1}^{N}\beta_{i} z_{i} $$

證明：令 w<sub>opt</sub> = w<sub>||</sub> + w<sub>⊥</sub>，w<sub>||</sub> 由 z<sub>n</sub> 生成，w<sub>⊥</sub> 與 z<sub>n</sub> 是線性相依的，所以\ err(y<sub>n</sub>, z<sub>opt</sub><sup>T</sup>, z<sub>n</sub>) = err(y<sub>n</sub>, (w<sub>⊥</sub>+w<sub>||</sub>)<sup>T</sup>, z<sub>n</sub>)\
那麼 w<sub>opt</sub>w<sub>opt</sub><sup>T</sup> = w<sub>||</sub>w<sub>||</sub><sup>T</sup> + w<sub>⊥</sub>w<sub>⊥</sub><sup>T</sup> + 2w<sub>||</sub>w<sub>⊥</sub> > w<sub>||</sub>w<sub>||</sub><sup>T</sup> (-><-)，所以 w<sub>⊥</sub><sup>T</sup> = 0

所以經由表現理論，$$\ w_{opt} = \sum_{i=1}^{N}\beta_{i} z_{i} $$

$$\ min\_{w}(\frac{\lambda}{N}ww^T + \frac{1}{N} \sum_{i}err_{i}) = min\_{w}(\sum_{i=1}^{N}\sum_{j=1}^{N}\beta_{i} \beta_{j} K(x_{i}, x_{j}) + \frac{1}{N}\sum_{i=1}^{N} log(1+exp(-y_{n}\sum_{j=1}^{N}\beta_{j} K(x_{i}, x_{j})))) $$

在此 K(x<sub>i</sub>, x<sub>j</sub>) 為 kernel，
$$\ \sum_{j=1}^{N}\beta_{j} K(x_{i}, x_{j}) $$ 
為線性模型

$$\ \sum_{i=1}^{N}\sum_{j=1}^{N}\beta_{i} \beta_{j} K(x_{i}, x_{j}) = \beta K \beta^T$$ 為正則化

所以就可以被看成 $$\ \beta_{i} $$ 伴隨 kernel 的線性模型轉換與 kernel 正則化，就像 SVM。

$$\ \beta_{i} $$ 不常為 0，但是 $$\ \alpha_{i} $$ 在 SVM 中常為 0。

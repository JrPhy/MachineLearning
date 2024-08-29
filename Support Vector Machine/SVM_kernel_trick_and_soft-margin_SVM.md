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

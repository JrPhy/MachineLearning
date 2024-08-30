目前我們已經知道 AdaBoost 跟隨機森林，AdaBoost 是一種弱根據的學習算法+最佳化重新調整權重+線性聚合 α。隨機森林則是藉由取後放回與決策樹來造出許多顆決策樹。現在要將這兩者結合起來，但是隨機森林只有聚合決策樹而沒有權重，所以我們需要有權重的決策樹。

## 1. AdaBoost 決策樹
AdaBoost 與決策樹都需要取後放回，在 BAGging 算法中，權重代表資料集中有多少份取後放回的樣本，而在隨機為底的算法中，權重代表比例，所以我們不需要修改決策樹算法，可以從取樣來得到權種資訊，所以此算法為 AdaBoost + 取樣 + 決策樹。在 AdaBoost 中，權重為 $$\ \alpha = ln\sqrt{\frac{1-t}{t}} $$，t 為權重的錯誤率。\
若樹為完滿的，那麼 E<sub>in</sub> = 0，所以 E<sup>u</sup><sub>in</sub> = 0 --> t = 0 --> α --> ∞\
所以我們不需要一顆完滿的樹，只需要用部分取樣來取代所有的資料，所以修改後的算法為\
AdaBoost + 取樣 + 剪枝後的決策樹\
剪枝是為了限制樹的高度。

## 2. AdaBoost 的權重
$$\ u_{n}^{t+1} = u_{n}^{t}\sqrt{\frac{1-t}{t}} 為正確，u_{n}^{t+1} = u_{n}^{t}\sqrt{\frac{t}{1-t}} 為錯誤$$

$$\ \alpha = ln\sqrt{\frac{1-t}{t}} \rightarrow \sqrt{\frac{1-t}{t}} = e^\alpha \rightarrow u_{n}^{t+1} = u_{n}^{t}\sqrt{\frac{1-t}{t}}^{-y_{n}\alpha_{i}g_{i}x_{n}}$$

$$\ u_{n}^{N+1} = u_{n}^{1}\prod_{i=1}^{N}exp(-y_{n}\alpha_{i}g_{i}x_{n}) = \frac{1}{N}exp(-y_{n}\sum_{i=1}^{N}\alpha_{i}g_{i}(x_{n})) = \frac{1}{N}exp(-y_{n} 票數)$$

票數就像 SVM 的邊界，我們想要最大邊界，表示希望分數是正的且要大，也就是 exp(-y<sub>n</sub> 票數) 要小，所以 u<sup>N+1</sup>n 要小，所以就要最小化

$$\ \sum_{i=1}^{M}u_{n}^{N+1} = \frac{1}{N}\sum_{i=1}^{M}exp(-y_{n}\sum_{i=1}^{N}\alpha_{i}g_{i}(x_{n})) $$

由梯度下降可以知道 f(x+kv) ~ f(x) + kvf(x), ||v|| = 1，下方 y = h(x)

$$\ \frac{1}{N}\sum_{i=1}^{M}exp(-y_{n}\sum_{i=1}^{N}\alpha_{i}g_{i}(x_{n})) \rightarrow  \frac{1}{N}\sum_{i=1}^{M}exp(-y_{n}\sum_{i=1}^{N}\alpha_{i}g_{i}(x_{n}) + kv)$$

$$\ = \sum_{i=1}^{M}u_{n}^{N}exp(-y_{n}kv) \approx \sum_{i=1}^{M}u_{n}^{N}(1-y_{n}kv) = \sum_{i=1}^{M}u_{n}^{N} + \sum_{i=1}^{M}u_{n}^{N}(-y_{n}kv) = E_{in}^{u}$$

所以要最小化最後一項

$$\ = \sum_{i=1}^{M}u_{n}^{N}exp(-y_{n}kh(x)) = \sum_{i=1}^{M}u_{n}^{N}, if y_{n} = h(x), = \sum_{i=1}^{M}-u_{n}^{N}, if y_{n} \neq h(x)$$

$$\ = -\sum_{i=1}^{M}u_{n}^{N}exp(-y_{n}kh(x)) = 0, if y_{n} = h(x), = 2\sum_{i=1}^{M}-u_{n}^{N}, if y_{n} \neq h(x) \rightarrow -\sum_{i=1}^{M}u_{n}^{N} + 2E_{in}^{u}N $$

在 Adaboost 中，基礎算法是要最小化 E<sub>in</sub>，所以 $$\ E_{ada} = \sum_{n=1}^{M}u_{n}^{N}exp(-y_{n}kg_{i}(x_{n})) $$，在梯度下降則是要找 h = g<sub>i</sub>。總結來說

正確 $$\ y_{n}g_{i}(x_{n}) \rightarrow u_{n}^{N}exp(-k) $$

錯誤 $$\ y_{n}g_{i}(x_{n}) \rightarrow u_{n}^{N}exp(-k) $$

$$\ E_{ada} = \sum_{n=1}^{M}u_{n}^{N}((1-t)exp(-k) + texp(k)) $$

對 k 微分後就可以找到最佳的梯度下降 $$\ k* = ln\sqrt{\frac{1-t}{t}} = \alpha $$

所以在 Adaboost，找到 g<sub>i</sub> 去近似 h，然後固定 y<sub>n</sub> 與 h 再去找到 k*，再帶入梯度下降。這方法稱為 **最陡梯度下降**

## 5. 平方誤的梯度加速
在此我們想用一般化的誤差函數，如平方誤，或是每個誤差都可以用梯度下降來解

$$\ min_{k}(min_{h}\frac{1}{N}\sum_{n=1}^{N}err (\sum_{i=1}^{M}\alpha_{i}g_{i}(x_{i}) + kh(x_{i}), y_{n})), err(s, y) = (s-y)^2 $$

$$\ \min_{h}\frac{1}{N}\sum_{n=1}^{N}err (\sum_{i=1}^{M}\alpha_{i}g_{i}(x_{i}) + kh(x_{i}), y_{n}) \approx \min_{h}\frac{1}{N}err(s, y) + \frac{1}{N}\sum_{n=1}^{N}kh(x_{i}) \frac{\partial}{\partial s} err(s_{n}, y_{n})|_ {s=s_{0}}$$

$$\ = \min_{h}\frac{1}{N}err(s, y) + \frac{k}{N}\sum_{n=1}^{N}kh(x_{i}) 2err(s_{n}, y_{n}) $$

第一項為常數，所以要最小化第二項。若 h(x) 沒有限制條件，那麼就可以為負無窮，所以我們需要給限制條件

$$\ \frac{k}{N}\sum_{n=1}^{N}kh(x_{i}) 2err(s_{n}, y_{n}) \rightarrow \frac{k}{N}\sum_{n=1}^{N}(h(x_{i}) 2err(s_{n}, y_{n}) + h^2(x_{i})) = \frac{k}{N}\sum_{n=1}^{N}(h(x_{i} - (y_{n} - s_{n}))^2 + C) $$

在找到 h 後，接者找最佳化的 k

$$\ min_{k}(min_{h}\frac{1}{N}\sum_{n=1}^{N}err (\sum_{i=1}^{M}\alpha_{i}g_{i}(x_{i}) + kh(x_{i}), y_{n})) = min_{k}(\frac{1}{N}\sum_{n=1}^{M}(kh(x_{i} - y_{n} + s_{n})^2))$$

此即為線性回歸問題，目標函數為 g<sub>j</sub>(x<sub>n</sub>), (y<sub>n</sub> - s<sub>n</sub>) 為殘差，k 就是梯度下降的步長。

## 6. 算法
s<sub>1</sub> = s<sub>2</sub> = ... = s<sub>N</sub> = 0\
for t = 1, 2, ..., T
1. 得到 g<sub>t</sub>
2. 計算 α<sub>t</sub>
3. 更新 s<sub>n</sub> = s<sub>n</sub> + α<sub>t</sub>g<sub>t</sub>(x<sub>n</sub>)

回傳 $$\ G(x) = \sum_{t=1}^{T} \alpha_{t}g_{t}(x) $$

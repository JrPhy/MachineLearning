假設要把資料均勻地分成 k 組，那就會有 k 個模型，要如何選擇?其中一個是做驗證就會得到最小的 E<sub>in</sub> + E<sub>out</sub>，另一個是用投票，然後拼接結果，所以可能所有的模型都會用到，稱為聚合。這個過程會藉由投票用到強特徵轉換與正則化，兩者的優點都可以保留。

## 1. 均勻拼接
假設每個投票權重相同，且若我們只想做二元分類，那麼

$$\ G(x) = sign(\sum_{i=1}^{N}g_{i}(x)) $$

g<sub>i</sub>(x) 就是得到的模型，G(x) 就是聚合結果。每個 g<sub>i</sub>(x) 要相異才能保留每個模型的優點。假設目標為 f(x)，E<sub>out</sub> = (G(x) - f(x))<sup>2</sup>，我們想證明 avg(g<sub>i</sub>(x) - f(x))<sup>2</sup> ≥ (G(x) - f(x))<sup>2</sup>

#### 證明
avg(g<sub>i</sub>(x) - f(x))<sup>2</sup> = avg(g<sub>i</sub><sup>2</sup>(x) - 2g<sub>i</sub>(x)f(x) + f<sup>2</sup>(x))

= avg(g<sub>i</sub><sup>2</sup>(x)) - 2G(x)f(x) + f<sup>2</sup>(x))

= avg(g<sub>i</sub><sup>2</sup>(x)) - G<sup>2</sup>(x) + (G(x) - f(x))<sup>2</sup>

= avg(g<sub>i</sub><sup>2</sup>(x)) - 2G<sup>2</sup>(x) + (G(x) - f(x))<sup>2</sup> + G<sup>2</sup>(x)

= avg(g<sub>i</sub><sup>2</sup>(x) - 2g<sub>i</sub>(x)G(x)) + G<sup>2</sup>(x) + (G(x) - f(x))<sup>2</sup>

= avg((g<sub>i</sub>(x) - G(x))<sup>2</sup>) + (G(x) - f(x))<sup>2</sup> ≥ (G(x) - f(x))<sup>2</sup> <sub>#</sub>

所以我們使用模型的加權平均，平均的 E<sub>out</sub> 是小於任意一個模型的誤差。假設模型是從同一組資料來的，但是這組資料被分成很多組，那麼就會遵守同一個分配。如果越多模型在加權內，那麼

$$\ \overline{g} = \lim_{N\to\infty}G = \lim_{N\to\infty} \frac{1}{N}\sum_{i=1}^{N}g_{i}(x)$$

由前面推導可知，(G(x) - f(x))<sup>2</sup> 為偏差，avg((g<sub>i</sub>(x) - G(x))<sup>2</sup>) 為變異數，所以步驟是要減少變異數。現在假設權種不同，所以

$$\ G(x) = sign(\sum_{i=1}^{N}w_{i}g_{i}(x)), err = \frac{1}{N}\sum_{j=1}^{N}(y_{i} - \sum_{i=1}^{N}w_{i}g_{i}(x))^2,  w_{i} ≥ 0 $$

g(x) 就像 SVM 中的和函數，所以拼接可以看成是 線性模型 + 假設轉換 + 限制條件。而在 SVM 中，w<sub>i</sub> 永遠是非負的，在此我們 g(x) 皆取絕對值來讓其為非負。如果在二元分類中為負，表示分類錯誤，但也沒關係，因為其他是正確的，所以在實際問題中是可以被忽略的。

在實際問題中會有許多的 g<sub>i</sub>(x)，並且以E<sub>in</sub> 作為標準來得到 G(x)，但這樣複雜度會非常高而且可能會過擬合。所以推薦將資料集分成訓練集跟驗證集來做交叉驗證，並且使用轉換將其轉到高維度空間中，如此一來就可以得到一個超平面。也有非線性拼接，又叫做堆疊法，比線性強很多，所以只須關心過擬合。

## BAGging
由前面推導得知，g<sub>i</sub>(x) 越多誤差就越小，但實務上很難實現，畢竟 g<sub>i</sub>(x) 為有限值且是從同一組資料集中來的。一種做法是取得樣本後再放回，稱為 Bootstrap，而此作法的距離即稱為為 Bootstrap AGgregating，引導聚集算法
![img](https://upload.wikimedia.org/wikipedia/commons/d/de/Ozone.png)\
且因為每個資料的權重都不同，所以誤差變為

$$\ E_{in}(h) = \frac{1}{N}\sum_{i=1}^{N}u_{i}err(y \neq h(x))$$

我們想要將即最小化，此類似於 SVM 中的限制條件 0 ≤ a ≤ Cu<sub>i</sub>，或是有限制條件與機率 u<sub>i</sub> 的 Logistic 回歸。我們希望 g<sub>j</sub> 是很發散的，意思是指 g<sub>i</sub> 與 g<sub>j+1</sub> 差異很大。對於二元分類來說，期望值會接近 0.5，表示若 g<sub>j</sub> -> 1，那麼 g<sub>j+1</sub> -> 0。所以如果是用 g<sub>j</sub> 來更新 g<sub>j+1</sub>，那結果會變得很糟，所以想要的是

$$\ \frac{\sum_{j=1}^{N}u_{j}^{i+1}[y \neq g_{j}(x)]}{\sum_{j=1}^{N}u_{j}^{i+1}} $$

$$\ = \frac{\sum_{j=1}^{N}u_{j}^{i+1}[y \neq g_{j}(x)]}{\sum_{j=1}^{N}u_{j}^{i+1}[y \neq g_{j}(x)] + \sum_{j=1}^{N}u_{j}^{i+1}[y = g_{j}(x)]} $$

$$\ = \frac{true_{j+1}}{true_{j+1}+flase_{j+1}} = \frac{1}{2} $$

這表示正確與錯誤的量是相同的。假設正確的數量為 n<sub>1</sub>，錯誤的數量為 n<sub>2</sub>，那麼使用加權平均就可以得到

$$\ \frac{n_{1}n_{2}}{n_{1}n_{2}+n_{1}n_{2}} = \frac{1}{2} $$

一般來說我們會假設正確率為 t = α，錯誤率為 1 - t = β

$$\ \alpha(1-t) = \beta t ⮕ \frac{\alpha(1-t)}{\sqrt{t(1-t)}} = \frac{\beta t}{\sqrt{t(1-t)}} ⮕ \alpha\sqrt{\frac{(1-t)}{t}} = \alpha r = \beta\sqrt{\frac{t}{(1-t)}} = \frac{\beta}{r}$$

若 t ≤ 0.5，則 r ≥ 1，表示算法的錯誤率會增加而正確率會降低。

## 3. 演算法
1. 初始化 u
2. for t = 1, 2, ..., T
3. 1. 由 A(D, u<sup>t</sup>)得到 g<sub>t</sub>，A 是想要最小化 u<sup>t</sup> 的 0/1 權重
   2. 藉由 $$\ r = \sqrt{\frac{(1-t)}{t}} $$ 更新 u<sup>t</sup>, t 為 g<sub>t</sub> 的錯誤率
4. 回傳 G(x)

因為 r > 1，所以最好選 u<sub>j</sub><sup>1</sup> = 1/N。若 E<sub>in</sub>(g<sub>j</sub>) 是好的，那麼 E<sub>in</sub>(g<sub>j+1</sub>) 由前面可知就是差的，所以不建議用均勻拼接。

#### 實時版本
1. 初始化 u = [1/N, 1/N, ..., 1/N]
2. for t = 1, 2, ..., T
3. 1. 由 A(D, u<sup>t</sup>)得到 g<sub>t</sub>
   2. 更新 u<sup>t</sup>
   3. 計算 α<sub>t</sub> = ln(r)
4. 回傳 $$\ G(x) = sign(\sum_{t=1}^{T}\alpha_{t}g_{t}(x))

## 4. AdaBoost
對於一個好的 g<sub>j</sub> 而言，乘上一個較大的係數 α 可以來隨時做調整，可以取 $$\ \alpha = ln\sqrt{\frac{(1-t)}{t}} $$，當 g<sub>j</sub> 增加 α 也增加，這比隨機取好。

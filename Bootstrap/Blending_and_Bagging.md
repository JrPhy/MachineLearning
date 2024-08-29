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

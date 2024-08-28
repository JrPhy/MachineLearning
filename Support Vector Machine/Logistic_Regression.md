如果資料是可以被線性分離的，那麼我們只需要直線就能將資料完美分開，但若資料有雜訊呢??假設有一資料 x，其為正確的機率為 P(1|x) = 0.9，分數就不是 1，而是介於 0~1 之間，所以我們就需要使用一個比較彈性的函數去決定是否正確，稱為 Logistic Regression，其數學式如下

$$\ h(s = w^Tx) = \dfrac{exp(w^Tx)}{1+exp(w^Tx)} = \dfrac{1}{1+exp(-w^Tx)} $$ 

## 1. 性質
我們想要來做一個二元分類，但是是用機率決定其為 0 或 1。假設 f(x) = P(1|x)，那麼 f(y) = P(-1|x) = 1 - P(1|x)，然後套用 Logistic 函數

$$\ P(1|x) = \dfrac{exp(w^Tx)}{1+exp(w^Tx)} = \dfrac{1}{1+exp(-w^Tx)} $$ 

$$\ P(0|x) = 1 - P(0|x) = 1 - \dfrac{exp(w^Tx)}{1+exp(w^Tx)} = \dfrac{1}{1+exp(w^Tx)} = -P(-1|x) $$

## 2. 誤差
考慮一組資料集 D = {(x<sub>1</sub>, x), (x<sub>2</sub>, o), ... (x<sub>n</sub>, o), }, f(x<sub>i</sub>) = 0 or 1。假設 f(s<sub>i</sub>)~h(s<sub>i</sub>), 此稱為可能性，所以機率為

P(x<sub>1</sub>)f(s<sub>1</sub>)* P(x<sub>1</sub>)(1-f(s<sub>2</sub>))* P(x<sub>3</sub>)(1-f(s<sub>3</sub>))* ... * P(x<sub>n</sub>)(1-f(s<sub>n</sub>))

~ P(x<sub>1</sub>)f(s<sub>1</sub>)* P(x<sub>1</sub>)(1-h(s<sub>2</sub>))* P(x<sub>3</sub>)(1-h(s<sub>3</sub>))* ... * P(x<sub>n</sub>)(1-h(s<sub>n</sub>))

= P(x<sub>1</sub>)f(s<sub>1</sub>)* P(x<sub>1</sub>)h(-s<sub>2</sub>)* P(x<sub>3</sub>)h(-s<sub>3</sub>)* ... * P(x<sub>n</sub>)h(-s<sub>n</sub>)

= $$\ \prod_{i=1}^{n} P(x_{i})h(y_{i}x_{i}) = \prod_{i=1}^{n} P(x_{i})\theta(y_{i}w^Tx_{i}) $$

## 2. 最佳化
在此我們要最佳化 $$\ \theta(y_{i}w^Tx_{i}) $$，但是因為有相乘所以直接算不好算，所以先取 log 後就可以換成相加再做計算

$$\ \prod_{i=1}^{n} h(y_{i}x_{i}) -> ln\prod_{i=1}^{n} \theta(y_{i}w^Tx_{i}) = \sum_{i=1}^{n} ln\theta(y_{i}w^Tx_{i}) $$

= $$\ max_{w} \prod_{i=1}^{n} h(y_{i}x_{i}) = max_{w} \sum_{i=1}^{n} ln\theta(y_{i}w^Tx_{i}) = min_{w} \sum_{i=1}^{n} -ln\theta(y_{i}w^Tx_{i}) $$

= $$\ min_{w} \sum_{i=1}^{n} -ln(1+exp(-y_{i}w^Tx_{i})) = min_{w}E_{in}(w,x_{i}, y_{i}) $$

接著就對 E<sub>in</sub> 做微分

$$\ \nabla_{i} E_{in}(w,x_{i}, y_{i}) = \frac{\partial}{\partial w_{i}} ln(1+exp(-y_{i}w^Tx_{i})) = \frac{exp(-y_{i}w^Tx_{i})}{1+exp(-y_{i}w^Tx_{i})}(-y_{i}x_{i}) $$

$$\ \nabla_{i} E_{in}(w,x_{i}, y_{i}) = \frac{1}{N} \sum_{i=1}^{N} \theta(-y_{i}w^Tx_{i})(-y_{i}x_{i}) = 0 $$

上式很難證明存在一個唯一的極值，且該極值為最小值，不過確實是存在的，所以可以把它當作一個已知的事實，所以就可以帶入梯度下降算法來找最小值。可以觀察，當 $$\ y_{i}w^Tx_{i} >> 0 $$ 那麼 $$\ exp(y_{i}w^Tx_{i}) $$ ~ 0，此時就退回到線性可分。而如果該 exp 項不為 0，那就表示非線性可分，所以也不是線性函數，就只能找到近似解。

## 3. 回歸分類
到目前為止我們已知知道了線性分類 h(x) = sign(s)，線性回歸h(x) = s 跟 Logistic 回歸 $$\ h(x) = \theta(s) $$。那麼可以比較一下每個方法的誤差如下圖


對於實際要用哪種分類，比較如下

|     | PLA | 線性回歸 | Logistic 回歸 |
| --- | --- | --- | --- |
| 優點 | 效能較高 | 最簡單的方法 | 簡單的方法 |
| 優點 | 保證線性可分 |  |  |
| 缺點 | 只能用於線性可分 | ys 數值很大時誤差很大 | -ys 大時誤差很大 |

## 4. Logistic 回歸的平方誤
為何 Logistic 回歸不使用平方誤?直接帶進去看看

$$\ \theta(s) = \theta(\sum_{i=0}^{n}w_{i}x_{i}), E(w) = \dfrac{1}{2} ( \hat{y} - \theta\sum_{i=0}^{n}w_{i}x_{i}))^2$$

若 $$\ \theta(w^Tx) = 1 $$ -> 接近目標。 $$\ \theta(w^Tx) = 0 $$ -> 遠離目標，不論是近或遠步伐都很小，但我們希望步伐大一點在遠離時，直到離目標近時才變小。

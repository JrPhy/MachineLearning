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
在此我們要最佳化 $$\ \theta(y_{i}w^Tx_{i}) $$，但是因為這

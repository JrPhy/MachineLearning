梯度下降 (Gradient descent) 法在在 AI 的領域中是很常使用的算法，核心概念就是直接解他的差分方程並得到最小值，當然也可以用 QR 分解來求得，且能夠得到較精確的最小值，但是就需要花較多的時間，梯度下降則是捨去精度來換時間。其中又有共軛(conjugate)梯度法、隨機(stochastic)梯度法與調步(Adaptive)梯度法。接著從最基本的再逐一介紹。

## 1. 數學理論
在最佳化問題中就是利用微分去找極值，在此我們要找的是極(最)小值，在數學上的條件為 $\nabla f(x) = 0$ 且 $\nabla^2 f(x) > 0$。而二階微分在數值求解通常會算很久，所以多數只用一階微分來判斷，所以有時候並不知道是否存在一個非邊界點的極(最)小值。例如 $x^3$ 就沒有除了邊界以外的極值， $x^2 - y^2$ 則是有鞍點(saddle point)。在此我們就介紹一些數值方法來找極(最)小值。

## 2. 梯度下降法
對於一個有 Global/Local minimum 的函數而言，我們可以由一階微分來做尋找，並直接帶入原函數就可以比較出結果。

#### 1. 一維
$$\ f(x) = 2x^2 + 4x + 3, f'(x) = 4x + 4 $$ -> $$\ f_{min}(x*) = (-1, 1) $$\
而電腦無法算微分，所以轉為插分方程可得\
$f'(x) = \dfrac{df(x)}{dx} \sim \dfrac{ \Delta f(x)}{ \Delta x} = \dfrac{df(x_2) - df(x_1)}{x_2 - x_1}$\
所以就直接將 x 帶入去找到 x*\
$x_{n+1} = x_n - \lambda f'(x_n)$\
其中 $\lambda$ 就是每一步要跨多大步，太大步有可能走不到最小值，但太小步又會跑太久，所以就有以上最後提到的方法。而實際問題是很多維的，所以只要分個別維度去做即可。

#### 2. 多維
假設函數 f(x, y) = z，則其梯度為

$$\ \nabla f(x, y) = \dfrac{\partial f(x, y)}{\partial x}\hat{x} + \dfrac{\partial f(x, y)}{\partial y}\hat{y} = f_{x}(x, y)\hat{x} + f_{y}(x, y)\hat{y}$$

$$ \begin{equation}
   \begin{Bmatrix} 
   x_{n+1} \\
   y_{n+1}
   \end{Bmatrix} 
   \quad
    = 
   \begin{Bmatrix} 
   x_{n} \\
   y_{n}
   \end{Bmatrix} 
   \quad
   -k
   \begin{Bmatrix} 
   f_{x} \\
   f_{y}
   \end{Bmatrix} 
   \quad
\end{equation}$$

如同做一維梯度下降兩次，多維的就做多次。

## 範例
$$\ z = f(x, y) = x^2 - 4x + 2 + y^2 - 4y + 2 $$，最小值為 x = 2, y = 2。可以使用我的程式碼，改變 $$\ x_{0}, y_{0} $$ 和 $$\ \lambda $$，並觀察迭代次數。當 $$\ \lambda $$ 越小，就需要越多次迭代，但也越接近正確答案。

## 使用限制
對於二維二次多項式有兩個最小值，要如何知道哪個才是全域最小值?答案是沒辦法，因為一次微分只能找到極值，且對於多變數多項式來說，可能還存在一個鞍點，當然有其他方法來避免這問題。

## 線性回歸的梯度下降
[其定義與概念可參考這篇](https://github.com/JrPhy/numerical/tree/master/least-square)，在此直接跳到使用梯度下降法。此問題至少有兩個變數 a, b，所以使用偏微分

$$\ \dfrac{\partial E}{\partial b} = \dfrac{1}{n}\sum_{i=1}^{n}(-2x_{i})(y_{i} - bx_{i} - a)$$

$$\ \dfrac{\partial E}{\partial a} = \dfrac{1}{n}\sum_{i=1}^{n}-2(y_{i} - bx_{i} - a)$$

使用梯度下降法即可將上式改為下式

$$\ b_{i+1} = b_{i} - \dfrac{\lambda}{n}\sum_{i=1}^{n}(-2x_{i})(y_{i} - bx_{i} - a)$$

$$\ a_{i+1} = a_{i} - \dfrac{\lambda}{n}\sum_{i=1}^{n}-2(y_{i} - bx_{i} - a)$$

大部分的步驟與線性回歸相同，只有誤差不同。假設我們想要對 m 次多項式回歸

$$\ y = \sum_{i=0}^{n}a_{i}x^i, E = (\sum_{j=1}^{n}y_{i} - \sum_{i=0}^{n}a_{i}x^i)^2 $$

$$\ \dfrac{\partial E}{\partial a_{i}} = \dfrac{\lambda}{n}(\sum_{j=1}^{n}2(y_{i} - \sum_{i=0}^{n}a_{i}x^i))^2$$

在此使用指標 k 來代表下次迭代

$$\ a_{2}^{k+1} = a_{2}^{k} - \dfrac{\lambda}{n}(\sum_{j=1}^{n}2x^2(y_{i} - \sum_{i=0}^{2}a_{i}^kx^i))^2$$

如果是四次式，就將第二個 summation 累加至 4。

## 自式應梯度下降
在隨機梯度下降學習率為定值，所以收斂速率跟學習率有關。在一開始學習率最大，隨著迭代學習率會一直變小，所以就能夠更快且更接近目標。其中一個方法就是用以下方式調整

$$\ w^{k+1} = w^k - \dfrac{\eta}{\sqrt(\sum_{i=1}^{k}g_{i}^2)} $$

當然也有其他方法如以下方程式

$$\ w^{k+1} = w^k - \dfrac{\eta^k}{\sigma^k}g^k, \eta^k = \dfrac{\eta}{\sqrt(k+1)}, g^k = \dfrac{\partial f(\theta^k)}{\partial w}, \sigma^k = \dfrac{1}{\sqrt(k+1)}{\sqrt(\sum_{i=1}^{k}g_{i}^2)}$$

對於二次方程式 $$\ f(x) = ax^2 + bx + x, a > 0 $$，x* = b/2a。假設初始值為 $$\ x_{0} $$，迭代次數與和 x* 的距離有關，越遠不乏就越大，一次差分也就越大。那麼對於高次方程式呢?考慮 f(x, y)，明顯可知

$$\ \dfrac{\partial f(x, y)}{\partial x} at...x_{0} > \dfrac{\partial f(x, y)}{\partial x} at...x_{1} $$ 

$$\ \dfrac{\partial f(x, y)}{\partial y} at...y_{0} > \dfrac{\partial f(x, y)}{\partial y} at...y_{1} $$ 

但是

$$\ \dfrac{\partial f(x, y)}{\partial y} at...y_{0} > \dfrac{\partial f(x, y)}{\partial x} at...x_{0}, y_{0} > x_{0} $$ 

所以我們需要除以二次微分才能得到到最小值的距離

$$\ |x_{0} + \dfrac{b}{2a}| = |\dfrac{2ax_{0} + b}{2a}| = |\dfrac{\dfrac{\partial f(x)}{\partial w} at...x_{0} }{\dfrac{\partial^2 f(x)}{\partial w^2} at...x_{0} }|$$ 

但是算二次微分需要花費很多時間，所以用一次微分的和來近似。

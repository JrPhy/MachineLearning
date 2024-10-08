PLA 是一個最簡單的二元分類法，也就是將物件標上 1 或 -1，然後找出一個方法來分辨。假設所有資料可以被線分開，目標就是找到該條線，如果不行就要用其他方法。
## 1. 分類方式
假設平面被直線 L: ax + by + c = 0 分成兩部分，A(x<sub>0</sub>, y<sub>0</sub>) 帶入 L 後若 ax<sub>0</sub> + by<sub>0</sub> + c > 0，則 A 在線的右邊，B(x<sub>1</sub>, y<sub>1</sub>) 帶入 L 後若 ax<sub>1</sub> + by<sub>1</sub> + c < 0，則 B 在線的左邊。所以可知若被標記為 1，則應該在線的右邊，反之則在左邊，所以資料集為 (x<sub>0</sub>, y<sub>0</sub>, 1), (x<sub>1</sub>, y<sub>1</sub>, -1)\

## 2. 矩陣表示
將分類方式用矩陣表達後，可以更方便的推廣的高維度的空間。在此有三個係數 a, b, c，將其放入矩陣 **w** 中，(x<sub>i</sub>, y<sub>i</sub>) 放入另一個矩陣 **x** 中，則 **w** = [a b c], **x** = [x<sub>i</sub>, y<sub>i</sub> 1] 

$$\begin{equation}
    \begin{bmatrix}
        w_{1} \\ w_{2} \\ w_{0}
    \end{bmatrix}
    \begin{bmatrix}
        x_{1} \\
        x_{2} \\
        x_{0} \\
    \end{bmatrix}
    = w_{1}x_{1} + w_{2}x_{2} + w_{0}x_{0} = 0
\end{equation}$$

在高維空間中就可寫成 

$$\ w_{n}x_{n} + w_{n-1}x_{n-1} + ... + w_{0}x_{0} = \sum_{j=i-1}^{n} w_{j}x_{j} + w_{0}x_{0} = \sum_{j=0}^{n} w_{j}x_{j} \ = <w, x^T>$$

w<sub>0</sub>x<sub>0</sub> 為閾值項，x<sub>0</sub> ≠ 0，w<sub>0</sub> 被初始為 0，並且在每一步去更新\
<w, x<sup>T</sup>> > 0 --> 1；<w, x<sup>T</sup>> < 0 --> -1；<w, x<sup>T</sup>> == label --> continue, <w, x<sup>T</sup>> != label --> w<sup>t+1</sup> = w<sup>t</sup> + x<sup>i</sup>y<sup>i</sup>

## 3. 範例
在此將 w[0] = c, w[1] = a, w[2] = b。假設座標平面上有四點 (0.3, 0.7, -1), (-0.4, -0.6, -1), (0.9, 0.2, -1), (-0.3, 0.7, 1)，做完第一步後可得
1. w[0] = -1, w[1] = -0.3, w[2] = -0.7
2. -0.3*(-0.4) + -0.7*(-0.6) + 1 > 0，與預測不同，故更新 **w**--> w[0] = 0, w[1] = -0.7, w[2] = -1.3
3. -0.7* 0.9 + -1.3*0.2 < 0，與預測相同，不更新 **w**
4. -0.7*(-0.3) + -1.3*0.7 < 0，與預測不同，故更新 **w**--> w[0] = 1, w[1] = -1, w[2] = 0.6

算出來的結果如下圖所示\
![img](https://github.com/JrPhy/MachineLearning/blob/master/Perceptron%20Learning%20Algorithm/PLA.gif)\
程式會算到所有分類為正確的，所以如果不是可線性分類，那就會陷入無窮回圈。如果丟入的資料順序不同，那就會得到不同的直線，道仍可將資料分開。所以在實務上我們可以允許有部分的資料是不可被分離的。

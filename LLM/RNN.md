在看文章時我們會規定文章的書寫順序，閱讀就需要依照書寫順序，例如英文跟中文的書寫順序就不同。在機器學習中我們會把讀到的物件依序丟進神經網路中，每次進入的權重都不相同，這種神經網路稱為循環神經網路 Recurrent Neural Networks(RNN)。

## 1. 循環神經網路 Recurrent Neural Networks (RNN)
![img](https://minio.cvmart.net/cvmart-community/images/202001/10/72/t068jrftYA.png)\
[來源](https://www.cvmart.net/community/detail/1360)。當 x<sub>0</sub> 丟進去後會得到一組 o<sub>0</sub> 與 w<sub>1</sub>，接著再把 x<sub>1</sub> 丟進去，得到 o<sub>1</sub> 與 w<sub>2</sub>，...。
![img](https://i-blog.csdnimg.cn/blog_migrate/92e4fde622794bf4c11c517096789eb4.png)\
[來源](https://blog.csdn.net/qq_36696494/article/details/89028956) o<sub>t</sub> = g(Vh<sub>t</sub>), h<sub>t</sub> = f(Ux<sub>t</sub> + Ws<sub>t-1</sub>)，可以一直展開寫成

o<sub>t</sub> = g(Vs<sub>t</sub>) = g(Vf(Ux<sub>t</sub> + Ws<sub>t-1</sub>)) = g(Vf(Ux<sub>t</sub> + Wf(Ux<sub>t-1</sub> + Ws<sub>t-2</sub>)) = g(Vf(Ux<sub>t</sub> + Wf(Ux<sub>t-1</sub> + Wf(Ux<sub>t-2</sub> + Ws<sub>t-3</sub>)) = ......

## 2. 隨時間的反向傳播 Back-Propagation Through Time (BPTT)
因為 RNN 處理的是時間序列，所以每一步都是與時間有關，不過計算方法與一般的反向傳播相同，在此有 U, V, W 三個參數要去更新。假設在 t 時刻的總輸入為 net<sub>t</sub> = Ux<sub>t</sub> + Ws<sub>t-1</sub>, s<sub>t</sub> = f(net<sub>t-1</sub>)，可以看出 s 只跟下一刻的輸出有關，所以對其他時候的偏微分皆為 0。接著要求誤差的最小值

$$\ \frac{\partial net_{t}}{\partial net_{t-1}} = \frac{\partial net_{t}}{\partial s_{t-1}}\frac{\partial s_{t-1}}{\partial net_{t-1}} = Wdiag[f'(net_{t-1})] $$

$$\begin{equation}
    \frac{\partial net_{t}}{\partial s_{t-1}} = 
    \begin{bmatrix}
        \frac{\partial net_{1}^{t}}{\partial s_{1}^{t-1}}&\frac{\partial net_{1}^{t}}{\partial s_{2}^{t-1}}&...&\frac{\partial net_{1}^{t}}{\partial s_{n}^{t-1}} \\
        \frac{\partial net_{2}^{t}}{\partial s_{1}^{t-1}}&\frac{\partial net_{2}^{t}}{\partial s_{2}^{t-1}}&...&\frac{\partial net_{2}^{t}}{\partial s_{n}^{t-1}} \\
        .&.&...&. \\
        \frac{\partial net_{n}^{t}}{\partial s_{1}^{t-1}}&\frac{\partial net_{n}^{t}}{\partial s_{2}^{t-1}}&...&\frac{\partial net_{n}^{t}}{\partial s_{n}^{t-1}} \\
    \end{bmatrix}
    =
    \begin{bmatrix}
        w_{11}&w_{12}&...&w_{1n} \\
        w_{21}&w_{22}&...&w_{2n} \\
        .&.&...&. \\
        w_{n1}&w_{n2}&...&w_{nn} \\
    \end{bmatrix}
    = W
\end{equation}$$

$$\begin{equation}
    \frac{\partial s_{t-1}}{\partial net_{t-1}} = 
    \begin{bmatrix}
        \frac{\partial s_{1}^{t-1}}{\partial net_{1}^{t-1}}&\frac{\partial s_{1}^{t-1}}{\partial net_{2}^{t-1}}&...&\frac{\partial s_{1}^{t-1}}{\partial net_{n}^{t-1}} \\
        \frac{\partial s_{2}^{t-1}}{\partial net_{1}^{t-1}}&\frac{\partial s_{2}^{t-1}}{\partial net_{2}^{t-1}}&...&\frac{\partial s_{2}^{t-1}}{\partial net_{n}^{t-1}} \\
        .&.&...&. \\
        \frac{\partial s_{n}^{t-1}}{\partial net_{1}^{t-1}}&\frac{\partial s_{n}^{t-1}}{\partial net_{2}^{t-1}}&...&\frac{\partial s_{n}^{t-1}}{\partial net_{n}^{t-1}} \\
    \end{bmatrix}
    =
    \begin{bmatrix}
        f'(net_{1}^{t-1})&0&...&0 \\
        0&f'(net_{2}^{t-1})&...&0 \\
        .&.&...&. \\
        0&0&...&f'(net_{n}^{t-1}) \\
    \end{bmatrix}
    = diag[f'(net_{n}^{t-1})]
\end{equation}$$

所以誤差為

$$\ \delta_{k}^{T} = \frac{\partial E}{\partial net_{k}} = \frac{\partial E}{\partial net_{t}}\frac{\partial net_{t}}{\partial net_{k}} = \frac{\partial E}{\partial net_{t}}\frac{\partial net_{t}}{\partial net_{t-1}}\frac{\partial net_{t-1}}{\partial net_{t-2}}...\frac{\partial net_{k+1}}{\partial net_{k}} $$

$$\ = Wdiag[f'(net_{t-1})]Wdiag[f'(net_{t-2})]...Wdiag[f'(net_{k})]\delta_{T}^{t} $$

$$\ = net_{l}^{t} = \delta_{T}^{t} \prod_{i=k}^{t-1}Wdiag[f'(net_{i})] $$

接著再往回一層一層傳過去

$$\ \delta_{T}^{t} = Ua_{t}^{l-1} + Ws_{t-1}, a_{t}^{l-1} = f^{l-1} (net_{t}^{l-1}) $$

$$\ net_{l}^{t} $$ 為第 l 層的神經元加權輸入， $$\ a_{t}^{l-1} $$ 為第 l-1 層的神經元輸出。f 為激勵函數。所以

$$\ (\delta_{l-1}^{t})^{T} = \frac{\partial E}{\partial net_{t}^{l-1}} = \frac{\partial E}{\partial net_{t}^{l}} \frac{\partial net_{t}^{l}}{\partial net_{t}^{l-1}} $$

$$\ = (\delta_{l}^{t})^{T} U diag[f'^{l-1}(net_{t}^{l-1})] $$

即是將誤差向傳到上一層的算法。最後在計算每個權重的梯度 

$$\begin{equation}
    \frac{\partial E}{\partial W} = \nabla_{W} E = 
    \begin{bmatrix}
        \delta_{1}^{t}s_{1}^{t-1}&\delta_{1}^{t}s_{2}^{t-1}&...&\delta_{1}^{t}s_{n}^{t-1} \\
        \delta_{2}^{t}s_{1}^{t-1}&\delta_{2}^{t}s_{2}^{t-1}&...&\delta_{2}^{t}s_{n}^{t-1} \\
        .&.&...&. \\
        \delta_{n}^{t}s_{1}^{t-1}&\delta_{n}^{t}s_{2}^{t-1}&...&\delta_{n}^{t}s_{n}^{t-1} \\
    \end{bmatrix}
\end{equation}$$

因為 net<sub>t</sub> = Ux<sub>t</sub> + Ws<sub>t-1</sub>，所以對 W 的偏為止與 s 有關，即可寫成

$$\ \frac{\partial E}{\partial w_{ij}} = \frac{\partial E}{\partial net_{j}^{t}} \frac{\partial net_{j}^{t}}{\partial w_{ij}} $$

所以最後的誤差即為每個時刻的和 $$\ \nabla_{W} E = \sum_{i=1}^{t} \nabla_{W_{i}} E $$

每丟一個新的輸入就會去更一次權重。這種方法有一個很明顯的缺點，就是一句話如果太長，那最一開始進來的文字就會被遺忘，在數學上就是因為一直做微分，如果是使用 tanh 或是 sigmoid，越前面的輸入到後面梯度就會消失。雖然此種行為很接近人的行為，但人們可以藉由某些方式來改進，而 RNN 的改進方式有以下三種
1. 初始化權重，不要取極大或極小值。
2. 使用 relu 代替 sigmoid 和 tanh 。
3. 使用其他結構的 RNNs，比如（LTSM）和 Gated Recurrent Unit（GRU）。

## 1. 長短期記憶模型 Long Short-Term Memory networks (LSTM)
為了要解決遺忘問題，LSTM 在模型中加入了「記憶」，會把最一開始的輸入一直傳進去跟後面的輸入做計算，且內部多了一個「遺忘門」與「更新門(與輸入門一起)」。下方符號皆為向量或矩陣\
![img](https://mlarchive.com/wp-content/uploads/2023/07/1_S0rXIeO_VoUVOyrYHckUWg.gif) [來源](https://mlarchive.com/deep-learning/understanding-long-short-term-memory-networks/)
1. 遺忘門 $$\ f_{t} = \sigma (W_{f}.[h_{t-1}, x_{t}] + b_{f}) $$
2. 輸入門 $$\ i_{t} = \sigma (W_{i}.[h_{t-1}, x_{t}] + b_{i}), \tilde{C_{t}} = tanh(W_{C}.[h_{t-1}, x_{t}] + b_{C}) $$
3. 更新層 $$\ C_{t} = f_{t}。C_{t-1} + i_{t}。\tilde{C_{t}} $$
4. 輸出門 $$\ o_{t} = \sigma (W_{o}[h_{t-1}, x_{t}] + b_{o}), h_{t} = o_{t}。tanh(C_{t}) $$

其中的 $$\ \tilde{C_{t}} $$ 是用來決定要更新哪些訊息用， $$\ \sigma $$ 為 sigmoid 函數，[] 符號為矩陣拼接，例如 $$\ h_{t-1} \in M_{128 x 64}(\mathbb{R}), x_{t} \in M_{128 x 16}(\mathbb{R}) $$，那麼拼接後的矩陣大小為 128 x 80。. 代表內積，。代表按元積，也就是相同位置的元素相乘，例如 a = [1, 2], b = [3, 4]，a。b = [3, 8]。

在看文章時我們會規定文章的書寫順序，閱讀就需要依照書寫順序，例如英文跟中文的書寫順序就不同。在機器學習中我們會把讀到的物件依序丟進神經網路中，每次進入的權重都不相同，這種神經網路稱為遞迴神經網路 Recurrent Neural Networks(RNN)。\
![img](https://media.geeksforgeeks.org/wp-content/uploads/20231204130857/neuron-200.png)\
[來源](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/)。當 x<sub>0</sub> 丟進去後會得到一組 h<sub>0</sub> 與 w<sub>1</sub>，接著再把 x<sub>1</sub> 丟進去，得到 h<sub>1</sub> 與 w<sub>2</sub>，...。
![img](https://i-blog.csdnimg.cn/blog_migrate/92e4fde622794bf4c11c517096789eb4.png)\
[來源](https://blog.csdn.net/qq_36696494/article/details/89028956) h = σ(UX + W<sub>h-1</sub> + B), Y = O(Vh + C) 

每丟一個新的輸入就會去更一次權重。這種方法有一個很明顯的缺點，就是一句話如果太長，那最一開始進來的文字就會被遺忘，會造成梯度消失或暴增。雖然此種行為很接近人的行為，但人們可以藉由某些方式來改進，而 RNN 的改進方式就是 LSTM。

## 1. 長短期記憶模型 Long Short-Term Memory networks (LSTM)
為了要解決遺忘問題，LSTM 在模型中加入了「記憶」，會把最一開始的輸入一直傳進去跟後面的輸入做計算，且內部多了一個「遺忘門」與「更新門(與輸入門一起)」。下方符號皆為向量或矩陣\
![img](https://mlarchive.com/wp-content/uploads/2023/07/1_S0rXIeO_VoUVOyrYHckUWg.gif) [來源](https://mlarchive.com/deep-learning/understanding-long-short-term-memory-networks/)
1. 遺忘門 $$\ f_{t} = \sigma (W_{f}.[h_{t-1}, x_{t}] + b_{f}) $$
2. 輸入門 $$\ i_{t} = \sigma (W_{i}.[h_{t-1}, x_{t}] + b_{i}), \tilde{C_{t}} = tanh(W_{C}.[h_{t-1}, x_{t}] + b_{C}) $$
3. 更新層 $$\ C_{t} = f_{t}。C_{t-1} + i_{t}。\tilde{C_{t}} $$
4. 輸出門 $$\ o_{t} = \sigma (W_{o}[h_{t-1}, x_{t}] + b_{o}), h_{t} = o_{t}。tanh(C_{t}) $$

其中的 $$\ \tilde{C_{t}} $$ 是用來決定要更新哪些訊息用， $$\ \sigma $$ 為 sigmoid 函數，[] 符號為矩陣拼接，例如 $$\ h_{t-1} \in M_{128 x 64}(\mathbb{R}), x_{t} \in M_{128 x 16}(\mathbb{R}) $$，那麼拼接後的矩陣大小為 128 x 80。. 代表內積，。代表按元積，也就是相同位置的元素相乘，例如 a = [1, 2], b = [3, 4]，a。b = [3, 8]。

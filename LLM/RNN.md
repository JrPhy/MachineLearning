在看文章時我們會規定文章的書寫順序，閱讀就需要依照書寫順序，例如英文跟中文的書寫順序就不同。在機器學習中我們會把讀到的物件依序丟進神經網路中，每次進入的權重都不相同，這種神經網路稱為遞迴神經網路 Recurrent Neural Networks(RNN)。\
![img](https://media.geeksforgeeks.org/wp-content/uploads/20231204130857/neuron-200.png)\
[來源](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/)。當 x<sub>0</sub> 丟進去後會得到一組 h<sub>0</sub> 與 w<sub>1</sub>，接著再把 x<sub>1</sub> 丟進去，得到 h<sub>1</sub> 與 w<sub>2</sub>，...。

h = σ(UX + W<sub>h-1<sub> + B), Y = O(Vh + C) 

每丟一個新的輸入就會去更一次權重。這種方法有一個很明顯的缺點，就是一句話如果太長，那最一開始進來的文字就會被遺忘，會造成梯度消失或暴增。雖然此種行為很接近人的行為，但人們可以藉由某些方式來改進，而 RNN 的改進方式就是 LSTM。

## 1. 長短期記憶模型 Long Short-Term Memory networks (LSTM)
為了要解決遺忘問題，LSTM 在模型中加入了「記憶」，會把最一開始的輸入一直傳進去跟後面的輸入做計算，且內部多了一個「遺忘門」
![img](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

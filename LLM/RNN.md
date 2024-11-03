在看文章時我們會規定文章的書寫順序，閱讀就需要依照書寫順序，例如英文跟中文的書寫順序就不同。在機器學習中我們會把讀到的物件依序丟進神經網路中，每次進入的權重都不相同，這種神經網路稱為遞迴神經網路 Recurrent Neural Networks(RNN)。\
![img](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)\
這種方法有一個很明顯的缺點，就是一句話如果太長，那最一開始進來的文字就會被遺忘，如此一來效果就不好。雖然此種行為很接近人的行為，但人們可以藉由某些方式來改進，而 RNN 的改進方式就是 LSTM。

## 1. 長短期記憶模型 Long Short Term Memory networks (LSTM)

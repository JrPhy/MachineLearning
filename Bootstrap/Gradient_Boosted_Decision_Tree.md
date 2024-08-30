目前我們已經知道 AdaBoost 跟隨機森林，AdaBoost 是一種弱根據的學習算法+最佳化重新調整權重+線性聚合 α。隨機森林則是藉由取後放回與決策樹來造出許多顆決策樹。現在要將這兩者結合起來，但是隨機森林只有聚合決策樹而沒有權重，所以我們需要有權重的決策樹。

## 1. AdaBoost 決策樹
AdaBoost 與決策樹都需要取後放回，在 BAGging 算法中，權重代表資料集中有多少份取後放回的樣本，而在隨機為底的算法中，權重代表比例，所以我們不需要修改決策樹算法，可以從取樣來得到權種資訊，所以此算法為 AdaBoost + 取樣 + 決策樹。在 AdaBoost 中，權重為 $$\ \alpha = ln\sqrt{\frac{1-t}{t}} $$，t 為權重的錯誤率。\
若樹為完滿的，那麼 E<sub>in</sub> = 0，所以 E<sup>u</sup><sub>in</sub> = 0 --> t = 0 --> α --> ∞\
所以我們不需要一顆完滿的樹，只需要用部分取樣來取代所有的資料，所以修改後的算法為\
AdaBoost + 取樣 + 剪枝後的決策樹\
剪枝是為了限制樹的高度。

## 2. AdaBoost 的權重
$$\ u_{n}^{t+1} = u_{n}^{t}\sqrt{\frac{1-t}{t}} 為正確，u_{n}^{t+1} = u_{n}^{t}\sqrt{\frac{t}{1-t}} 為錯誤$$

$$\ \alpha = ln\sqrt{\frac{1-t}{t}} \rightarrow \sqrt{\frac{1-t}{t}} = e^\alpha \rightarrow u_{n}^{t+1} = u_{n}^{t}\sqrt{\frac{1-t}{t}}^{-y_{n}\alpha_{i}g_{i}x_{n}}$$

$$\ u_{n}^{N+1} = u_{n}^{1}\prod_{i=1}^{N}exp(-y_{n}\alpha_{i}g_{i}x_{n})$$

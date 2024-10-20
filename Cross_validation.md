目前為止已經介紹很多機器學習中的算法，每種算法都可以產生很多模型，所以這邊要討論如何評價與選擇最好的模型。機器學習的步驟如下
1. 收集資料
2. 選擇算法與訓練模型
3. 實際預測

## 交叉驗證
在機器學習中我們關注兩種誤差，第一個是輸入資料的誤差，來自於收集資料所得到預測的模型。假設 f(x<sub>collected</sub>) 是從資料訓練得到的模型，y<sub>collected</sub> 是答案，則\
E<sub>in</sub> = || y<sub>collected</sub> - f(x<sub>collected</sub>) ||，或是其他形式\
另一個錯誤是 E<sub>out</sub>，來自於訓練模型的預測。假設 f(x<sub>predict</sub>) 是從資料訓練得到的模型，y<sub>predict</sub> 是答案，則\
E<sub>out</sub> = || y<sub>predict</sub> - f(x<sub>predict</sub>) ||，或是其他形式\
如果我們把所有資料用於訓練，那麼我們只能在實際預測中得到 E<sub>out</sub>，在實務上是非常危險的。所以我們將資料分成 k 分，每一份有 N 組，一組是拿來做驗證用，其他則是用來訓練。使用每一組訓練集來得到模型 f<sub>(N-1)</sub><sup>(i)</sup>，表示由第 N-1 組資料集得到的模型，然後用驗證集來得到誤差 E<sub>val</sub>(f<sub>(N-1)</sub><sup>(i)</sup>)，且希望 E<sub>val</sub> 為最小。所以使用 E<sub>val, min</sub> 的假設，且使用全部的資料訓練出新模型 f，然後就可能可以得到對於該問題的最佳模型，就可以進一步說服其他人此模型的強固性。其中每一份資料都來自於相同的資料，所以遵守相同的分布。

## 要分多少份
使用 C[f<sub>(N-1)</sub><sup>(i)</sup>] 來表示交叉驗證的誤差，平均為
 C<sub>avg</sub>，假設 C[f<sub>(N-1)</sub><sup>(i)</sup>] 是用 E<sub>X</sub>(C[f<sub>(N-1)</sub><sup>(i)</sup>])，C<sub>avg</sub> 也是，所以\
MSE(C<sub>avg</sub>) = E<sub>X</sub>[(C<sub>avg</sub> - E<sub>X</sub>(C[f<sub>(N)</sub>]))<sup>2</sup>] = Var<sub>X</sub>(C<sub>avg</sub>) + bias(C<sub>avg</sub>)<sup>2</sup>\
bias(C<sub>avg</sub>)<sup>2</sup> = E<sub>X</sub>(C<sub>avg</sub>) - E<sub>X</sub>(C[f<sub>(N)</sub>]) - 

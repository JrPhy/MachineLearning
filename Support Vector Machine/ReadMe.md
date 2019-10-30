SVM is a supervised learning method.

PLA
SVM is an algorithm bases on th PLA, and with some constrains, so the solution of PLA may be different by the same data with different order. PLA can be just apply the linear separable data with label +1 and -1, if not, it does not terminate.

Hard-margin SVM
So we add the constrains that the on the PLA, the constrain is finding a line(or hyperplane) on the +1 data, which is closest tothe -1 data L1, and so is -1 data, the line is L2, then find the maximum distance between L1 and L2. It can only separate linear separable data, so we can use some tricks or add some conditions.

Soft-margin SVM
If the data is not linear separable, then we accept some mistakes

Kernel trick
Kernel trick transforms data to higher dimension space, then find a hyperplane to separate in that space.

Regularize
If use the strong kernel trick, the model may be over-fitting, so use Regularization to avoid over-fitting

import pandas as pd
from math import log
from math import sqrt
pd.options.mode.chained_assignment = None


def quadratic(a, b, c): 
    if (b * b - 4 * a * c) < 0: 
        return 'None' 
    Delte = sqrt(b * b - 4 * a * c) 
    if Delte > 0: 
        x = (- b + Delte) / (2 * a) 
        y = (- b - Delte) / (2 * a) 
        return x, y 
    else: 
        x = (- b) / (2 * a) 
        return x, x

data = pd.read_csv('input_2.csv')
length = data.index.size
train = int(length*0.8) # training data length
test = length-train # test data length  

data_train = data[0:train]
data_test = data[train:length]

# priors of C1 and C2
print("priors of C1 and C2")
prior_c1 = (data_train[data_train['class'] == 1].feature_value.count()) / train
prior_c2 = (data_train[data_train['class'] == 2].feature_value.count()) / train
print(prior_c1)
print(prior_c2)

# estimate mean and var
m1 = data_train[(data_train['class'] == 1)].feature_value.mean()
var1 = data_train[(data_train['class'] == 1)].feature_value.var()
m2 = data_train[(data_train['class'] == 2)].feature_value.mean()
var2 = data_train[(data_train['class'] == 2)].feature_value.var()
print(m1, var1, m2, var2)
s1, s2 = sqrt(var1), sqrt(var2)

# find 
x1, x2 = quadratic((var1-var2)/(2*var1*var2), 
    (m1/var1-m2/var2), 
    (m2*m2/2/var2)-(m1*m1/2/var1)-log(s1)+log(s2)+log(prior_c1)-log(prior_c2) )

print("check points are")
print(x1, x2)

if (var1-var2)/(2*var1*var2) > 0:
    e1 = 2
    e2 = 1
else:
    e1 = 1
    e2 = 2 

data_test['estimate'] = 0
data_test.loc[(data_test['feature_value'] > x1) & (data_test['feature_value'] < x2), 'estimate'] = e1
data_test.loc[data_test['estimate'] == 0, 'estimate'] = e2

tp = data_test[(data_test['class'] == 1) & (data_test['estimate'] == 1)].estimate.count()
fp = data_test[(data_test['class'] == 2) & (data_test['estimate'] == 1)].estimate.count()
tn = data_test[(data_test['class'] == 2) & (data_test['estimate'] == 2)].estimate.count()
fn = data_test[(data_test['class'] == 1) & (data_test['estimate'] == 2)].estimate.count()
confusion_matrix = [[tn, fp], [fn, tp]]
print("confusion matrix:")
print(confusion_matrix)

accuracy = (tp+tn) / test
print(accuracy)
precision1 = tp / (tp+fp)
recall1 = tp / (tp+fn)
f1_1 = 2*precision1*recall1 / (precision1+recall1)

precision2 = tn / (tn+fn)
recall2 = tn / (tn+fp)
f1_2 = 2*precision2*recall2 / (precision2+recall2)

macro_p = (precision1+precision2) / 2
macro_c = (recall1+recall2) / 2
macro_f1 = 2*macro_p*macro_c / (macro_c+macro_p)

print(precision1, precision2, recall1, recall2, f1_1, f1_2, macro_f1)
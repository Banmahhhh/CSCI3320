import pandas as pd
# from numpy import log
pd.options.mode.chained_assignment = None

data = pd.read_csv('input_1.csv')
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

# estimate p1 and p2 with parametric estimation
print("estimate p1 and p2")
temp = data_train[(data_train['feature_value'] == 1) & (data_train['class'] == 1)]
p1 = temp.feature_value.count() / ((data_train[data_train['class'] == 1]).feature_value.count())
print(p1)

temp = data_train[(data_train['feature_value'] == 1) & (data_train['class'] == 2)]
p2 = temp.feature_value.count() / ((data_train[data_train['class'] == 2]).feature_value.count())
print(p2)

# discriminant function
e0, e1 = 0, 0
if (1-p1)*prior_c1 > (1-p2)*prior_c2:
    e0 = 1
else:
    e0 = 2
if p1*prior_c1 > p2*prior_c2:
    e1 = 1
else:
    e1 = 2
# print(e0, e1)
data_test['estimate'] = 0
data_test.loc[data_test['feature_value'] == 0, 'estimate'] = e0
data_test.loc[data_test['feature_value'] == 1, 'estimate'] = e1
# print(data_test)

# analysis estimator, class 1 as positive
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

f1 = (f1_1 + f1_2) / 2

print(precision1, precision2, recall1, recall2, f1_1, f1_2, f1)
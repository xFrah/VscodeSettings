import pickle
from matplotlib import pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
import numpy as np

# load model
model_allChest = pickle.load(open("model_allChest.pkl", "rb"))

# make predictions for test data
data = r"C:\Users\fdimo\Desktop\stress_affect_detection-master\Datasets\WESAD"
s14_file = pd.read_pickle(data + "\\S16\\S16.pkl")

# cax = s16_file["signal"]["chest"]["ACC"][0:, 0]
# cay = s16_file["signal"]["chest"]["ACC"][0:, 1]
# caz = s16_file["signal"]["chest"]["ACC"][0:, 2]
cecg = s14_file["signal"]["chest"]["ECG"][:, 0]
cemg = s14_file["signal"]["chest"]["EMG"][:, 0]
ceda = s14_file["signal"]["chest"]["EDA"][:, 0]
ctemp = s14_file["signal"]["chest"]["Temp"][:, 0]
cresp = s14_file["signal"]["chest"]["Resp"][:, 0]
label = s14_file["label"]

chest_list_all = [
    # cax,
    # cay,
    # caz,
    cecg,
    cemg,
    ceda,
    ctemp,
    cresp,
    label,
]
chest_length = all(
    [len(a) == len(b) for a, b in list(itertools.combinations(chest_list_all, 2))]
)
print(chest_length)
print(len(label))

# wax = s16_file["signal"]["wrist"]["ACC"][0:, 0]
# way = s16_file["signal"]["wrist"]["ACC"][0:, 1]
# waz = s16_file["signal"]["wrist"]["ACC"][0:, 2]
weda = s14_file["signal"]["wrist"]["EDA"][:, 0]
wtemp = s14_file["signal"]["wrist"]["TEMP"][:, 0]
wbvp = s14_file["signal"]["wrist"]["BVP"][:, 0]
label = s14_file["label"]

wrist_list = [
    # wax,
    # way,
    # waz,
    weda,
    wtemp,
    wbvp,
]
wrist_length = all(
    [len(a) == len(b) for a, b in list(itertools.combinations(wrist_list, 2))]
)


# wrist_ACC = [wax, way, waz]
# wrist_length_ACC = all(
#     [len(a) == len(b) for a, b in list(itertools.combinations(wrist_ACC, 2))]
# )
# print(len(wax), len(way), len(waz))
# print(len(weda), len(wtemp), len(wbvp))
chest = [
    # cax,
    # cay,
    # caz,
    cecg,
    cemg,
    ceda,
    ctemp,
    cresp,
    label,
]
ch_array = np.array(chest)
ch_array = ch_array.T  # transpose
Columns = [
    # "cax",
    # "cay",
    # "caz",
    "cecg",
    "cemg",
    "ceda",
    "ctemp",
    "cresp",
    "label",
]
ch_df = pd.DataFrame(ch_array, columns=Columns)

ch_df.isnull().sum()
ch_df.describe().T

ch_df.info()
ch_df["label"].value_counts()

ch_df[["label"]].plot(figsize=(15, 2))
plt.text(400000, 1.2, "Baseline"), plt.text(
    1550000, 2.2, '"Stress"', c="darkred"
), plt.text(2600000, 4.2, "Meditation1", c="darkblue")
plt.text(3000000, 3.3, "Amusement", c="darkgreen"), plt.text(
    3500000, 4.2, "Meditation2", c="darkblue"
)
print(len(label))

Group = ch_df.groupby("label")
u1 = Group.get_group(0)
baseline = Group.get_group(1)
stress = Group.get_group(2)
amusement = Group.get_group(3)
meditation = Group.get_group(4)
u5 = Group.get_group(5)  # in Yellow for timed questionnaire
u6 = Group.get_group(6)  # in Yellow for timed questionnaire
u7 = Group.get_group(7)  # in Yellow for timed questionnaire

loc0 = ch_df.loc[ch_df["label"] == 0]
loc1 = ch_df.loc[ch_df["label"] == 1]
loc2 = ch_df.loc[ch_df["label"] == 2]
loc3 = ch_df.loc[ch_df["label"] == 3]
loc4 = ch_df.loc[ch_df["label"] == 4]

ch_loc = pd.concat([loc0, loc1, loc2, loc3, loc4])

ch_loc.describe().T

y = ch_loc.label
x = ch_loc.drop("label", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
print(x_train.shape)
print(y_test.shape)
evalSet = [(x_train, y_train), (x_test, y_test)]

allChest_pred_proba = model_allChest.predict_proba(x_test)
allChest_pred = model_allChest.predict(x_test)
print(len(allChest_pred))
print(len(x_test))
with open("predictions2.csv", "w") as f:
    for i in range(len(allChest_pred)):
        stringa = (
            str(x_test.iloc[i]["cecg"])
            + ","
            + str(x_test.iloc[i]["ceda"])
            + ","
            + str(x_test.iloc[i]["ctemp"])
            + ","
            + str(x_test.iloc[i]["cresp"])
            + ","
            + str(allChest_pred[i])
            + ","
            + str(allChest_pred_proba[i][0])
            + ","
            + str(allChest_pred_proba[i][1])
            + ","
            + str(allChest_pred_proba[i][2])
            + ","
            + str(allChest_pred_proba[i][3])
            + ","
            + str(allChest_pred_proba[i][4])
            + "\n"
        )
        f.write(stringa)
        print(stringa)
# for j, i in enumerate(allChest_pred):
#     print(j, i)
allChest_report = classification_report(
    y_test, allChest_pred, labels=np.unique(allChest_pred), digits=4
)

print("----------------------------------------")
print(
    "Balanced Accuracy: {0:.4f}".format(balanced_accuracy_score(y_test, allChest_pred))
)
print("----------------------------------------")
print("------------ S16 All Chest Classification Report------------")
print(allChest_report)

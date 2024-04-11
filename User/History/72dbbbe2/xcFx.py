Group = ch_df.groupby('label')
u1 = Group.get_group(0)
baseline = Group.get_group(1)
stress = Group.get_group(2)
amusement = Group.get_group(3)
meditation = Group.get_group(4)
u5 = Group.get_group(5) # in Yellow for timed questionnaire
u6 = Group.get_group(6) # in Yellow for timed questionnaire
u7 = Group.get_group(7) # in Yellow for timed questionnaire

loc0 = ch_df.loc[ch_df['label'] == 0]
loc1 = ch_df.loc[ch_df['label'] == 1]
loc2 = ch_df.loc[ch_df['label'] == 2]
loc3 = ch_df.loc[ch_df['label'] == 3]
loc4 = ch_df.loc[ch_df['label'] == 4]

ch_loc = pd.concat([loc0, loc1, loc2, loc3, loc4])

ch_loc.describe().T

y = ch_loc.label
x = ch_loc.drop('label',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train.shape)
print(y_test.shape)
evalSet = [(x_train, y_train), (x_test, y_test)] 

plt.figure(figsize=(9,7))
sns.heatmap(x_train.corr(), annot=True, cmap=plt.cm.BuPu);

allChest = XGBClassifier(objective = 'multi:softmax',
                              tree_method = 'gpu_hist',
                              learning_rate = 0.1,
                              n_estimators = 300, 
                              # deterministic_histogram = 'false',
                              gradient_based = 0.1,
                              num_early_stopping_rounds = 20,
                              gamma = 3,
                              #seed = 35,
                              verbosity = 2) 

model_allChest = allChest.fit(x_train,y_train, 
                              eval_metric=['merror'], 
                              eval_set = evalSet) 
[0]	validation_0-merror:0.04298	validation_1-merro
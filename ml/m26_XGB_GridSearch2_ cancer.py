parameters = [
    {'n_estimators' : [100,200,300], 'learning_late' : [0.1,0.01],'max_depth' : [6,8,10,12]},
    {'n_estimators' : [100,200,300],'max_depth' : [6,8,10,12],'learning_late' : [0.1,0.01], 'colsample_bytree':[0.6,0.9,1]},
    {'n_estimators' : [100,200,300],'max_depth' : [6,8,10,12],'learning_late' : [0.1,0.01], 'colsample_bytree':[0.6,0.9,1], 'colsample_bylevel':[0.6,0.7,0.9]},

]
n_jobs =-1
 
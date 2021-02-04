'''
XGBoost의 하이퍼 파라미터

파라미터명[파이썬, 사이킷런] 
*()안은 기본값

[eta(0.3), learning rate(0.1)] : GBM에 학습율과 유사하고 일반적으로 0.01 ~ 0.2 값이 사용됨

[num_boost_around(10), nestimators(100)] : early stopping 기능. 학습오류가 감소하지 않으면 더 이상 부스팅을 진행하지 않고 종료

[min_child_weight(1)]: 과적합(overfitting)을 방지할 목적으로 사용, 너무 높은 값은 과소적합(underfitting)을 야기하기 때문에 CV를 사용해서 적절한 값이 제시되어야 한다.

[gamma(0), min_split_loss(0)] : 리프노드의 추가분할을 결정할 최소손실 감소값, 해당값보다 손실이 크게 감소할 때 분리, 값이 클수록 과적합 감소효과, 범위 0~무한대

[max_depth(6),(3)]: 너무 크면 과적합(보통 3-10 사이 값이 적용), 0을 지정하면 깊이의 제한이 없다.

[subsample(1)]: 데이터 샘플링 비율 지정(과적합 제어), 보통 0.5 ~ 1 사용됨.

[colsample_bytree (1)]: 트리 생성에 필요한 피처의 샘플링에 사용, 피처가 많을 때 과적합 조절에 사용. 보통 0.5 ~ 1 사용됨.

[lambda(1), reg_lambda(1)]: L2 Regularization 적용 값. 피처 개수가 많을 때 적용을 검토. 클수록 과적합 감소 효과. 그다지 많이 사용되고 있지는 않음.

[alpha(0), reg_alpha(0)]: L1 Regularization 적용. 차원이 높은 경우 알고리즘 속도를 높일 수 있음.

[scale_pos_weight(1)] : 클래스 불균형이 심한 경우 0보다 큰 값을 지정하여 효과를 볼 수 있음.

max_leaf_nodes: max_leaf_nodes 값이 설정되면 max_depth는 무시된다. 따라서 두값 중 하나를 사용한다.

max_delta_step (0)]: 일반적으로 잘 사용되지 않음.

colsample_bylevel (1)]: 각 level마다 샘플링 비율

<eval_metric: 평가 지표>
rmse: root mean square error
mae: mean absolute error
logloss: negative log-likelihood
error: Binary classification error rate (0.5 threshold)
merror: Multiclass classification error rate
mlogloss: Multiclass logloss
auc: Area under the curve
'''
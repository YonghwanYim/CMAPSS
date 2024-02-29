import pickle

# RL environment 구현.
# 데이터셋 정의부터, state 변경 등등..
# agent는 따로 어떻게 파일로 만들지도 생각하자.
# 알고리즘과는 분리해야 함. value-based, policy-based 모두 적용할 수 있도록.

# Load average_by_loss_dfs from the file
with open('average_by_loss_dfs.pkl', 'rb') as f:
    average_by_loss_dfs = pickle.load(f)

print(average_by_loss_dfs)
# 이 데이터를 가지고 plot을 그리는 method는 따로 만들어야함.
from simulation_env import SimulationEnvironment

num_dataset = 1                # number of dataset (1, 2, 3, 4)
split_unit_number = 70         # The boundary value of unit numbers for dividing the train and test datasets
num_sample_datasets = 20       # Number of sample datasets
observation_probability = 0.1  # Randomly extract only a subset of observational probability data from the entire dataset

# class 인스턴스 생성
env = SimulationEnvironment()


# dataset 분할
train_data, test_data, full_data = env.load_data(num_dataset,split_unit_number)

# 샘플링 수행
sampled_datasets = env.sampling_datasets(num_sample_datasets, observation_probability, train_data, test_data, full_data)

add

"""
# 결과 출력
for idx, (sampled_train, sampled_test, sampled_full) in enumerate(sampled_datasets):
    print(f"Sampled Dataset {idx+1}:")
    print("Sampled Train Dataset:")
    print(sampled_train)
    print("\nSampled Validation Dataset:")
    print(sampled_test)
    print("\nSampled Full Dataset:")
    print(sampled_full)
    print("\n")
"""

print(sampled_datasets[0][0])
print(sampled_datasets[0][1])
print(sampled_datasets[0][2])
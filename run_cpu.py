from simulation_env import SimulationEnvironment
import configparser

# generate configuration instance
config = configparser.ConfigParser()

class RunCPU():
    def __init__(self, config_path):
        config.read(config_path)
        # number of dataset (1, 2, 3, 4)
        self.num_dataset = int(config['SimulationSettings']['num_dataset'])
        # The boundary value of unit numbers for dividing the train and test datasets
        self.split_unit_number = int(config['SimulationSettings']['split_unit_number'])
        # Number of sample datasets
        self.num_sample_datasets = int(config['SimulationSettings']['num_sample_datasets'])
        # Randomly extract only a subset of observational probability data from the entire dataset
        self.observation_probability = float(config['SimulationSettings']['observation_probability'])
        # Constant for crucial_moment loss
        self.crucial_moment = int(config['SimulationSettings']['crucial_moment'])

        # class instance 생성
        self.env = SimulationEnvironment()
        # dataset 분할
        self.train_data, self.test_data, self.full_data = self.env.load_data(self.num_dataset, self.split_unit_number)
        # sampling
        self.sampled_datasets = self.env.sampling_datasets(self.num_sample_datasets, self.observation_probability,
                                                      self.train_data, self.test_data, self.full_data)
        # sampled_datasets에 RUL column 추가.
        self.sampled_datasets_with_RUL = self.env.add_RUL_column_to_sampled_datasets(self.sampled_datasets)

    def run(self, data_sample_index):
        print('run')
        train_data = self.sampled_datasets_with_RUL[data_sample_index][0].copy()
        test_data = self.sampled_datasets_with_RUL[data_sample_index][1].copy()
        full_data = self.sampled_datasets_with_RUL[data_sample_index][2].copy()

    def run_many(self):
        # run()을 data sample만큼 반복수행.
        # 나중에 코드 추가하자.
        print("test")





runCPU = RunCPU('config1.ini')
# test
runCPU.run_many()



from FeatureCloud.app.engine.app import AppState, app_state
from FeatureCloud.app.engine.app import LogLevel 
from FeatureCloud.app.engine.app import Role
import numpy as np

import pandas as pd
from sklearn.linear_model import LinearRegression
from enum import Enum


df_train = pd.DataFrame()
df_test = pd.DataFrame()


class States(Enum):
    initial = 'initial'
    distribute_data  = 'distribute_data'
    receive_data = 'receive_data'
    compute = 'compute'
    #send_compute_results = 'send_compute_results'
    agg_results = 'agg_results'
    receive_aggregation = 'receive_aggregation'
    terminal = 'terminal'


# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
@app_state(States.initial.value, Role.BOTH)
class InitialState(AppState):

    def register(self):
        self.register_transition(States.distribute_data.value, role=Role.COORDINATOR)  # We declare that 'terminal' state is accessible from the 'initial' state.
        self.register_transition(States.receive_data.value, role=Role.PARTICIPANT)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        self.log(f'Starting Initialization for node {self.id} ...')
        if self.is_coordinator:
            return States.distribute_data.value
        else:
            return States.receive_data.value

@app_state(States.distribute_data.value, Role.COORDINATOR)
class DistributeDataState(AppState):

    def register(self):
        self.register_transition(States.receive_data.value, role=Role.COORDINATOR)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        self.log(f'{self.id} Split data for {len(self.clients)} clients ...')

        self.log('Distribute initial model parameters ...')
        #self.send_data_to_participant(np.random.random(10), self.clients)
        self.broadcast_data((0, np.random.random(1), np.random.random(10)))

        return States.receive_data.value  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.


@app_state(States.receive_data.value, Role.BOTH)
class ReceiveDataState(AppState):

    def register(self):
        self.register_transition(States.compute.value, role=Role.BOTH)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        self.log(f'{self.id} Loading data ...')
        # Load the client data and split it into training and test set
        df = pd.read_csv('/mnt/input/data.csv')
        df_train = df.sample(frac=0.8)
        df_test = df.drop(df_train.index)

        self.store('train', df_train)
        self.store('test', df_test)
        
        self.log(f'{self.id} loaded data {df.shape} ...')

        return States.compute.value  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.



@app_state(States.compute.value, Role.BOTH)
class ComputeState(AppState):

    def register(self):
        self.register_transition(States.agg_results.value, role=Role.COORDINATOR)  # We declare that 'terminal' state is accessible from the 'initial' state.
        self.register_transition(States.compute.value, role=Role.PARTICIPANT)  # We declare that 'terminal' state is accessible from the 'initial' state.
        self.register_transition(States.terminal.value, role=Role.BOTH)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):

        self.log(f'{self.id} performing compute ...')
        df_train = self.load('train')
        df_test = self.load('test')
        self.log(f'Received as {self.id} the data {df_train.shape}')

        iter, intercept, coef = self.await_data()
        self.log(f'Iter[{iter}] Received newest coefficients ... {intercept} {coef}')
        self.log(f'{self.id}[{"Coordinator" if self.is_coordinator else "Participants"}]: training model ...')

        # Create a model with the latest model parameters
        reg = LinearRegression()
        reg.coef_ = coef
        reg.intercept_ = intercept

        if iter == -1:
            test_score = reg.score(df_test.drop(columns='target'), df_test['target'])
            self.log(f'{self.id} Final model score {test_score} ...')
            return States.terminal.value
        else:
            self.log(f'Training local model one more time ...')
            reg = LinearRegression().fit(df_train.drop(columns='target'), df_train['target'])
            test_score = reg.score(df_test.drop(columns='target'), df_test['target'])
            self.log(f'test score: {test_score}')

            self.log(f'Send to coordinator updated coefficients')
            self.send_data_to_coordinator((reg.intercept_, reg.coef_))

        if self.is_coordinator:
            return States.agg_results.value
        else:
            return States.compute.value  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.



@app_state(States.agg_results.value, Role.COORDINATOR)
class AggregationState(AppState):

    def register(self):
        self.register_transition(States.compute.value, role=Role.COORDINATOR)  # We declare that 'terminal' state is accessible from the 'initial' state.
        self.iteration = 0
        self.store('iteration', self.iteration)

    def run(self):
        self.iteration = self.load('iteration')
        #data = self.gather_data(2, is_json=False)
        self.log(f'{self.id}[{"Coordinator" if self.is_coordinator else "Participants"}, Iteration {self.iteration}')
        data = self.await_data(len(self.clients), is_json=False)
        agg_data = np.mean(data, axis=0)
        self.log(f'Aggregation data received {data}')
        self.log(f'Result {agg_data}')

        self.iteration+=1
        self.store('iteration', self.iteration)

        # Stop the process after 3 iterations
        if self.iteration >= 3:
            self.iteration =-1
        
        self.log(f'{self.id} send results back ...')
        self.broadcast_data((self.iteration, agg_data[0], agg_data[1]))
        return States.compute.value  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.



@app_state(States.receive_aggregation.value, Role.BOTH)
class RecvAggregationState(AppState):

    def register(self):
        self.register_transition(States.terminal.value, role=Role.BOTH)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        data = self.await_data(1)
        self.log(f'Received results {data}')
        return States.terminal.value  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.



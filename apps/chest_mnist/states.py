from FeatureCloud.app.engine.app import AppState, app_state
from FeatureCloud.app.engine.app import Role
import numpy as np

import pandas as pd
from enum import Enum

import torch
from torch.utils.data import Dataset

from model import ModelTraining
from functools import partial

df_train = pd.DataFrame()
df_test = pd.DataFrame()

def log(obj : AppState, msg : str):
    obj.log(f'{obj.id}/{"Coordinator" if obj.is_coordinator else "Participants"}: {msg}')

class States(Enum):
    initial = 'initial'
    distribute_initial_model  = 'distribute_initial_model'
    receive_data = 'receive_data'
    compute = 'compute'
    #send_compute_results = 'send_compute_results'
    aggregate_models = 'aggregate_models'
    #receive_aggregation = 'receive_aggregation'
    terminal = 'terminal'


# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
@app_state(States.initial.value, Role.BOTH)
class InitialState(AppState):

    def register(self):
        #self.register_transition(States.distribute_initial_model.value, role=Role.COORDINATOR)  # We declare that 'terminal' state is accessible from the 'initial' state.
        self.register_transition(States.compute.value, role=Role.BOTH)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        self.log(f'Starting Initialization for node {self.id} ...')

        md = ModelTraining()
        self.store('md', md)

        if self.is_coordinator:
            W = md.get_weights()
            self.broadcast_data((0, W))
            #return States.distribute_initial_model.value
            #return States.receive_data.value
        
        return States.compute.value

"""
@app_state(States.distribute_initial_model.value, Role.COORDINATOR)
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
        return States.compute.value  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.

"""

@app_state(States.compute.value, Role.BOTH)
class ComputeState(AppState):

    def register(self):
        self.register_transition(States.aggregate_models.value, role=Role.COORDINATOR)  # We declare that 'terminal' state is accessible from the 'initial' state.
        self.register_transition(States.compute.value, role=Role.PARTICIPANT)  # We declare that 'terminal' state is accessible from the 'initial' state.
        self.register_transition(States.terminal.value, role=Role.BOTH)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def mylog(self, msg):
       self.log(f'{self.id}/{"Coordinator" if self.is_coordinator else "Participants"}: {msg}')

    def run(self):

        log(self, "Performing compute ...")
        md = self.load('md')

        log(self, "Waiting for weights ...")
        iter, params = self.await_data()

        if iter == -1:
            log(self, f'Done!')

            if self.is_coordinator:
                out_path = '/mnt/output/global_model.pth'
                log(self, f'Storing global model to {out_path} ...')
                md.save_model(out_path)
            
            return States.terminal.value
        else:
            log(self, f'Training local model one more time ...')
            #lens = [x.size for x in params]
            #log(self, f'Param lens: {lens}')
            md.set_weights(params)
            md.train_single_epoch()
            #auc = md.get_test_score()
            #log(self, f'Local model performance AUC {auc}. Send model to coordinator')
            p, r, f1 = md.get_test_score()
            log(self, f'Local model performance precision: {p}, recall: {r}, f1: {f1}. Send model to coordinator')
            self.send_data_to_coordinator(md.get_weights())

        if self.is_coordinator:
            return States.aggregate_models.value
        else:
            return States.compute.value  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.


@app_state(States.aggregate_models.value, Role.COORDINATOR)
class AggregationState(AppState):

    def register(self):
        self.register_transition(States.compute.value, role=Role.COORDINATOR)  # We declare that 'terminal' state is accessible from the 'initial' state.
        iteration = 0
        self.store('iteration', iteration)

    def run(self):

        iteration = self.load('iteration')

        log(self, f'Starting model weight aggregation for iteration {iteration}')
        weights = self.await_data(len(self.clients), is_json=False)
        log(self, f'Got {len(weights)} model weights. Averaging them ...')
        agg_weights = np.mean(weights, axis=0)

        log(self, f'Evaluate global model ...')
        md = self.load('md')
        md.set_weights(agg_weights)
        p, r, f1 = md.get_test_score()
        log(self, f'[Iteration {iteration}] Global model performance precision: {p}, recall: {r}, f1: {f1}')

        iteration+=1
        self.store('iteration', iteration)

        # Stop the process after 3 iterations
        if iteration >= 3:
            iteration =-1

        log(self, f'Send results back ...')
        self.broadcast_data((iteration, agg_weights))
        return States.compute.value  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.



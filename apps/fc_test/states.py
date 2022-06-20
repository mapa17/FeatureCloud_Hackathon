from FeatureCloud.app.engine.app import AppState, app_state
from FeatureCloud.app.engine.app import LogLevel 
from FeatureCloud.app.engine.app import Role
import numpy as np


# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
@app_state('initial', Role.BOTH)
class InitialState(AppState):

    def register(self):
        self.register_transition('execution', role=Role.BOTH)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        self.log('Starting Initialization for node {self.id} ...')
        return 'execution'  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.

@app_state('execution', Role.BOTH)
class ExchangeState(AppState):

    def register(self):
        self.register_transition('terminal', role=Role.PARTICIPANT)  # We declare that 'terminal' state is accessible from the 'initial' state.
        self.register_transition('aggregation', role=Role.COORDINATOR)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        data = np.random.random(10)
        self.log(f'Starting execution on {self.id} with data {data}')

        self.send_data_to_coordinator(data)
        if self.is_coordinator:
            return 'aggregation'
        else:
            return 'terminal'  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.

@app_state('aggregation', Role.COORDINATOR)
class AggregationState(AppState):

    def register(self):
        self.register_transition('terminal', role=Role.COORDINATOR)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        data = self.gather_data(is_json=False)
        agg_data = np.mean(data, axis=0)
        self.log(f'Aggregation data received {data}')
        self.log(f'Result {agg_data}')
        return 'terminal'  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.

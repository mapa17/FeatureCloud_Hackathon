from model import ModelTraining, ChestMNIST
import numpy as np

md = ModelTraining(x_train_path='/Users/manuel.pasieka/Documents/projects/FeatureCloud_Hackathon/controler_workspace/data/input/client_1/x_train.npy', x_test_path='/Users/manuel.pasieka/Documents/projects/FeatureCloud_Hackathon/controler_workspace/data/input/client_1/x_test.npy', y_train_path='/Users/manuel.pasieka/Documents/projects/FeatureCloud_Hackathon/controler_workspace/data/input/client_1/y_train.npy', y_test_path='/Users/manuel.pasieka/Documents/projects/FeatureCloud_Hackathon/controler_workspace/data/input/client_1/y_test.npy', x_val_path='/Users/manuel.pasieka/Documents/projects/FeatureCloud_Hackathon/controler_workspace/data/input/client_1/x_val.npy', y_val_path='/Users/manuel.pasieka/Documents/projects/FeatureCloud_Hackathon/controler_workspace/data/input/client_1/y_val.npy', device_str='mps')

overfitting_cnt = 0
best_val_loss = np.inf
losses = []
best_weights = None

for i in range(100):
    print(f'Training iteration {i} ...')
    t_loss, v_loss = md.train_single_epoch(print)
    losses.append((t_loss, v_loss))
    print(f'{i} Losses: {t_loss}, {v_loss}')

    if v_loss > best_val_loss:
        overfitting_cnt+=1
    else:
        overfitting_cnt = 0
        best_val_loss = v_loss
        best_weights = md.get_weights()
    
    if overfitting_cnt >= 3:
        print('Stopping ...')
        break

print(f'Restore best weights ...')
md.set_weights(best_weights)

p, r, f1, s, l = md.get_test_score(print)
print(f'Average Precision: {np.mean(p, axis=0)}')
print(f'Average Recall: {np.mean(r, axis=0)}')

import numpy as np
import os

data = np.load('chestmnist.npz')

print(data.files)


x_train = data.f.train_images
x_val = data.f.val_images
x_test = data.f.test_images

y_train = data.f.train_labels
y_val = data.f.val_labels
y_test = data.f.val_labels

print(x_train.shape)
print(y_train[0])

np.random.seed(0)
x_train_sample = x_train[np.random.randint(x_train.shape[0], size=100), :]
x_val_sample = x_val[np.random.randint(x_val.shape[0], size=20), :]
x_test_sample = x_test[np.random.randint(x_test.shape[0], size=20), :]

y_train_sample = y_train[np.random.randint(y_train.shape[0], size=100), :]
y_val_sample = y_val[np.random.randint(y_val.shape[0], size=20), :]
y_test_sample = y_test[np.random.randint(y_test.shape[0], size=20), :]

print(x_train_sample.shape)
print(x_val_sample.shape)
print(x_test_sample.shape)
print(y_train_sample.shape)
print(y_val_sample.shape)
print(y_test_sample.shape)

np.save('x_train_sample', x_train_sample)
np.save('x_val_sample', x_val_sample)
np.save('x_test_sample', x_test_sample)
np.save('y_train_sample', y_train_sample)
np.save('y_val_sample', y_val_sample)
np.save('y_test_sample', y_test_sample)


def fractionate(n, fractions):
    
    result = []
    for fraction in fractions[:-1]:
        result.append(round(fraction * n))
    result.append(n - np.sum(result))
    
    return result

#split data for multiple clients
def client_split(x, y, fractions):
    
    m = x.shape[0]
    np.random.seed(0)
    permutation = np.random.permutation(m)
    shuffled_X = x[permutation,:]
    shuffled_Y = y[permutation,:]
    
    samples = fractionate (m, fractions)
    arr = np.cumsum(samples)
    
    x_clients = np.array_split(shuffled_X, arr)
    y_clients = np.array_split(shuffled_Y, arr)
    
    return x_clients, y_clients



def save_client_data(x, y, num_clients, fractions, data):
    clients_x, clients_y = client_split(x, y, fractions)
    for i in range(num_clients):
        np.save(os.path.join('client '+str(i+1),'client_'+str(i+1)+'_x_'+data+'.npy'), clients_x[i])
        np.save(os.path.join('client '+str(i+1),'client_'+str(i+1)+'_y_'+data+'.npy'), clients_y[i])
        
    
num_clients = 2     
client_fractions = [0.5, 0.5]

#save clients data (samples)


save_client_data(x_train_sample, y_train_sample, num_clients, client_fractions, 'train')
save_client_data(x_val_sample, y_val_sample, num_clients, client_fractions, 'val')

#save clients data
save_client_data(x_train, y_train, num_clients, client_fractions, 'train')
save_client_data(x_val, y_val, num_clients, client_fractions, 'val')






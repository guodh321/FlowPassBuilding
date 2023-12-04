import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{0}"


gridsize = 1024
foldername = 'InterpolatedResult{}'.format(gridsize)


samples = []
for i in range(501):
    s = np.load('/home/dg321/gitTest/PRI/irp/Flow_Data/'+ foldername + '/FpB_Interpolated_t{}_Velocity_{}_{}.npy'.format(i, gridsize, gridsize))
    samples.append(s)
    print(s.shape)

sampels_stacked = np.stack(samples)
print(sampels_stacked.shape)


np.save('/home/dg321/gitTest/PRI/irp/Flow_Data/'+ foldername + '/FpB_Interpolated_Velocity_{}_{}.npy'.format(gridsize, gridsize), sampels_stacked)



# combine all x and y
sample_Xs = []
sample_Ys = []

for i in range(501):
    xs = np.rot90(samples[i][0,:,:], 1)
    ys = np.rot90(samples[i][1,:,:])
    sample_Xs.append(xs)
    sample_Ys.append(ys)

sampelXs_stacked = np.stack(sample_Xs)
sampelYs_stacked = np.stack(sample_Ys)
print(sampelXs_stacked.shape)
print(sampelYs_stacked.shape)


# save 
np.save("/home/dg321/gitTest/PRI/irp/Flow_Data/"+ foldername + "/sampelXs_stacked.npy", sampelXs_stacked)
np.save("/home/dg321/gitTest/PRI/irp/Flow_Data/"+ foldername + "/sampelYs_stacked.npy", sampelYs_stacked)


# Train and test split and normalization ... to get latent variable finally
sampleXs_train450 = sampelXs_stacked[:450]
print(sampleXs_train450.shape)

sampleYs_train450 = sampelYs_stacked[:450]
print(sampleYs_train450.shape)

scalerX = MinMaxScaler(feature_range=(-1, 1))

sampleXs_train450_normalized = scalerX.fit_transform(sampleXs_train450.reshape((sampleXs_train450.shape[0], sampleXs_train450.shape[1]*sampleXs_train450.shape[1])))
sampleXs_train450_normalized = sampleXs_train450_normalized.reshape((sampleXs_train450.shape[0], sampleXs_train450.shape[1], sampleXs_train450.shape[1]))

print(sampleXs_train450_normalized.shape)

# Save the scaler
scaler_filename = "/home/dg321/gitTest/PRI/irp/Flow_Data/"+ foldername + "/train450_scalerX-1_1.save"
joblib.dump(scalerX, scaler_filename)


scalerY = MinMaxScaler(feature_range=(-1, 1))

sampleYs_train450_normalized = scalerY.fit_transform(sampleYs_train450.reshape((sampleYs_train450.shape[0], sampleYs_train450.shape[1]*sampleYs_train450.shape[1])))
sampleYs_train450_normalized = sampleYs_train450_normalized.reshape((sampleYs_train450.shape[0], sampleYs_train450.shape[1], sampleYs_train450.shape[1]))

print(sampleYs_train450_normalized.shape)


# Save the scaler
scaler_filename = "/home/dg321/gitTest/PRI/irp/Flow_Data/"+ foldername + "/train450_scalerY-1_1.save"
joblib.dump(scalerY, scaler_filename)

np.save("/home/dg321/gitTest/PRI/irp/Flow_Data/"+ foldername + "/sampleXs_train450_normalized.npy", sampleXs_train450_normalized)
np.save("/home/dg321/gitTest/PRI/irp/Flow_Data/"+ foldername + "/sampleYs_train450_normalized.npy", sampleYs_train450_normalized)


### Test
sampleXs_test = sampelXs_stacked[450:]
print(sampleXs_test.shape)
sampleYs_test = sampelYs_stacked[450:]
print(sampleYs_test.shape)


sampleXs_test_normalized = scalerX.transform(sampleXs_test.reshape((sampleXs_test.shape[0], sampleXs_test.shape[1]*sampleXs_test.shape[1])))
sampleXs_test_normalized = sampleXs_test_normalized.reshape((sampleXs_test.shape[0], sampleXs_test.shape[1], sampleXs_test.shape[1]))

print(sampleXs_test_normalized.shape)


sampleYs_test_normalized = scalerY.transform(sampleYs_test.reshape((sampleYs_test.shape[0], sampleYs_test.shape[1]*sampleYs_test.shape[1])))
sampleYs_test_normalized = sampleYs_test_normalized.reshape((sampleYs_test.shape[0], sampleYs_test.shape[1], sampleYs_test.shape[1]))

print(sampleYs_test_normalized.shape)


np.save("/home/dg321/gitTest/PRI/irp/Flow_Data/"+ foldername + "/sampleXs_test_normalized.npy", sampleXs_test_normalized)
np.save("/home/dg321/gitTest/PRI/irp/Flow_Data/"+ foldername + "/sampleYs_test_normalized.npy", sampleYs_test_normalized)

# Add a new axis for the channel
sampleXs_test_normalized = np.expand_dims(sampleXs_test_normalized, axis=-1)
sampleYs_test_normalized = np.expand_dims(sampleYs_test_normalized, axis=-1)

# Concatenate
concatenated_data_test51 = np.concatenate((sampleXs_test_normalized, sampleYs_test_normalized), axis=-1)

print(concatenated_data_test51.shape)

np.save("/home/dg321/gitTest/PRI/irp/Flow_Data/"+ foldername + "/concatenated_data_test51.npy", concatenated_data_test51)

# Add a new axis for the channel
sampleXs_train450_normalized = np.expand_dims(sampleXs_train450_normalized, axis=-1)
sampleYs_train450_normalized = np.expand_dims(sampleYs_train450_normalized, axis=-1)

# Concatenate
concatenated_data_train450 = np.concatenate((sampleXs_train450_normalized, sampleYs_train450_normalized), axis=-1)

print(concatenated_data_train450.shape)

np.save("/home/dg321/gitTest/PRI/irp/Flow_Data/"+ foldername + "/concatenated_data_train450.npy", concatenated_data_train450)
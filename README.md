# For 10K samples, typical is 3-5 epochs
num_samples = 10_000
batch_size = 128
steps_per_epoch = num_samples // batch_size  # = 78 steps

# Train for 3-5 epochs
max_steps = steps_per_epoch * 3  # = 234 steps (minimum)
# or
num_train_epochs = 3  # Let trainer calculate steps
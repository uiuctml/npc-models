import logger

data_loader_batch_size = 20
data_loader_worker_count = data_loader_batch_size
data_loader_shuffle = True

dataset_dir = "../../mapillary-dataset"
dataset_dir_images = dataset_dir + "/images" + "/sliced" + "/original"
dataset_split_percentage_train = 0.8
dataset_split_percentage_validation = 1 - dataset_split_percentage_train
dataset_view_row_count = 5

learning_rate_scheduler_mode = "min"
learning_rate_scheduler_factor = 0.1
learning_rate_scheduler_patient = 2
learning_rate_scheduler_threshold = 1e-4
learning_rate_scheduler_threshold_mode = "abs"
learning_rate_scheduler_cooldown = 0
learning_rate_scheduler_min_learning_rate = 1e-6
learning_rate_scheduler_min_learning_rate_decay = 1e-8

learning_rate_scheduler_gamma = 0.1
learning_rate_scheduler_step_size = 7

log_level = logger.LogLevel.info

model_epochs = 25
model_input_height = 224
model_input_width = model_input_height
model_input_channels = 3
model_pretrained_weights = "IMAGENET1K_V2"
model_file_name = "model.pth"

optimizer_learning_rate = 0.001
optimizer_momentum = 0.9


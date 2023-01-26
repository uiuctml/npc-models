import logger

dataset_dir = "../../mapillary-dataset"
dataset_dir_images = dataset_dir + "/images" + "/sliced" + "/original"
dataset_split_percentage_train = 0.8
dataset_split_percentage_validation = 1 - dataset_split_percentage_train

log_level = logger.LogLevel.trace
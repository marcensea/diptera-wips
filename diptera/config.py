# Image parameters
image_height = 116
image_width = 256
channels = 3  # RGB

# Preprocessing parameters
PREPROCESS_BATCH_SIZE = 256

# Training parameters
LEARNING_RATE = 0.1
BATCH_SIZE = 32
EPOCHS = 100
START_EPOCH = 0
FOLDS = 5
START_FOLD = 0
SEED = 2023

dataset = "spp68"
model_name = "mobilenet-9"

# Data augmentation parameters
augmentations = dict(
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
    zoom_range=0.025,
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True,  # randomly flip images
    fill_mode='nearest',  # points outside the boundaries
)



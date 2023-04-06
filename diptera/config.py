# Image parameters
image_height = 116
image_width = 256
channels = 3  # RGB

# Preprocessing parameters
PREPROCESS_BATCH_SIZE = 256

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
START_EPOCH = 0
FOLDS = 5
SEED = 2023

dataset = "spp67"
model_name = "mbn9"

# Data augmentation parameters
augmentations = dict(
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
    zoom_range=0.025,
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True,  # randomly flip images
    fill_mode='nearest'  # points outside the boundaries
)



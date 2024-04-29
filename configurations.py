MAX_SEQ_LENGTH = 120
FRAME_SKIP = True

if FRAME_SKIP:
    target_fps = 10
    MAX_SEQ_LENGTH = target_fps*2 #(time in seconds)

model_name = 'cnn_model'

if model_name == 'vit_model':
    IMG_SIZE = 256
elif model_name == 'inception_model':
    IMG_SIZE = 299
else:
    IMG_SIZE = 256
    
IMAGE_SIZE = (IMG_SIZE,IMG_SIZE)

NUM_FEATURES = 512
EPOCHS = 20
batch_size = 32
random_seed = 2024


learning_rate = 0.0001
weight_decay = 0.0001

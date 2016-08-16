import time
from ResNet import Meta, Blob, Producer, Preprocess, Batch, Net, ResNet50

# Set environment variables
WORKING_DIR = '/mnt/data/dish-clean-save/' + time.strftime('%Y-%m-%d-%H%M%S')
IMAGE_DIR = '/mnt/data/dish-clean/'
IS_IMAGE_CHECKED = True  # set to False on the first run to enable checking

# Create pipelines
# image and label are wrapped by blobs, and passed along the pipeline
Meta.load(working_dir=WORKING_DIR)  # always register working_dir first  # always register
producer = Producer()  # produce blobs from existing image files
preprocess = Preprocess()  # process images (e.g. resize, crop, ...)
batch = Batch()  # batching the images, not needed in online version

# Blobs allow function chaining for readability
trainBlob = producer.trainFile(image_dir=IMAGE_DIR, check=not IS_IMAGE_CHECKED).func(preprocess.train).func(batch.train)
testBlob = producer.testFile(image_dir=IMAGE_DIR).func(preprocess.test).func(batch.test)

# Declare net now for net.case usage below
net = ResNet50(
    learning_rate=1e-1,
    learning_rate_decay_steps=1500,
    learning_rate_decay_rate=0.5,
    is_train=True,
    is_show=True)

image = net.case([
    (Net.Phase.TRAIN, lambda: trainBlob.image),
    (Net.Phase.TEST, lambda: testBlob.image)],
    shape=(batch.batch_size,) + preprocess.shape)
label = net.case([
    (Net.Phase.TRAIN, lambda: trainBlob.label),
    (Net.Phase.TEST, lambda: testBlob.label)],
    shape=(None,))
blob = Blob(image=image, label=label)

# Training function defined in Net
net.build(blob)
net.train(iteration=10000)

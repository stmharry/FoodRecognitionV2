import time
from ResNet import Meta, Blob, FileProducer, Preprocess, Batch, Net, ResNet50

# Set environment variables
IMAGE_DIR = '/mnt/data/dish-clean/'
WORKING_DIR = '/mnt/data/dish-clean-save/' + time.strftime('%Y-%m-%d-%H%M%S')
IS_IMAGE_CHECKED = True  # set to False on the first run to enable checking

# Create pipelines
# image and label are wrapped by blobs, and passed along the pipeline
Meta.train(image_dir=IMAGE_DIR, working_dir=WORKING_DIR)  # always register working_dir first
producer = FileProducer()  # produce blobs from existing image files
preprocess = Preprocess()  # process images (e.g. resize, crop, ...)
batch = Batch(min_after_dequeue=128)  # batching the images, not needed in online version
net = ResNet50(
    learning_rate=1e-1,
    learning_rate_decay_steps=1500,
    learning_rate_decay_rate=0.5,
    is_train=True,
    is_show=True)

# Blobs allow function chaining for readability
trainBlob = producer.trainBlob(image_dir=IMAGE_DIR, check=not IS_IMAGE_CHECKED).func(preprocess.train).func(batch.train)
testBlob = producer.testBlob(image_dir=IMAGE_DIR).func(preprocess.test).func(batch.test)

image = net.case([
    (Net.Phase.TRAIN, lambda: trainBlob.images[0]),
    (Net.Phase.TEST, lambda: testBlob.images[0])],
    shape=(batch.batch_size,) + preprocess.shape)
label = net.case([
    (Net.Phase.TRAIN, lambda: trainBlob.labels[0]),
    (Net.Phase.TEST, lambda: testBlob.labels[0])],
    shape=(None,))
Blob(images=[image], labels=[label]).func(net.build)

# Training function defined in Net
net.train(iteration=10000)

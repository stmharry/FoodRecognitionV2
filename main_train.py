from ResNet import set_meta, Meta, Blob, FileProducer, Preprocess, Batch, Net, ResNet50
from env import *

if __name__ == '__main__':
    meta = Meta.train(image_dir=IMAGE_DIR, working_dir=WORKING_DIR)
    set_meta(meta)

    producer = FileProducer()
    preprocess = Preprocess()
    batch = Batch()
    net = ResNet50(
        learning_rate=1e-1,
        learning_rate_decay_steps=LEARNING_RATE_DECAY_STEPS,
        learning_rate_decay_rate=0.5,
        is_train=True,
        is_show=True,
    )

    trainBlob = producer.trainBlob(image_dir=IMAGE_DIR, check=not IS_IMAGE_ALREADY_CHECKED).func(preprocess.train).func(batch.train)
    testBlob = producer.testBlob(image_dir=IMAGE_DIR).func(preprocess.test).func(batch.test)

    (image, label) = net.case([
            (Net.Phase.TRAIN, lambda: trainBlob.as_tuple_list()[0]),
            (Net.Phase.TEST, lambda: testBlob.as_tuple_list()[0])
        ],
        shapes=[(batch.batch_size,) + preprocess.shape, (None,)],
    )
    Blob(images=image, labels=label).func(net.build)

    net.start()
    net.train(iteration=ITERATION)

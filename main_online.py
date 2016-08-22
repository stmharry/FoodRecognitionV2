from __future__ import print_function

import collections
import cStringIO
import json
import numpy as np
import PIL.Image
import sys
import urllib

# from ResNet import Meta, QueueProducer, Preprocess, Batch, Net, ResNet50, Postprocess, Consumer, Timer
from ResNet import Meta, QueueProducer, Preprocess, Batch, Net, ResNet50, Postprocess, Timer

json.encoder.FLOAT_REPR = '{:.4f}'.format

WORKING_DIR = '/mnt/data/dish-clean-save/2016-08-16-191753/'
TOP_K = 6


def url_to_image(url):
    if url.startswith('http'):
        pipe = urllib.urlopen(url)
        stringIO = cStringIO.StringIO(pipe.read())
        pil = PIL.Image.open(stringIO)
    else:
        pil = PIL.Image.open(url)
    image = np.array(pil.getdata(), dtype=np.uint8).reshape((pil.height, pil.width, -1))
    return image


class Request(object):
    def __init__(self, urls):
        self.urls = urls
        num_urls = len(urls)

        net.online(**batch.kwargs(total_size=num_urls, phase=Net.Phase.TEST))

        with Timer('ResNet50 running prediction on %d images... ' % num_urls):
            for url in urls:
                print('.', end='')
                sys.stdout.flush()

                net.online(**producer.kwargs(image=url_to_image(url)))
            print('')

            probs = list()
            while True:
                fetch_val = net.online(**blob.kwargs())
                prob = fetch_val[net.prob.name]
                if prob.size == 0:
                    break
                probs.append(prob)

            self.probs = np.concatenate(probs, axis=0)

    def to_json(self):
        json_dict = collections.OrderedDict()
        for (url, prob) in zip(self.urls, self.probs):
            index = np.argsort(prob)[::-1]
            json_dict[url] = collections.OrderedDict([(Meta.CLASS_NAMES[i], float(prob[i])) for i in index[:TOP_K]])
        return json.dumps(json_dict, indent=4)


if __name__ == '__main__':
    Meta.test(working_dir=WORKING_DIR)
    producer = QueueProducer()
    preprocess = Preprocess()
    batch = Batch()
    net = ResNet50()
    postprocess = Postprocess()

    with Timer('Building network...'):
        producer.blob().func(preprocess.test).func(batch.test).func(net.build)
        # blob = postprocess.blob(net.prob).func(consumer.build)
        blob = postprocess.blob(net.prob)
        net.start(default_phase=Net.Phase.TEST)

    request = Request(np.loadtxt('request.txt', dtype=np.str))
    print(request.to_json())

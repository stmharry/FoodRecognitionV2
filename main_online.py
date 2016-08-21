# TODO: second request

from __future__ import print_function

import collections
import cStringIO
import json
import numpy as np
import PIL.Image
import sys
import urllib

from ResNet import Meta, QueueProducer, Preprocess, Batch, Net, ResNet50, Postprocess, Consumer, Timer

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


class Query(object):
    def __init__(self, urls):
        self.urls = urls
        num_urls = len(urls)

        net.online(**batch.kwargs(total_size=num_urls, phase=Net.Phase.TEST))
        net.online(**consumer.kwargs(total_size=num_urls))
        net.start(default_phase=Net.Phase.TEST)

        with Timer('ResNet50 running prediction on %d images... ' % num_urls):
            for url in urls:
                print('.', end='')
                sys.stdout.flush()

                net.online(**producer.kwargs(image=url_to_image(url)))
            print('')

            fetch_val = net.online(**blob.kwargs())
            self.probs = fetch_val[blob.values[0].name]

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
    consumer = Consumer()

    with Timer('Building network...'):
        producer.blob().func(preprocess.test).func(batch.test).func(net.build)
        blob = postprocess.blob(net.prob).func(consumer.build)

    query = Query(np.loadtxt('query.txt', dtype=np.str))
    print(query.to_json())

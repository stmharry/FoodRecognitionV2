from __future__ import print_function

import collections
import cStringIO
import json
import numpy as np
import PIL.Image
import sys
import time
import urllib

from ResNet import Meta, QueueProducer, Preprocess, Batch, ResNet50

json.encoder.FLOAT_REPR = '{:.4f}'.format

WORKING_DIR = '/mnt/data/dish-clean-save/2016-08-16-191753/'
TOP_K = 6

Meta.test(working_dir=WORKING_DIR)
producer = QueueProducer()
preprocess = Preprocess()
batch = Batch()
net = ResNet50()

producer.blob().func(preprocess.test).func(batch.test).func(net.build)


class Query(object):
    def __init__(self, urls):
        self.urls = urls
        num_urls = len(urls)

        _ = net.online(
            feed_dict={batch.test_total_size: num_urls},
            fetch={'assign': batch.test_assign})

        print('Retrieving and queueing images... ')
        start = time.time()
        for (num_url, url) in enumerate(urls):
            print('\033[2K\r%d / %d' % (num_url + 1, num_urls), end='')
            sys.stdout.flush()

            if url.startswith('http'):
                pipe = urllib.urlopen(url)
                stringIO = cStringIO.StringIO(pipe.read())
                pil = PIL.Image.open(stringIO)
            else:
                pil = PIL.Image.open(url)
            image = np.array(pil.getdata(), dtype=np.uint8).reshape((pil.height, pil.width, -1))

            fetch_val = net.online(
                feed_dict={producer.placeholder: image},
                fetch={'enqueue': producer.enqueue})
        print('')
        print('Time: %.3f s' % (time.time() - start))

        print('Processing %d images... ' % num_urls, end='')
        start = time.time()
        total_size = num_urls
        probs = list()
        while total_size > 0:
            fetch_val = net.online(
                fetch={
                    'prob': net.prob,
                    'total_size': batch.test_total_size})
            probs.append(fetch_val['prob'])
            total_size = fetch_val['total_size']
        print('time: %.3f s' % (time.time() - start))

        self.probs = np.concatenate(probs, axis=0)

    def get_json(self):
        json_dict = collections.OrderedDict()
        for (url, prob) in zip(self.urls, self.probs):
            index = np.argsort(prob)[::-1]
            json_dict[url] = collections.OrderedDict([(Meta.CLASS_NAMES[i], float(prob[i])) for i in index[:TOP_K]])
        return json.dumps(json_dict, indent=4)


def query_from_file(filename):
    return Query(open(filename, 'r').read().rstrip('\n').split('\n'))

if __name__ == '__main__':
    query = query_from_file('query.txt')

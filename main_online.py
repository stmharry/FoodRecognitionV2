#!/usr/bin/env python

from __future__ import print_function

import BaseHTTPServer
import collections
import cStringIO
import json
import numpy as np
import PIL.Image
import urllib
import urlparse

from ResNet import Meta, QueueProducer, Preprocess, Batch, Net, ResNet50, Postprocess, Timer

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


class RequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_GET(self):
        requestString = urlparse.parse_qs(self.path.lstrip('/?'))
        url = requestString['url'][0]

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        request = Request([url])
        request.build()
        json = request.to_json()

        self.wfile.write('<img src="%s" width="368"></img>' % url)
        self.wfile.write('<p style="white-space: pre; font-family: monospace;">%s</p>' % json.replace('\n', '<br/>'))
        self.wfile.close()

    def do_POST(self):
        self.send_response(200)

    do_PUT = do_POST
    do_DELETE = do_GET


class Request(object):
    def __init__(self, urls):
        json.encoder.FLOAT_REPR = '{:.4f}'.format

        self.urls = urls
        self.num_urls = len(urls)

    def build(self):
        net.online(**batch.kwargs(total_size=self.num_urls, phase=Net.Phase.TEST))

        with Timer('ResNet50 running prediction on %d images... ' % self.num_urls):
            for url in self.urls:
                net.online(**producer.kwargs(image=url_to_image(url)))

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
        blob = postprocess.blob(net.prob)
        net.start(default_phase=Net.Phase.TEST)

    server = BaseHTTPServer.HTTPServer(('0.0.0.0', 6006), RequestHandler)
    server.serve_forever()

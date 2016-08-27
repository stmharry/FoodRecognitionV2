from rest_framework.decorators import parser_classes
from rest_framework.decorators import renderer_classes
from rest_framework.parsers import JSONParser
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

import collections
import skimage.io
import sys

# RESNET_ROOT = '/var/django/ResidualNetworkV2'
RESNET_ROOT = '/home/harry/Repository/FoodRecognitionV2'
WORKING_DIR = '/mnt/data/dish-clean-save/2016-08-16-191753'
TOP_K = 6

if RESNET_ROOT not in sys.path:
    sys.path.append(RESNET_ROOT)

from ResNet import Meta, QueueProducer, Preprocess, Batch, Net, ResNet50, Postprocess, Timer


class NetWrapper(object):
    def __init__(self):
        Meta.test(working_dir=WORKING_DIR)
        self.producer = QueueProducer()
        self.preprocess = Preprocess()
        self.batch = Batch()
        self.net = ResNet50()
        self.postprocess = Postprocess()

        with Timer('Building network...'):
            self.producer.blob().func(self.preprocess.test).func(self.batch.test).func(self.net.build)
            self.blob = self.postprocess.blob(self.net.prob)
            self.net.start(default_phase=Net.Phase.TEST)

    def get_results(self, urls):
        num_urls = len(urls)
        self.net.online(**self.batch.kwargs(total_size=num_urls, phase=Net.Phase.TEST))

        with Timer('ResNet50 running prediction on %d images... ' % num_urls):
            for url in urls:
                self.net.online(**self.producer.kwargs(image=skimage.io.imread(url)))

            results = list()
            while True:
                fetch = self.net.online(**self.blob.kwargs())
                probs = fetch[self.net.prob.name]
                for prob in probs:
                    indices = sorted(xrange(len(Meta.CLASS_NAMES)), key=prob.__getitem__)[:-(TOP_K + 1):-1]
                    classes = collections.OrderedDict([(Meta.CLASS_NAMES[index], round(prob[index], 4)) for index in indices])
                    results.append(dict(status='ok', classes=classes))
                if probs.size == 0:
                    break

        return results


class ClassifyService(APIView):
    NET_WRAPPER = NetWrapper()

    @parser_classes((JSONParser,))
    @renderer_classes((JSONRenderer,))
    def post(self, request, format=None):
        content = dict(results=ClassifyService.NET_WRAPPER.get_results(request.data['url']))
        return Response(content)

from rest_framework.decorators import parser_classes
from rest_framework.decorators import renderer_classes
from rest_framework.parsers import JSONParser
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

import skimage.io

from ResNet import Meta, QueueProducer, Preprocess, Batch, Net, ResNet50, Postprocess, Timer

DJANGO_ROOT = '/vol/django_server'
WORKING_DIR = '/mnt/data/dish-clean-save/2016-08-16-191753/'


class ResNetWrapper(object):
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

    def get(self, urls):
        num_urls = len(urls)
        with Timer('ResNet50 running prediction on %d images... ' % num_urls):
            for url in urls:
                self.net.online(**self.producer.kwargs(image=skimage.io.imread(url)))

            prob_dicts = list()
            while True:
                fetch = self.net.online(**self.blob.kwargs())
                probs = fetch[self.net.prob.name]
                for prob in probs:
                    prob_dicts.append(dict(zip(Meta.CLASS_NAMES, prob)))
                if probs.size == 0:
                    break

        return dict(status='ok', classes=prob_dicts)


class ClassifyService(APIView):
    RESNET_WRAPPER = ResNetWrapper()

    @parser_classes((JSONParser,))
    @renderer_classes((JSONRenderer,))
    def post(self, request, format=None):
        images = request.data['images']
        content = dict(results=ClassifyService.RESNET_WRAPPER.get(images))

        return Response(content)

import collections
import json
import requests
import time

json_obj = {'images': [
    "http://pic.2bite.com/event/5642f19c518f6e735e8b499e/classify/c_be4b8be6b428465ba3ae6430cb632dcb.jpg",
    "http://pic.2bite.com/event/5642f19c518f6e735e8b499e/classify/c_4798ea3195ca4b56a93e0fbc3d1227d8.jpg"]}

for addr in ['http://classify.2bite.com:8080/classify', 'http://52.52.99.175:8080/classify']:
    start = time.time()
    response = requests.post(addr, json=json_obj)

    print('%s: %.3f s' % (addr, time.time() - start))
    print(json.dumps(json.loads(response.text, object_pairs_hook=collections.OrderedDict), indent=4))

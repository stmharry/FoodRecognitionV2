import collections
import json
import requests
import time

start = time.time()
response = requests.post('http://52.52.99.175:8080/classify', json={'images': []})

print('%.3f s' % (time.time() - start))
print(json.dumps(json.loads(response.text, object_pairs_hook=collections.OrderedDict), indent=4))

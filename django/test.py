import collections
import json
import requests

response = requests.post('http://127.0.0.1:8000/classify', json={'url': [
    'http://pic.pimg.tw/sinea100/1393852602-2597251841.jpg',
    'http://the-sun.on.cc/cnt/lifestyle/20150710/photo/0710-00479-002b1.jpg']})

print('Raw JSON:')
print(response.text)

print('Decoded JSON:')
print(json.dumps(json.loads(response.text, object_pairs_hook=collections.OrderedDict), indent=4))

import collections
import json
import requests

response = requests.post('http://127.0.0.1:8080/classify', json={'images': [
	"http://a.ecimg.tw/pic/v1/data/item/201608/D/Y/A/J/0/0/DYAJ00-A9007AJY8000_57c4eee1ef0d8.jpg",
	"http://a.ecimg.tw/pic/v1/data/item/201608/D/Y/A/J/0/0/DYAJ00-A9007AJY8000_57c4eee1ef0d8.jpg",
	"http://a.ecimg.tw/pic/v1/data/item/201608/D/Y/A/J/0/0/DYAJ00-A9007AJY8000_57c4eee1ef0d8.jpg",
	"http://a.ecimg.tw/pic/v1/data/item/201608/D/Y/A/J/0/0/DYAJ00-A9007AJY8000_57c4eee1ef0d8.jpg",
	"http://a.ecimg.tw/pic/v1/data/item/201608/D/Y/A/J/0/0/DYAJ00-A9007AJY8000_57c4eee1ef0d8.jpg",
	"http://a.ecimg.tw/pic/v1/data/item/201608/D/Y/A/J/0/0/DYAJ00-A9007AJY8000_57c4eee1ef0d8.jpg",
	"http://a.ecimg.tw/pic/v1/data/item/201608/D/Y/A/J/0/0/DYAJ00-A9007AJY8000_57c4eee1ef0d8.jpg",
	"http://a.ecimg.tw/pic/v1/data/item/201608/D/Y/A/J/0/0/DYAJ00-A9007AJY8000_57c4eee1ef0d8.jpg",
	"http://a.ecimg.tw/pic/v1/data/item/201608/D/Y/A/J/0/0/DYAJ00-A9007AJY8000_57c4eee1ef0d8.jpg",
	"http://a.ecimg.tw/pic/v1/data/item/201608/D/Y/A/J/0/0/DYAJ00-A9007AJY8000_57c4eee1ef0d8.jpg",
	"http://a.ecimg.tw/pic/v1/data/item/201608/D/Y/A/J/0/0/DYAJ00-A9007AJY8000_57c4eee1ef0d8.jpg",
	"http://a.ecimg.tw/pic/v1/data/item/201608/D/Y/A/J/0/0/DYAJ00-A9007AJY8000_57c4eee1ef0d8.jpg"]})

print('Raw JSON:')
print(response.text)

print('Decoded JSON:')
print(json.dumps(json.loads(response.text, object_pairs_hook=collections.OrderedDict), indent=4))

import time

IS_IMAGE_ALREADY_CHECKED = True
CURRENT_TIME = time.strftime('%Y-%m-%d-%H%M%S')

# CONTENT_TYPE
IMAGE_DIR = '/mnt/data/content-img'
WORKING_DIR = '/mnt/data/content-save/' + CURRENT_TIME
LEARNING_RATE_DECAY_STEPS = 500
ITERATION = 5000

# FOOD_TYPE
'''
IMAGE_DIR = '/mnt/data/food-img'
WORKING_DIR = '/mnt/data/food-save/' + CURRENT_TIME
LEARNING_RATE_DECAY_STEPS = 4000
ITERATION = 25000
'''

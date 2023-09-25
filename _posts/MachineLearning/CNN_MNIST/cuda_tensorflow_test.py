import tensorflow as tf
print("GPU 사용 가능:", tf.test.is_gpu_available())
print("GPU?:", tf.config.list_physical_devices('GPU'))
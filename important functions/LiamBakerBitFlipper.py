import tensorflow as tf
import random

def tensorBitFlipper(tensor):
    if len(tf.shape(tensor)[0:]) > 1:
        num_subtensors = tensor.shape[0]
        subtensor_index = random.randint(0, num_subtensors - 1)
        subtensor = tensor[subtensor_index]

        bit_position = random.randint(0, tf.size(subtensor) - 1)
        element_index = random.randint(0, subtensor.shape[0] - 1)

        flip_value = tf.bitwise.left_shift(tf.constant(1, dtype=subtensor.dtype), bit_position)
        element_to_flip = subtensor[element_index]
        flipped_element = tf.bitwise.bitwise_xor(element_to_flip, flip_value)

        flipped_subtensor = tf.tensor_scatter_nd_update(subtensor, [[element_index]], [flipped_element])
        flipped_tensor = tf.tensor_scatter_nd_update(tensor, [[subtensor_index]], [flipped_subtensor])

        return flipped_tensor
    else:
        length = tf.size(tensor)
        bit_position = random.randint(0, length - 1)
        element_index = random.randint(0, length - 1)
        flip_value = 1 << bit_position
        flipped_element = tensor[element_index] ^ flip_value
        tensor_array = tensor.numpy()
        tensor_array[element_index] = flipped_element
        updated_tensor = tf.constant(tensor_array)
        return updated_tensor

# Example usage
tensor1 = tf.constant([5, 3, 2, 100], dtype=tf.uint8)
tensor2 = tf.constant([10, 20, 30, 40], dtype=tf.uint8)
tensor3 = tf.constant([1, 2, 3, 4], dtype=tf.uint8)
tensor = tf.stack([tensor1, tensor2, tensor3])

print("Original Stacked Tensor:")
print(tensor)

flipped_tensor = tensorBitFlipper(tensor)
print("Flipped Stacked Tensor:")
print(flipped_tensor)

import torch
import random

def tensorBitFlipper(tensor):
    bit_position = random.randint(0,len(tensor)-1)
    element_index = random.randint(0,len(tensor)-1)
    flip_value = 1 << bit_position
    flipped_element = tensor[element_index] ^ flip_value

    # Create a copy of the tensor and update the flipped element
    flipped_tensor = tensor.clone()
    flipped_tensor[element_index] = flipped_element
    return flipped_tensor

# Example usage
tensor = torch.tensor([5, 3, 2, 100], dtype=torch.uint8)
print(tensorBitFlipper(tensor))

import torch

# Mock cos_projections tensor with 8 channels (directions) for a 3x3 image
cos_projections = torch.tensor([
    [[1, 2, 0], [3, 2, 1], [0, 1, 3]],   # Channel 0 (direction 1)
    [[4, 0, 1], [1, 3, 0], [3, 2, 0]],   # Channel 1 (direction 2)
    [[0, 3, 2], [4, 0, 3], [1, 0, 1]],   # Channel 2 (direction 3)
    [[2, 1, 0], [0, 4, 2], [0, 3, 4]],   # Channel 3 (direction 4)
    [[1, 3, 4], [2, 1, 0], [2, 0, 1]],   # Channel 4 (direction 5)
    [[0, 2, 1], [0, 1, 3], [4, 1, 3]],   # Channel 5 (direction 6)
    [[2, 1, 0], [4, 2, 1], [0, 2, 1]],   # Channel 6 (direction 7)
    [[3, 1, 4], [0, 0, 2], [3, 1, 2]],   # Channel 7 (direction 8)
])

# Mock gradient magnitudes (3x3 image)
magnitudes = torch.tensor([
    [5, 10, 8],
    [9, 12, 6],
    [4, 11, 7]
])


# Compute the occupancy tensor (binary mask of the dominant directions)
occupancy_tensor = (cos_projections == torch.max(cos_projections, dim=0, keepdim=True)[0])
print("Occupancy Tensor:")
print(occupancy_tensor)

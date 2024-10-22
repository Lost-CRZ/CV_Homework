from pathlib import Path

import torch
from vision.part3 import my_conv2d_pytorch

ROOT = Path(__file__).resolve().parent.parent  # ../..


def test_my_conv2d_pytorch():
    """Assert that convolution output is correct, and groups are handled correctly
    for a 2-channel image with 4 filters (yielding 2 groups).
    """
    image = torch.zeros((1, 2, 3, 3), dtype=torch.int)
    image[0, 0] = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.int)
    image[0, 1] = torch.tensor(
        [[9, 10, 11], [12, 13, 14], [15, 16, 17]], dtype=torch.int
    )

    identity_filter = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.int)
    double_filter = torch.tensor([[0, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=torch.int)
    triple_filter = torch.tensor([[0, 0, 0], [0, 3, 0], [0, 0, 0]], dtype=torch.int)
    ones_filter = torch.ones(3, 3, dtype=torch.int)
    filters = torch.stack(
        [identity_filter, double_filter, triple_filter, ones_filter], 0
    )

    filters = filters.reshape(4, 1, 3, 3)
    feature_maps = my_conv2d_pytorch(image.int(), filters)

    assert feature_maps.shape == torch.Size([1, 4, 3, 3])

    gt_feature_maps = torch.zeros((1, 4, 3, 3), dtype=torch.int)

    # identity filter on channel 1
    gt_feature_maps[0, 0] = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.int
    )
    # doubling filter on channel 1
    gt_feature_maps[0, 1] = torch.tensor(
        [[0, 2, 4], [6, 8, 10], [12, 14, 16]], dtype=torch.int
    )
    # tripling filter on channel 2
    gt_feature_maps[0, 2] = torch.tensor(
        [[27, 30, 33], [36, 39, 42], [45, 48, 51]], dtype=torch.int
    )
    gt_feature_maps[0, 3] = torch.tensor(
        [[44, 69, 48], [75, 117, 81], [56, 87, 60]], dtype=torch.int
    )

    assert torch.allclose(gt_feature_maps.int(), feature_maps.int())

    image_2 = torch.zeros((1,2,7,7), dtype = torch.int)
    image_2[0,0] = torch.tensor([[0,1,2,3,4,5,6],[7,8,9,10,11,12,13],[14,15,16,17,18,19,20],[21,22,23,24,25,26,27],[28,29,30,31,32,33,34],[35,36,37,38,39,40,41],[42,43,44,45,46,47,48]],dtype = torch.int)
    image_2[0,1] = torch.tensor([[48,47,46,45,44,43,42],[41,40,39,38,37,36,35],[34,33,32,31,30,29,28],[27,26,25,24,23,22,21],[20,19,18,17,16,15,14],[13,12,11,10,9,8,7],[6,5,4,3,2,1,0]],dtype = torch.int)


    filters_2 = torch.stack([identity_filter, double_filter], dim=0).reshape(2, 1, 3, 3)
    stride_feature_maps = my_conv2d_pytorch(image_2, filters_2, stride=2)
    assert stride_feature_maps.shape == torch.Size([1,2,4,4])
    gt_feature_maps_stride = torch.zeros((1,2,4,4), dtype = torch.int)
    gt_feature_maps_stride[0,0] = torch.tensor([[0,2,4,6],[14,16,18,20],[28,30,32,34],[42,44,46,48]],dtype = torch.int)
    gt_feature_maps_stride[0,1] = torch.tensor([[96,92,88,84],[68, 64, 60, 56],[40, 36, 32, 28],[12,8,4,0]],dtype = torch.int)
    assert torch.allclose(stride_feature_maps.int(), gt_feature_maps_stride.int())

    dilation_feature_maps = my_conv2d_pytorch(image_2.float(), filters_2.float(), dilation=2).int()
    assert dilation_feature_maps.shape == torch.Size([1,2,5,5])
    gt_feature_maps_dilation = torch.zeros((1,2,5,5), dtype = torch.int)
    gt_feature_maps_dilation[0,0] = torch.tensor([[8,9,10,11,12],[15, 16, 17, 18, 19],[22, 23, 24, 25, 26],[29, 30, 31, 32, 33],[36, 37, 38, 39, 40]],dtype = torch.int)
    gt_feature_maps_dilation[0,1] = torch.tensor([[80,78,76,74,72],[66, 64, 62, 60, 58],[52, 50, 48, 46, 44],[38, 36, 34, 32, 30],[24, 22, 20, 18, 16]],dtype = torch.int)
    assert torch.allclose(dilation_feature_maps, gt_feature_maps_dilation)

if __name__ == "__main__":
    test_my_conv2d_pytorch()

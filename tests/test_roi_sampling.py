"""
CommandLine:
    pytest tests/test_roi_sampling.py
"""
import numpy as np
import torch

from mmdet.ops.roi_sampling import roi_sampling, invert_roi_bbx


def test_nms_device_and_dtypes_gpu():
    """
    CommandLine:
        xdoctest -m tests/test_roi_sampling.py test_roi_sampling_device_and_dtypes_gpu
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('test requires GPU and torch+cuda')

    bbx_dets = np.array([[ 366.8981,  503.8129,  483.4658,  558.4958],
                         [ 314.6978, 1728.0537,  503.4887, 1786.7266],
                         [ 312.2163, 1783.3762,  501.1314, 1858.0264],
                         [ 357.2108, 1666.8306,  427.9639, 1692.9177],
                         [ 352.9979, 1686.4675,  426.5080, 1708.1284],
                         [ 369.2542, 1040.2343,  506.5618, 1213.3898],
                         [ 344.8744, 1337.1475,  527.9548, 1576.3521],
                         [ 379.8827,  956.4868,  471.0074, 1065.2467],
                         [ 396.4836,  774.0189,  461.9510,  909.5333],
                         [ 362.6898, 1572.2887,  426.6845, 1652.2772],
                         [ 407.2052,  566.6193,  457.3239,  651.4299],
                         [ 380.3073, 1558.8389,  402.9362, 1577.3118],
                         [ 380.5341, 1560.0563,  411.1904, 1579.3331],
                         [ 409.9388,  565.4561,  441.7789,  650.7983]])

    mask_preds = np.random.rand(bbx_dets.shape[0], 28, 28)


    for device_id in range(torch.cuda.device_count()):
        print('Run NMS on device_id = {!r}'.format(device_id))
        # GPU can handle float32 but not float64
        dets = torch.FloatTensor(bbx_dets.astype(np.float32)).to(device_id)
        bbx_inv = invert_roi_bbx(dets, (28,28), (1024, 2048)) 
        bbx_idx = torch.arange(0, dets.size(0), 
                  dtype=torch.long, device=dets.device)
        preds = torch.FloatTensor(mask_preds.astype(np.float32)).to(device_id)
        preds = roi_sampling(preds.unsqueeze(1), bbx_inv, bbx_idx, (1024, 2048),
                            padding="zero") 
       assert preds.shape[0] == dets.shape[0]
       assert preds.shape[2] == 1024
       assert preds.shape[3] == 2048


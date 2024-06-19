
from mmcv.transforms.base import BaseTransform

from mmseg.datasets.transforms import LoadAnnotations
from mmseg.registry import TRANSFORMS
from skimage import transform
import numpy as np




@TRANSFORMS.register_module()
class RescaleT(BaseTransform):
    """Resize images & seg to output size.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map
    - img_shape
    - pad_shape
    """
    def __init__(self, output_size=None):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def transform(self, results: dict) -> dict:
        """Call function to resize images and segment map

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape' keys are updated.
        """
        # Align image to multiple of size divisor.

        img = results['img']

        # resize the image to (self.output_size, self.output_size) and convert image from range [0,255] to [0,1]
        img = transform.resize(img, self.output_size, mode='constant')

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['pad_shape'] = img.shape[:2]

        if 'gt_seg_map' in results:
            gt_seg = results['gt_seg_map']
            gt_seg = transform.resize(gt_seg, self.output_size, mode='constant', order=0,
                                      preserve_range=True)
            results['gt_seg_map'] = gt_seg

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(out_size={self.out_size})')
        return repr_str

@TRANSFORMS.register_module()
class Normal(BaseTransform):
    def transform(self, results: dict) -> dict:

        # Align image to multiple of size divisor.
        img = results['img']

        img = img/np.max(img)
        results['img'] = img
        if 'gt_seg_map' in results:
            gt = results['gt_seg_map']
            if np.max(gt) < 1e-6:
                gt = gt
            else:
                gt = gt / np.max(gt)
            results['gt_seg_map'] = gt

        return results









# Ultralytics YOLO 🚀, AGPL-3.0 license
import torch
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from research.pytorch_utils import transfer_tensor_to_device_nested


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolov8n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs, head_name):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            multi_label=True if head_name == 'action' else False,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img.shape, path=img_path, names=self.model.names, boxes=pred))
        return results

    @torch.no_grad()
    def simple_predict(self, batch, orig_shape, to_cpu=True):
        im = self.preprocess(batch)
        preds = self.inference(im)
        if to_cpu:
            preds = {k: p[0] for k, p in preds.items()}  # Drop loss outputs
            # filtered_preds = {}  # Pre-filter not significantly faster - need to investigate
            # for head_name, pred in preds.items():
            #     if head_name == 'action':
            #         filtered_tensor = pred
            #     else:
            #         filtered_tensor = get_topk_by_conf(pred)
            #     filtered_preds[head_name] = filtered_tensor
            # preds = filtered_preds
            preds = transfer_tensor_to_device_nested(preds, 'cpu')
        return {head_name: self.postprocess_simple(pred, im, orig_shape, head_name) for head_name, pred in preds.items()}

    def postprocess_simple(self, preds, img, orig_shape, head_name):
        conf_threshold = self.args.conf[head_name]
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            conf_threshold,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            multi_label=True if head_name == 'action' else False,
        )

        results = []
        for i, pred in enumerate(preds):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_shape)
            results.append(Results(orig_shape, path="", names=self.model.names, boxes=pred))
        return results


def get_topk_by_conf(preds, top_k=200):
    # Get top_k boxes for each batch element by confidence
    conf_val = preds[:, 4, :]
    top_indices = torch.topk(conf_val, k=top_k, dim=1).indices
    expanded_indices = top_indices.unsqueeze(1).expand(-1, 5, -1)
    filtered_tensor = torch.gather(preds, 2, expanded_indices)

    return filtered_tensor

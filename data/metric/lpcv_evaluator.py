import json
import numpy as np
from up.data.metrics.base_evaluator import Metric, Evaluator
from up.utils.general.registry_factory import EVALUATOR_REGISTRY
import numpy
from PIL import Image
from up.tasks.seg.data.seg_evaluator import intersectionAndUnion


class AccuracyTracker(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = numpy.zeros((n_classes, n_classes))

    def reset(self):
        self.confusion_matrix = numpy.zeros((self.n_classes, self.n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = numpy.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc
        """
        hist = self.confusion_matrix
        self.acc = numpy.diag(hist).sum() / hist.sum()
        acc_cls = numpy.diag(hist) / (hist.sum(axis=1) + 0.000000001)
        self.acc_cls = numpy.nanmean(acc_cls)

        with numpy.errstate(invalid='ignore'):
            dice = 2 * numpy.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))

        self.mean_dice = numpy.nanmean(dice)
        freq = hist.sum(axis=1) / hist.sum()
        self.fwavacc = (freq[freq > 0] * dice[freq > 0]).sum()
        self.cls_dice = dict(zip(range(self.n_classes), dice))

        return {
            "Overall Acc: \t": self.acc,
            "Mean Acc : \t": self.acc_cls,
            "FreqW Acc : \t": self.fwavacc,
            "Mean Dice : \t": self.mean_dice,
        }


@EVALUATOR_REGISTRY.register('seg_with_dice')
class SegDiceEvaluator(Evaluator):
    def __init__(self, num_classes=19, ignore_label=255, cmp_key=None):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.cmp_key = cmp_key
        self.tracker = AccuracyTracker(num_classes)

    def load_res(self, res_file, res=None):
        res_dict = {}
        if res is not None:
            lines = []
            for device_res in res:
                for items in device_res:
                    for line in items:
                        lines.append(line)
        else:
            lines = []
            with open(res_file, 'r') as fin:
                for line in fin:
                    lines.append(json.loads(line))
        for line in lines:
            info = line
            for key in info.keys():
                if key not in res_dict.keys():
                    res_dict[key] = [info[key]]
                else:
                    res_dict[key].append(info[key])
        return res_dict

    def eval(self, res_file, res=None):
        self.tracker.reset()
        res_dict = self.load_res(res_file, res)
        print(res_dict['pred'][0].shape)
        # outImage: np.ndarray = np.squeeze(res_dict['pred'][0], axis=0)
        outImage=res_dict['pred'][0].astype(np.uint8)
        outImage = Image.fromarray(outImage, mode='L')
        outImage.save('val_0.png')
        # print(res_dict.keys(),res_dict)
        inter_sum = 0.0
        union_sum = 0.0
        target_sum = 0.0
        dice = 0.0  # Add this line
        if 'inter' in res_dict:
            image_num = len(res_dict['inter'])
        else:
            image_num = len(res_dict['pred'])
            preds = res_dict['pred']
            targets = res_dict['gt_semantic_seg']
        for idx in range(image_num):
            self.tracker.reset()
            if 'inter' not in res_dict:
                inter, union, target = intersectionAndUnion(preds[idx],
                                                            targets[idx],
                                                            self.num_classes,
                                                            self.ignore_label)
                # pred = preds[idx].sum()  # Add this line
            else:
                inter = np.array(res_dict['inter'][idx])
                union = np.array(res_dict['union'][idx])
                target = np.array(res_dict['target'][idx])
                # pred = np.array(res_dict['pred'][idx]).sum()  # Add this line
                self.tracker.update(res_dict['gt'][idx], res_dict['pred'][idx])
                self.tracker.get_scores()
                dice += self.tracker.mean_dice
            inter_sum += inter
            union_sum += union
            target_sum += target
            # pred_sum += pred  # Add this line
        miou_cls = inter_sum / (union_sum + 1e-10)
        miou = np.mean(miou_cls)
        acc_cls = inter_sum / (target_sum + 1e-10)
        macc = np.mean(acc_cls)
        # dice_cls = 2 * inter_sum / (target_sum + pred_sum + 1e-10)  # Add this line
        # dice = np.mean(dice_cls)  # Add this line
        dice /= image_num
        res = {}
        res['dice'] = dice  # Add this line
        res['mIoU'] = miou
        res['mAcc'] = macc

        metric = Metric(res)
        metric_name = self.cmp_key if self.cmp_key and metric.get(self.cmp_key, False) else 'mIoU'
        metric.set_cmp_key(metric_name)
        return metric

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(name, help='subcommand for Seg evaluation')
        subparser.add_argument('--res_file', required=True, help='results file of detection')
        return subparser

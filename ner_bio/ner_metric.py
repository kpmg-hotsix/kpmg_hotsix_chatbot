import numpy as np
from datasets import load_metric

class Seqeval():
    def __init__(self, label_list):
        self.metric = load_metric("seqeval")
        self.label_list = label_list

    def compute_seqeval(self, pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [[self.label_list[p] for p, l in zip(prediction, label) if l >= 0]
                            for prediction, label in zip(predictions, labels)]
        true_labels = [[self.label_list[l] for p, l in zip(prediction, label) if l >= 0]
                       for prediction, label in zip(predictions, labels)]
        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        eval_results = dict(
            precision=results["overall_precision"],
            recall=results["overall_recall"],
            f1=results["overall_f1"],
            accuracy=results["overall_accuracy"]
        )
        return eval_results
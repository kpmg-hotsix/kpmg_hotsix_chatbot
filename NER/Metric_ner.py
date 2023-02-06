import numpy as np
from datasets import load_metric
from tqdm import tqdm

class Metric :

    def __init__(self, ) :
        self.metric = load_metric("seqeval")
        # self.label_list = [
        #     "B-PS",
        #     "B-LC",
        #     "B-OG",
        #     "B-DT",
        #     "B-TI",
        #     "B-QT",
        #     "O",
        #     "I-PS",
        #     "I-LC",
        #     "I-OG",
        #     "I-DT",
        #     "I-TI",
        #     "I-QT",
        # ]
        self.label_list = [
            "O",
            "TR_SCIENCE",
            "TR_SOCIAL_SCIENCE",
            "TR_MEDICINE",
            "TR_ART",
            "TR_HUMANITIES",
            "TR_OTHERS"
        ]

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

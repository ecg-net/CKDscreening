import numpy as np
import sklearn.metrics
import os


def A1c_report(y, yh, model_name, w_dirs, best_epoch, split, cutoffs=None, mean=None, std=None):
    default_report(y, yh, model_name, w_dirs, best_epoch, split, cutoffs=[5.5, 6.5], mean=mean, std=std)

def hemoglobin_report(y, yh, model_name, w_dirs, best_epoch, split, cutoffs=None, mean=None, std=None):
    default_report(y, yh, model_name, w_dirs, best_epoch, split, cutoffs=[9, 10, 11, 12, 13], mean=mean, std=std)

def default_report(y, yh, model_name, w_dirs, best_epoch, split, cutoffs=None, mean=None, std=None):
    report = "{} results from epoch {} on {}:\n".format(model_name, best_epoch, split)

    for cutoff in cutoffs:
        n_cutoff = cutoff
        if mean is not None:
            n_cutoff -= mean
        if std is not None:
            n_cutoff /= std
        report += "AUC with cutoff {}: {}\n".format(cutoff, sklearn.metrics.roc_auc_score(n_cutoff <= y, yh))

        for p in [.5, .9, .95]:
            pres, recs, threshs = sklearn.metrics.precision_recall_curve(n_cutoff <= y, yh)
            threshs = np.append(threshs, threshs[-1]+1)
            recs, threshs = recs[p <= pres], threshs[p <= pres]
            k = np.argmax(recs)
            thresh = threshs[k]
            c = (y >= n_cutoff).astype(np.int32)
            ch = (yh >= thresh).astype(np.int32)
            report += "\nClassification report (cutoff: {}, precicion: {}):\n".format(cutoff, p)
            names = ["Below {}".format(cutoff), "Above {}".format(cutoff)]
            report += sklearn.metrics.classification_report(c, ch, target_names=names)
            report += "\nConfusion\n"
            report += str(sklearn.metrics.confusion_matrix(c, ch))
            report += "\n\n"
    report += "R2: {}\n".format(sklearn.metrics.r2_score(y, yh))

    print(report)

    for w_dir in w_dirs:
        with open(os.path.join(w_dir, "{}_{}_report.txt".format(model_name, split)), "a") as f:
            f.write(report)

import numpy as np
import fnmatch

def atrial_arr(row, interp_key='ECG_INTERP'):
    interp = row[interp_key]

    if type(interp) != str or interp == "":
        return False

    interp = interp.lower()

    terms = ['atrial fibrillation','atrial fibrilation', 'afib', 'afib/flut',
             'afib/flutter','afl/af', 'a-flutter' ,'flutter']

    negations = ["* has reverted to", '* has resolved', '* not present', '* has converted to', '* is no longer', '* no longer',
                "* now resolved", "* has disappeared", '* in prior ekg has resolved', '* is absent',
                '* that was present in the previous record is no longer evident', '* has been replaced by', '* is now absent',
                '* now absent', 'replaced *', 'no longer *', 'rhythm change from *','prior ekg had *', 'now absent *','no longer in *'
                ,'has replace *', "converted from *", "prior ekg was *", 'prior atrial rhythm was *', 'not *',
                'changed from *', 'has replaced previous *']

    cont = np.any([(term in interp) for term in terms])
    if not cont:
        return False

    neg = np.any([fnmatch.fnmatch(interp, term + negation)
                     for term in terms for negation in negations])
    return not neg

label_dict = {'atrial_arr': atrial_arr}

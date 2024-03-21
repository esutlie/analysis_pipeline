import numpy as np
from .load_data import load_templates


def gen_full_template(session):
    templates, template_ind, template_ids = load_templates(session)
    templates_full = np.zeros([templates.shape[0], templates.shape[1], 384])
    for i in range(template_ind.shape[0]):
        for j in range(template_ind.shape[1]):
            templates_full[i, :, template_ind[i, j]] = templates[i, :, j]
    return templates_full, template_ids



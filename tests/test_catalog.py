import numpy as np

from sfc_gdn2.data.catalog import dicom_age, dx, dx_group, hand, modality, sex


def test_normalizers():
    cases = {"T1w": "T1w", "acq-MPRAGE_run-1_T1w": "T1w", "acq-FLAIR_run-1_T2w": "FLAIR", "T1c_bias": "T1c",
             "PDw": "PDw", "gre": "T2starw", "bval1000": "DWI", "UNIT1": "MP2RAGE", "T1map": "qmap", "m0scan": "ASL",
             "t1 axial stealth-post": "T1c", "t2_Flair_axial": "FLAIR", "Axial T2 tse": "T2w", "MESE": "other"}
    assert {k: modality(k) for k in cases} == cases
    assert [sex(v) for v in ("male", "F", "Female")] == ["M", "F", "F"] and np.isnan(sex("n/a"))
    assert [hand(v) for v in ("right", "Ambi", "L")] == ["R", "A", "L"]
    assert dicom_age("025Y") == 25 and dicom_age("006M") == 0.5 and np.isnan(dicom_age(""))
    groups = {"Typically Developing": "control", "CN": "control", "Glioma WHO II": "tumor", "AD": "dementia",
              "Parkinson’s disease": "parkinson", "autism": "autism", "Stroke": "stroke", "dyslexia": "other"}
    assert {k: dx_group(dx(k)) for k in groups} == groups

#!module load StdEnv/2020 gcc/9.3.0 opencv python/3.8 scipy-stack hdf5 geos/3.10.2 arrow/7.0.0
#!source ~/HIPT_Embedding_Env/bin/activate
import sys
import os
sys.path.append(os.path.abspath('/home/sorkwos/projects/rrg-senger-ab/multimodality/contrastive_learning/tab-transformer-pytorch'))

import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import glob
import xmltodict
from IPython.display import display, HTML

# -------------------------------------------------------------------------
# 1. CLINICAL DATA PREPARATION FUNCTIONS (unchanged)
# -------------------------------------------------------------------------
def drop_constant_columns(df):
    df = df.dropna(axis=1, how='all')
    nunique = df.apply(pd.Series.nunique, dropna=False)
    cols_to_drop = nunique[nunique <= 1].index
    df = df.drop(cols_to_drop, axis=1)
    return df

def display_scrollable_dataframe(df, max_rows=20):
    display(HTML(df.to_html(max_rows=max_rows, classes='table table-striped table-bordered table-hover')))

def extract_tags_and_values(elem, parent_tag="", tag_count=None):
    if tag_count is None:
        tag_count = {}
    data = {}
    for child in elem:
        base_tag = child.tag.split('}')[-1]
        sequence = child.attrib.get('sequence')

        if parent_tag:
            full_tag = f"{parent_tag}.{base_tag}"
        else:
            full_tag = base_tag

        if sequence:
            full_tag += f"_seq_{sequence}"
        elif full_tag in tag_count:
            tag_count[full_tag] += 1
            full_tag += f"_{tag_count[full_tag]}"
        else:
            tag_count[full_tag] = 1

        if child.text and child.text.strip():
            data[full_tag] = child.text.strip()

        data.update(extract_tags_and_values(child, full_tag, tag_count))
    return data

def process_xml_files(root_folder):
    all_data = []
    xml_files = glob.glob(os.path.join(root_folder, '**/*.xml'), recursive=True)
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        patient_data = extract_tags_and_values(root)
        all_data.append(patient_data)

    df = pd.DataFrame(all_data)
    df_cleaned = drop_constant_columns(df)
    return df_cleaned

def extract_and_organize_followup(df, prefix="follow_up_seq"):
    follow_up_columns = [col for col in df.columns if prefix in col]
    seq_numbers = sorted(set([col.split(f"{prefix}_")[1].split('_')[0] for col in follow_up_columns if f"{prefix}_" in col]))
    organized_columns = []
    for seq in seq_numbers:
        seq_columns = [col for col in follow_up_columns if f"{prefix}_{seq}" in col]
        organized_columns.extend(seq_columns)
    return organized_columns

def extract_and_organize_drugs(df, base_prefix="patient.drugs.drug"):
    drug_columns = [col for col in df.columns if base_prefix in col]
    drug_1_columns = [col for col in drug_columns if base_prefix + "_" not in col]
    drug_seq_numbers = sorted(set([int(col.split(f"{base_prefix}_")[1].split('.')[0])
                                   for col in drug_columns if f"{base_prefix}_" in col]))
    organized_columns = drug_1_columns
    for seq in drug_seq_numbers:
        seq_columns = [col for col in drug_columns if f"{base_prefix}_{seq}." in col]
        organized_columns.extend(seq_columns)
    return organized_columns

def map_icd_to_site(icd_code):
    if icd_code in [
        'C02.9','C04.9','C06.9','C06.0','C03.9','C00.9','C05.0','C03.1','C04.0',
        'C06.2','C02.1','C05.9','C03.0','C02.2'
    ]:
        return 'Section 3'
    elif icd_code in ['C32.9','C32.1']:
        return 'Section 5'
    elif icd_code == 'C14.8':
        return 'Section 9'
    elif icd_code in ['C09.9','C01','C10.9','C10.3','C13.9']:
        return 'Section 4'
    elif icd_code == 'C41.1':
        return 'Section 27'
    else:
        return 'Unknown Site'

def map_section_3_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th):
    if t_stage_6th == 'T4':
        return 'T4a', n_stage_6th, m_stage_6th
    else:
        return t_stage_6th, n_stage_6th, m_stage_6th

def map_section_4_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th, tissue_or_organ):
    if tissue_or_organ in [
        'Oral Tongue','Oral Cavity','Floor of mouth','Tonsil','Base of tongue',
        'Buccal Mucosa','Alveolar Ridge','Hard Palate','Lip','Oropharynx',
        'Hypopharynx','Larynx'
    ]:
        if t_stage_6th == 'T4':
            return 'T4a', n_stage_6th, m_stage_6th
        else:
            return t_stage_6th, n_stage_6th, m_stage_6th
    else:
        return t_stage_6th, n_stage_6th, m_stage_6th

def map_section_5_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th):
    if t_stage_6th == 'T4':
        return 'T4a', n_stage_6th, m_stage_6th
    else:
        return t_stage_6th, n_stage_6th, m_stage_6th

def map_section_9_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th):
    return t_stage_6th, n_stage_6th, m_stage_6th

def map_section_27_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th):
    return t_stage_6th, n_stage_6th, m_stage_6th

def map_ajcc_6th_to_7th(icd_code, t_stage_6th, n_stage_6th, m_stage_6th, tissue_or_organ, grade=None):
    section = map_icd_to_site(icd_code)
    if section == 'Section 3':
        return map_section_3_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th)
    elif section == 'Section 4':
        return map_section_4_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th, tissue_or_organ)
    elif section == 'Section 5':
        return map_section_5_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th)
    elif section == 'Section 9':
        return map_section_9_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th)
    elif section == 'Section 27':
        return map_section_27_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th)
    else:
        return 'Unknown Section', t_stage_6th, n_stage_6th, m_stage_6th

def map_section_3_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th):
    if t_stage_5th == 'T4':
        return 'T4a', n_stage_5th, m_stage_5th
    else:
        return t_stage_5th, n_stage_5th, m_stage_5th

def map_section_4_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th, tissue_or_organ):
    if tissue_or_organ in [
        'Oral Tongue','Oral Cavity','Floor of mouth','Tonsil','Base of tongue',
        'Buccal Mucosa','Alveolar Ridge','Hard Palate','Lip','Oropharynx',
        'Hypopharynx','Larynx'
    ]:
        if t_stage_5th == 'T4':
            return 'T4a', n_stage_5th, m_stage_5th
        else:
            return t_stage_5th, n_stage_5th, m_stage_5th
    else:
        return t_stage_5th, n_stage_5th, m_stage_5th

def map_section_5_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th):
    if t_stage_5th == 'T4':
        return 'T4a', n_stage_5th, m_stage_5th
    else:
        return t_stage_5th, n_stage_5th, m_stage_5th

def map_section_9_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th):
    return t_stage_5th, n_stage_5th, m_stage_5th

def map_section_27_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th):
    if m_stage_5th == 'M1':
        return t_stage_5th, n_stage_5th, 'M1a'
    else:
        return t_stage_5th, n_stage_5th, m_stage_5th

def map_ajcc_5th_to_6th(icd_code, t_stage_5th, n_stage_5th, m_stage_5th, tissue_or_organ):
    section = map_icd_to_site(icd_code)
    if section == 'Section 3':
        return map_section_3_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th)
    elif section == 'Section 4':
        return map_section_4_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th, tissue_or_organ)
    elif section == 'Section 5':
        return map_section_5_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th)
    elif section == 'Section 9':
        return map_section_9_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th)
    elif section == 'Section 27':
        return map_section_27_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th)
    else:
        return t_stage_5th, n_stage_5th, m_stage_5th

def map_clinical_5th_to_6th(row):
    clinical_t, clinical_n, clinical_m = map_ajcc_5th_to_6th(
        icd_code=row['patient.icd_10'],
        t_stage_5th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_T'],
        n_stage_5th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_N'],
        m_stage_5th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_M'],
        tissue_or_organ=row['patient.anatomic_neoplasm_subdivision']
    )
    return pd.Series({
        'ajcc_clinical_t': clinical_t,
        'ajcc_clinical_n': clinical_n,
        'ajcc_clinical_m': clinical_m
    })

def map_clinical_6th_to_7th(row):
    clinical_t, clinical_n, clinical_m = map_ajcc_6th_to_7th(
        icd_code=row['patient.icd_10'],
        t_stage_6th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_T'],
        n_stage_6th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_N'],
        m_stage_6th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_M'],
        tissue_or_organ=row['patient.anatomic_neoplasm_subdivision']
    )
    return pd.Series({
        'ajcc_clinical_t': clinical_t,
        'ajcc_clinical_n': clinical_n,
        'ajcc_clinical_m': clinical_m
    })

def map_clinical_and_pathologic_5th_to_6th(row):
    clinical_t, clinical_n, clinical_m = map_ajcc_5th_to_6th(
        icd_code=row['patient.icd_10'],
        t_stage_5th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_T'],
        n_stage_5th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_N'],
        m_stage_5th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_M'],
        tissue_or_organ=row['patient.anatomic_neoplasm_subdivision']
    )
    pathologic_t, pathologic_n, pathologic_m = map_ajcc_5th_to_6th(
        icd_code=row['patient.icd_10'],
        t_stage_5th=row['patient.stage_event.tnm_categories.pathologic_categories.pathologic_T'],
        n_stage_5th=row['patient.stage_event.tnm_categories.pathologic_categories.pathologic_N'],
        m_stage_5th=row['patient.stage_event.tnm_categories.pathologic_categories.pathologic_M'],
        tissue_or_organ=row['patient.anatomic_neoplasm_subdivision']
    )
    return pd.Series({
        'patient.stage_event.tnm_categories.clinical_categories.clinical_T': clinical_t,
        'patient.stage_event.tnm_categories.clinical_categories.clinical_N': clinical_n,
        'patient.stage_event.tnm_categories.clinical_categories.clinical_M': clinical_m,
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_T': pathologic_t,
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_N': pathologic_n,
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_M': pathologic_m
    })

def map_clinical_and_pathologic_6th_to_7th(row):
    clinical_t, clinical_n, clinical_m = map_ajcc_6th_to_7th(
        icd_code=row['patient.icd_10'],
        t_stage_6th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_T'],
        n_stage_6th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_N'],
        m_stage_6th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_M'],
        tissue_or_organ=row['patient.anatomic_neoplasm_subdivision']
    )
    pathologic_t, pathologic_n, pathologic_m = map_ajcc_6th_to_7th(
        icd_code=row['patient.icd_10'],
        t_stage_6th=row['patient.stage_event.tnm_categories.pathologic_categories.pathologic_T'],
        n_stage_6th=row['patient.stage_event.tnm_categories.pathologic_categories.pathologic_N'],
        m_stage_6th=row['patient.stage_event.tnm_categories.pathologic_categories.pathologic_M'],
        tissue_or_organ=row['patient.anatomic_neoplasm_subdivision']
    )
    return pd.Series({
        'patient.stage_event.tnm_categories.clinical_categories.clinical_T': clinical_t,
        'patient.stage_event.tnm_categories.clinical_categories.clinical_N': clinical_n,
        'patient.stage_event.tnm_categories.clinical_categories.clinical_M': clinical_m,
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_T': pathologic_t,
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_N': pathologic_n,
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_M': pathologic_m
    })

# -------------------------------------------------------------------------
# 2. LOAD AND CLEAN CLINICAL DATA (unchanged)
# -------------------------------------------------------------------------
root_folder = '/home/sorkwos/tcga_hnsc_xml_clinical'
df = process_xml_files(root_folder)

radiation_columns = extract_and_organize_drugs(df,"patient.radiations.radiation")
reordered_columns_df_1 = pd.concat([df.drop(columns=radiation_columns), df[radiation_columns]], axis=1)

drug_columns = extract_and_organize_drugs(reordered_columns_df_1)
reordered_columns_df_2 = pd.concat([reordered_columns_df_1.drop(columns=drug_columns), reordered_columns_df_1[drug_columns]], axis=1)

follow_up_columns = extract_and_organize_followup(reordered_columns_df_2)
reordered_columns_df_3 = pd.concat([reordered_columns_df_2.drop(columns=follow_up_columns), reordered_columns_df_2[follow_up_columns]], axis=1)

new_df = reordered_columns_df_3.loc[:, ~reordered_columns_df_3.columns.str.contains("patient.follow_ups|patient.radiations|patient.drugs")]
new_df = new_df.drop(["admin.file_uuid", "admin.batch_number", "patient.patient_id"], axis=1, errors='ignore')

new_df = new_df.drop(columns=[
    "patient.radiation_therapy",
    "patient.postoperative_rx_tx",
    "patient.primary_therapy_outcome_success",
    "patient.new_tumor_events.new_tumor_event_after_initial_treatment",
    "patient.new_tumor_events.new_tumor_event.days_to_new_tumor_event_after_initial_treatment",
    "patient.new_tumor_events.new_tumor_event.new_neoplasm_event_occurrence_anatomic_site",
    "patient.new_tumor_events.new_tumor_event.new_neoplasm_occurrence_anatomic_site_text",
    "patient.new_tumor_events.new_tumor_event.progression_determined_by",
    "patient.new_tumor_events.new_tumor_event.new_tumor_event_additional_surgery_procedure",
    "patient.new_tumor_events.new_tumor_event.additional_radiation_therapy",
    "patient.new_tumor_events.new_tumor_event.additional_pharmaceutical_therapy",
    "patient.new_tumor_events.new_tumor_event.new_neoplasm_event_type",
    "patient.new_tumor_events.new_tumor_event.days_to_new_tumor_event_additional_surgery_procedure",
    "patient.vital_status",
    "patient.days_to_last_followup",
    "patient.days_to_death",
    "patient.days_to_last_known_alive",
    "patient.history_of_neoadjuvant_treatment",
    "patient.person_neoplasm_cancer_status",
], errors='ignore')

new_df = new_df.drop(columns=[
    "patient.tissue_prospective_collection_indicator",
    "patient.tissue_retrospective_collection_indicator",
    "patient.tissue_source_site",
    "patient.days_to_initial_pathologic_diagnosis",
    "patient.year_of_initial_pathologic_diagnosis",
    "patient.day_of_form_completion",
    "patient.month_of_form_completion",
    "patient.year_of_form_completion",
    "patient.age_at_initial_pathologic_diagnosis"
], errors='ignore')

clinical_data_filtered = new_df.copy()

mask_5th = clinical_data_filtered['patient.stage_event.system_version'].isin(['4th','5th'])
df_filtered_5th = clinical_data_filtered[mask_5th].copy()
mapped_values_5th = df_filtered_5th.apply(map_clinical_and_pathologic_5th_to_6th, axis=1)
df_filtered_5th.update(mapped_values_5th)
clinical_data_filtered.update(df_filtered_5th)

mask_6th_combined = clinical_data_filtered['patient.stage_event.system_version'].isin(['4th','5th','6th'])
df_filtered_6th = clinical_data_filtered[mask_6th_combined].copy()
mapped_values_6th_to_7th = df_filtered_6th.apply(map_clinical_and_pathologic_6th_to_7th, axis=1)
df_filtered_6th.update(mapped_values_6th_to_7th)
clinical_data_filtered.update(df_filtered_6th)

clinical_data_updated = clinical_data_filtered

clinical_data_updated['patient.days_to_birth'] = pd.to_numeric(clinical_data_updated['patient.days_to_birth'], errors='coerce')
clinical_data_updated['patient.days_to_birth'] = clinical_data_updated['patient.days_to_birth'].apply(
    lambda x: abs(x) if pd.notnull(x) and x < 0 else x
)

cols_to_convert = [
    'patient.number_of_lymphnodes_positive_by_ihc',
    'patient.number_of_lymphnodes_positive_by_he',
    'patient.lymph_node_examined_count'
]
clinical_data_updated[cols_to_convert] = clinical_data_updated[cols_to_convert].apply(pd.to_numeric, errors='coerce')

mask_nan = clinical_data_updated[cols_to_convert].isna().any(axis=1)
mask_zero_examined = clinical_data_updated['patient.lymph_node_examined_count'] == 0
mask_invalid = mask_nan | mask_zero_examined

clinical_data_updated['patient.lymphnodes_ratio_positive_by_ihc'] = np.where(
    mask_invalid,
    -1,
    clinical_data_updated['patient.number_of_lymphnodes_positive_by_ihc'] / clinical_data_updated['patient.lymph_node_examined_count']
)
clinical_data_updated['patient.lymphnodes_ratio_positive_by_he'] = np.where(
    mask_invalid,
    -1,
    clinical_data_updated['patient.number_of_lymphnodes_positive_by_he'] / clinical_data_updated['patient.lymph_node_examined_count']
)
clinical_data_updated.drop(columns=cols_to_convert, inplace=True, errors='ignore')

clinical_data_final = clinical_data_updated.drop('patient.stage_event.system_version', axis=1, errors='ignore')

file_path = "/home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/TCGA-CDR-SupplementalTableS1.xlsx"
df_endpoints = pd.read_excel(file_path)
df_hnsc = df_endpoints[df_endpoints['type'].str.contains("HNSC", na=False)]

file_path_hiv = "/home/sorkwos/HIV_TCGA.xls"
df_hiv_in = pd.read_excel(file_path_hiv, header=1)
df_hiv = df_hiv_in[df_hiv_in['Study'].str.contains("HNSC", na=False)]

df_hnsc_filtered = df_hnsc[['bcr_patient_barcode','PFI','PFI.time']]
clinical_data_final_updated = clinical_data_final.merge(
    df_hnsc_filtered,
    left_on='patient.bcr_patient_barcode',
    right_on='bcr_patient_barcode',
    how='left'
)
df_hiv = df_hiv[['SampleBarcode','HPV load','HPV.status']].copy()
df_hiv['SampleBarcode_short'] = df_hiv['SampleBarcode'].str[:12]
clinical_data_final_updated = clinical_data_final_updated.merge(
    df_hiv[['SampleBarcode_short','HPV load','HPV.status']],
    how='left',
    left_on='patient.bcr_patient_barcode',
    right_on='SampleBarcode_short'
).drop(columns=['SampleBarcode_short'], errors='ignore')
clinical_data_final_updated['HPV load'] = clinical_data_final_updated['HPV load'].fillna(-1)

clinical_data_yes_id = clinical_data_final_updated.drop(columns=['patient.bcr_patient_uuid'], errors='ignore')
clinical_data_yes = clinical_data_final_updated.drop(columns=['patient.bcr_patient_barcode','patient.bcr_patient_uuid'], errors='ignore')

df_hnsc_filtered_os = df_hnsc[['bcr_patient_barcode','OS','OS.time']]
clinical_data_final_updated_os = clinical_data_final.merge(
    df_hnsc_filtered_os,
    left_on='patient.bcr_patient_barcode',
    right_on='bcr_patient_barcode',
    how='left'
)
df_hiv_in = pd.read_excel(file_path_hiv, header=1)
df_hiv = df_hiv_in[df_hiv_in['Study'].str.contains("HNSC", na=False)]
df_hiv = df_hiv[['SampleBarcode','HPV load','HPV.status']].copy()
df_hiv['SampleBarcode_short'] = df_hiv['SampleBarcode'].str[:12]
clinical_data_final_updated_os = clinical_data_final_updated_os.merge(
    df_hiv[['SampleBarcode_short','HPV load','HPV.status']],
    how='left',
    left_on='patient.bcr_patient_barcode',
    right_on='SampleBarcode_short'
).drop(columns=['SampleBarcode_short'], errors='ignore')
clinical_data_final_updated_os['HPV load'] = clinical_data_final_updated_os['HPV load'].fillna(-1)

clinical_data_final_update_os = clinical_data_final_updated_os.drop(columns=['bcr_patient_barcode'], errors='ignore')
clinical_data_yes_id_os = clinical_data_final_updated_os.drop(columns=['patient.bcr_patient_uuid'], errors='ignore')
clinical_data_yes_os = clinical_data_final_updated_os.drop(columns=['patient.bcr_patient_barcode','patient.bcr_patient_uuid'], errors='ignore')

# -------------------------------------------------------------------------
# 3. CONTRASTIVE LEARNING & EMBEDDING REFINEMENT
# -------------------------------------------------------------------------
def run_contrastive_script(json_file_path, embeddings_folder, label_type):
    import json
    import os
    import torch
    import numpy as np
    import pandas as pd
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from networks.resnet_big import SupConEmbeddingNet, EnhancedSupConEmbeddingNet
    from losses import SupConLoss
    from util import TwoCropTransform, adjust_learning_rate, set_optimizer, save_model
    import argparse
    import time
    from sklearn.metrics.pairwise import cosine_distances
    import matplotlib.pyplot as plt

    def set_seeds(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    set_seeds(42)

    with open(json_file_path, 'r') as f:
        metadata = json.load(f)

    file_to_label = {}
    for entry in metadata:
        if pd.isnull(entry.get('censoring')) or pd.isnull(entry.get('time_to_event')):
            continue
        file_name_svs = entry['file_name']
        file_name = file_name_svs.replace('.svs','_flatten.pt')
        if "DX1" not in file_name:
            continue
        label_value = entry.get(label_type, -1)
        if label_value == -1:
            print(f"Info: Label '{label_type}' not found for file '{file_name}'. Assigning label_value as -1.")
        file_to_label[file_name] = label_value

    count = 0
    data_list = []
    counter_dx = 0
    for entry in metadata:
        if pd.isnull(entry.get('censoring')) or pd.isnull(entry.get('time_to_event')):
            continue
        file_name_svs = entry['file_name']
        file_name = file_name_svs.replace('.svs','_flatten.pt')
        if "DX1" not in file_name:
            continue
        file_path = os.path.join(embeddings_folder, file_name)
        counter_dx += 1
        if os.path.exists(file_path):
            embedding = torch.load(file_path, map_location='cpu').numpy().astype(np.float32).flatten()
            censoring = int(entry['censoring'])
            time_to_death = float(entry['time_to_event'])
            data_list.append(np.concatenate([embedding, [time_to_death, censoring, file_name]]))
            count += 1
            if count % 10 == 0:
                print(count)
        else:
            print(f"File {file_path} does not exist.")

    print("Total diagnostic slides is", str(counter_dx))
    print("done loading the data")
    print("total slides is:", count)

    columns = []
    if len(data_list) > 0:
        columns = [f'x{i}' for i in range(len(data_list[0]) - 3)] + ['time','event','file_name']
    df_ = pd.DataFrame(data_list, columns=columns)
    print(f"DataFrame shape: {df_.shape}")

    df_['label'] = df_['file_name'].map(file_to_label).fillna(-1).astype(int)
    missing_embeddings = df_[df_['label'] == -1]
    if not missing_embeddings.empty:
        print(f"Warning: {len(missing_embeddings)} embeddings do not have corresponding labels and will be excluded.")

    initial_count = len(df_)
    df_ = df_[df_['label'] != -1].reset_index(drop=True)
    filtered_count = len(df_)
    print(f"Filtered out {initial_count - filtered_count} entries with label -1.")

    if df_['file_name'].duplicated().any():
        duplicate_files = df_[df_['file_name'].duplicated(keep=False)]['file_name'].unique()
        print(f"Duplicate file names found: {duplicate_files}")
    else:
        print("No duplicate file names found.")

    for i in range(min(5, len(df_))):
        file_name_ = df_.loc[i, 'file_name']
        label_ = df_.loc[i, 'label']
        print(f"File: {file_name_}, Label: {label_}")

    print(f"Total embeddings after filtering: {len(df_)}")

    embeddings = df_.drop(['time','event','label','file_name'], axis=1).values.astype(np.float32)
    labels = df_['label'].values.astype(np.int64)

    for i in range(min(5, len(embeddings))):
        print(f"Embedding {i}: {embeddings[i][:5]}, Label: {labels[i]}")
    print(len(embeddings))
    print(len(labels))

    class EmbeddingDataset(Dataset):
        def __init__(self, embeddings, labels, transform=None):
            self.embeddings = embeddings
            self.labels = labels
            self.transform = transform
        def __len__(self):
            return len(self.embeddings)
        def __getitem__(self, idx):
            emb_ = self.embeddings[idx]
            if self.transform:
                emb_ = self.transform(emb_)
            lab_ = self.labels[idx]
            return emb_, lab_

    def augment_embedding(x_):
        x_ = torch.tensor(x_)
        x_ = x_ + torch.randn_like(x_) * 0.01
        x_ = F.dropout(x_, p=0.1, training=True)
        x_ = x_ * (1 + 0.1 * torch.randn_like(x_))
        return x_

    from util import TwoCropTransform

    def set_embedding_loader(embs, lbls, batch_size=32, num_workers=4):
        transform_ = TwoCropTransform(augment_embedding)
        dataset_ = EmbeddingDataset(embs, lbls, transform=transform_)
        loader_ = DataLoader(dataset_, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return loader_

    def parse_option():
        parser = argparse.ArgumentParser('argument for training')
        parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
        parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
        parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
        parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
        parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
        parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
        parser.add_argument('--lr_decay_epochs', type=str, default='180,190', help='where to decay lr, can be a list')
        parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--method', type=str, default='SupCon', choices=['SupCon','SimCLR'], help='choose method')
        parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')
        parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
        parser.add_argument('--syncBN', action='store_true', help='using synchronized batch normalization')
        parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
        parser.add_argument('--save_folder', type=str, default='./models', help='folder to save models')
        opt_ = parser.parse_args(args=[])
        if not os.path.isdir(opt_.save_folder):
            os.makedirs(opt_.save_folder)
        opt_.lr_decay_epochs = list(map(int, opt_.lr_decay_epochs.split(',')))
        return opt_

    from util import adjust_learning_rate, set_optimizer, save_model
    grads = []
    def save_grad(name_):
        def hook(grad_):
            grads.append((name_, grad_.clone()))
        return hook

    def set_model(opt_):
        model_ = EnhancedSupConEmbeddingNet(input_dim=embeddings.shape[1], feat_dim=embeddings.shape[1])
        criterion_ = SupConLoss(temperature=opt_.temp)
        if opt_.syncBN:
            model_ = model_.sync_batchnorm()
        if torch.cuda.is_available():
            model_ = model_.cuda()
            criterion_ = criterion_.cuda()
        for name_, param_ in model_.named_parameters():
            param_.register_hook(save_grad(name_))
        return model_, criterion_

    def train(train_loader_, model_, criterion_, optimizer_, epoch_, opt_):
        model_.train()
        epoch_loss_ = 0
        for idx_, (batch_embeddings, batch_labels) in enumerate(train_loader_):
            batch_embeddings = torch.cat([batch_embeddings[0], batch_embeddings[1]], dim=0)
            if torch.cuda.is_available():
                batch_embeddings, batch_labels = batch_embeddings.cuda(), batch_labels.cuda()
            optimizer_.zero_grad()
            features_ = model_(batch_embeddings)
            f1_, f2_ = torch.split(features_, [batch_embeddings.size(0)//2, batch_embeddings.size(0)//2], dim=0)
            features_ = torch.cat([f1_.unsqueeze(1), f2_.unsqueeze(1)], dim=1)
            loss_ = criterion_(features_, batch_labels)
            loss_.backward()
            optimizer_.step()
            epoch_loss_ += loss_.item()
            if (idx_+1) % opt_.print_freq == 0:
                print(f"Epoch [{epoch_}/{opt_.epochs}], Step [{idx_+1}/{len(train_loader_)}], Loss: {loss_.item():.4f}")
        avg_loss_ = epoch_loss_ / len(train_loader_)
        print(f"Epoch [{epoch_}/{opt_.epochs}] Average Loss: {avg_loss_:.4f}")
        return avg_loss_

    opt = parse_option()
    train_loader = set_embedding_loader(embeddings, labels, batch_size=opt.batch_size, num_workers=opt.num_workers)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)

    for epoch in range(1, opt.epochs+1):
        adjust_learning_rate(opt, optimizer, epoch)
        avg_loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        print(f"End of Epoch [{epoch}/{opt.epochs}], Average Loss: {avg_loss:.4f}")
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_file)

    print("===== MODEL STATE DICT =====")
    for k_, v_ in model.state_dict().items():
        print(f"{k_} -> shape: {list(v_.shape)}")

    print("\n===== BATCHNORM LAYER DETAILS =====")
    for name_, submodule_ in model.named_modules():
        import torch.nn as nn
        if isinstance(submodule_, nn.BatchNorm1d) or isinstance(submodule_, nn.BatchNorm2d):
            print(
                f"Layer: {name_}, "
                f"num_features: {submodule_.num_features}, "
                f"running_mean shape: {list(submodule_.running_mean.shape)}, "
                f"running_var shape: {list(submodule_.running_var.shape)}"
            )
    print(model)
    save_file = os.path.join(opt.save_folder, f'{label_type}.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

def refine_embeddings(
    name: str,
    embeddings_folder: str,
    model_dir: str = "models",
    output_base_dir: str = ".",
    input_dim: int = 192,
    feat_dim: int = 192,
    filter_condition=lambda filename: filename.endswith('.pt') and "DX1" in filename
):
    import torch
    import os
    from networks.resnet_big import EnhancedSupConEmbeddingNet
    import torch.nn as nn

    model_path = os.path.join(model_dir, f"{name}.pth")
    output_folder = os.path.join(output_base_dir, name)
    os.makedirs(output_folder, exist_ok=True)

    model = EnhancedSupConEmbeddingNet(input_dim=input_dim, feat_dim=feat_dim)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    checkpoint = torch.load(model_path)
    if 'model' not in checkpoint:
        raise KeyError(f"'model' key not found in the checkpoint at {model_path}")
    model.load_state_dict(checkpoint['model'])
    model.eval()

    counter = 0
    for filename in os.listdir(embeddings_folder):
        if filter_condition(filename):
            original_path = os.path.join(embeddings_folder, filename)
            try:
                embedding = torch.load(original_path, map_location='cpu').float()
            except Exception as e:
                print(f"Failed to load {original_path}: {e}")
                continue
            with torch.no_grad():
                try:
                    refined_embedding = model(embedding)
                except Exception as e:
                    print(f"Failed to refine embedding {filename}: {e}")
                    continue
            refined_path = os.path.join(output_folder, filename)
            try:
                torch.save(refined_embedding.squeeze(0), refined_path)
                print(f"Refined embedding saved: {refined_path}")
                counter += 1
            except Exception as e:
                print(f"Failed to save refined embedding {refined_path}: {e}")

    print("All embeddings have been refined and saved.")
    print(f"The total number of refined embeddings is {counter}")
    return counter

# -------------------------------------------------------------------------
# 4. MAIN DEEP LEARNING & SURVIVAL STRATIFICATION FUNCTION
#    WITH PARALLEL NESTED CV AND (OPTIONAL) BIASED CV
# -------------------------------------------------------------------------
def run_deep_learning_strat_contrastive(
    run_selected=False, 
    selected=None, 
    selected_name='_',
    run_biased_cv=False  # <--- NEW FLAG: set to True to also do single-level (biased) CV
):
    """
    Performs nested cross-validation on clinical + WSI embeddings for risk stratification,
    plus optional single-level (biased) cross-validation if `run_biased_cv=True`.
    """
    import os
    import sys
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from tab_transformer_pytorch import FTTransformer
    from pycox.models import CoxPH
    from sksurv.metrics import concordance_index_censored
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    import torchtuples as tt
    import warnings
    import random
    import json
    import seaborn as sns
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time as t

    warnings.filterwarnings("ignore")
    sns.set(style="whitegrid")
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -------------------------------------------
    # Use the 'clinical_data_yes_id' from above
    # -------------------------------------------
    clinical_data_test = clinical_data_yes_id.copy()

    json_path = 'svs_patient_map_PFI.json'
    split_json_path = 'svs_patient_map_PFI_DX_split.json'
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"The JSON file '{json_path}' was not found.")
    if not os.path.exists(split_json_path):
        raise FileNotFoundError(f"The JSON file '{split_json_path}' was not found.")

    with open(json_path, 'r') as f:
        survival_entries = json.load(f)
    wsi_df = pd.DataFrame(survival_entries)

    def extract_barcode(file_name):
        return file_name[:12]

    wsi_df['patient.bcr_patient_barcode'] = wsi_df['file_name'].apply(extract_barcode)
    wsi_df = wsi_df[wsi_df['file_name'].str.contains('DX1', case=False, na=False)].reset_index(drop=True)

    clinical_data_test_filtered = clinical_data_test[
        clinical_data_test["patient.bcr_patient_barcode"].isin(wsi_df["patient.bcr_patient_barcode"])
    ].copy()
    clinical_data_test = clinical_data_test_filtered.copy()

    with open(split_json_path, 'r') as f:
        split_entries = json.load(f)
    split_df = pd.DataFrame(split_entries)
    split_df['patient.bcr_patient_barcode'] = split_df['file_name'].apply(extract_barcode)
    barcode_to_split = dict(zip(split_df['patient.bcr_patient_barcode'], split_df['split']))
    clinical_data_test['split'] = clinical_data_test['patient.bcr_patient_barcode'].map(barcode_to_split)
    clinical_data_test = clinical_data_test.dropna(subset=['split'])

    train_mask = clinical_data_test['split'] == 'train'
    test_mask = clinical_data_test['split'] == 'test'

    patient_barcodes = clinical_data_test['patient.bcr_patient_barcode'].reset_index(drop=True)
    clinical_data_test = clinical_data_test.drop(columns=['split'], errors='ignore')

    columns_to_drop = [
        'patient.stage_event.clinical_stage',
        'patient.stage_event.tnm_categories.clinical_categories.clinical_T',
        'patient.stage_event.tnm_categories.clinical_categories.clinical_N',
        'patient.stage_event.tnm_categories.clinical_categories.clinical_M',
    ]
    clinical_data_test.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    t_mapping = {'TX':-1,'T0':0,'T1':1,'T2':2,'T3':3,'T4':4,'T4A':5,'T4B':6}
    n_mapping = {'NX':-1,'N0':0,'N1':1,'N2':2,'N2A':3,'N2B':4,'N2C':5,'N3':6}
    m_mapping = {'MX':-1,'M0':0,'M1':1}
    stage_mapping = {
        'STAGE I':1,'STAGE II':2,'STAGE III':3,'STAGE IV':4,
        'STAGE IVA':5,'STAGE IVB':6,'STAGE IVC':7
    }

    def encode_t(value):
        if pd.isnull(value): return -1
        return t_mapping.get(str(value).upper().strip(), -1)
    def encode_n(value):
        if pd.isnull(value): return -1
        return n_mapping.get(str(value).upper().strip(), -1)
    def encode_m(value):
        if pd.isnull(value): return -1
        return m_mapping.get(str(value).upper().strip(), -1)
    def encode_stage(value):
        if pd.isnull(value): return -1
        return stage_mapping.get(str(value).upper().strip(), -1)

    clinical_data_test['patient.stage_event.tnm_categories.pathologic_categories.pathologic_T'] = \
        clinical_data_test['patient.stage_event.tnm_categories.pathologic_categories.pathologic_T'].apply(encode_t)
    clinical_data_test['patient.stage_event.tnm_categories.pathologic_categories.pathologic_N'] = \
        clinical_data_test['patient.stage_event.tnm_categories.pathologic_categories.pathologic_N'].apply(encode_n)
    clinical_data_test['patient.stage_event.tnm_categories.pathologic_categories.pathologic_M'] = \
        clinical_data_test['patient.stage_event.tnm_categories.pathologic_categories.pathologic_M'].apply(encode_m)
    clinical_data_test['patient.stage_event.pathologic_stage'] = \
        clinical_data_test['patient.stage_event.pathologic_stage'].apply(encode_stage)

    clinical_data_test['patient.days_to_birth'].fillna(
        clinical_data_test['patient.days_to_birth'].mean(), inplace=True)
    clinical_data_test['age_of_patient'] = clinical_data_test['patient.days_to_birth'] / 365.25

    num_cols = [
        'patient.lymphnodes_ratio_positive_by_ihc',
        'patient.lymphnodes_ratio_positive_by_he',
        'age_of_patient',
        'patient.tobacco_smoking_history',
        'patient.year_of_tobacco_smoking_onset',
        'patient.number_pack_years_smoked',
        'patient.stopped_smoking_year',
        'patient.frequency_of_alcohol_consumption',
        'patient.amount_of_alcohol_consumption_per_day',
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_T',
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_N',
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_M',
        'patient.stage_event.pathologic_stage',
        'HPV load'
    ]
    if run_selected and selected is not None:
        num_cols = list(set(num_cols) & set(selected))

    for col_ in num_cols:
        clinical_data_test[col_] = pd.to_numeric(clinical_data_test[col_], errors='coerce')

    tobacco_alcohol_cols = [
        'patient.lymphnodes_ratio_positive_by_ihc',
        'patient.lymphnodes_ratio_positive_by_he',
        'patient.tobacco_smoking_history',
        'patient.year_of_tobacco_smoking_onset',
        'patient.number_pack_years_smoked',
        'patient.stopped_smoking_year',
        'patient.frequency_of_alcohol_consumption',
        'patient.amount_of_alcohol_consumption_per_day',
        'HPV load'
    ]
    clinical_data_test[tobacco_alcohol_cols] = clinical_data_test[tobacco_alcohol_cols].fillna(-1)

    pathologic_cols = [
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_T',
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_N',
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_M',
        'patient.stage_event.pathologic_stage'
    ]
    clinical_data_test[pathologic_cols] = clinical_data_test[pathologic_cols].fillna(-1)

    categorical_cols = clinical_data_test.select_dtypes(include=['object']).columns
    if run_selected and selected is not None:
        categorical_cols = list(set(categorical_cols) & set(selected))

    clinical_data_test[categorical_cols] = clinical_data_test[categorical_cols].fillna("missing")
    for col_ in categorical_cols:
        clinical_data_test[col_] = clinical_data_test[col_].astype('category').cat.codes

    clinical_data_test['PFI.time'].fillna(clinical_data_test['PFI.time'].mean(), inplace=True)
    clinical_data_test['PFI'].fillna(0, inplace=True)

    time_ = clinical_data_test['PFI.time'].values.astype(float)
    event_ = clinical_data_test['PFI'].values.astype(bool)

    if num_cols:
        scaler_ = StandardScaler()
        scaled_num_cols_ = scaler_.fit_transform(clinical_data_test[num_cols])
        scaled_num_cols_df = pd.DataFrame(scaled_num_cols_, columns=num_cols, index=clinical_data_test.index)
    else:
        scaled_num_cols_df = pd.DataFrame(index=clinical_data_test.index)

    if not scaled_num_cols_df.empty and not clinical_data_test[categorical_cols].empty:
        X_ = pd.concat([scaled_num_cols_df, clinical_data_test[categorical_cols]], axis=1)
    elif not scaled_num_cols_df.empty:
        X_ = scaled_num_cols_df
    else:
        X_ = clinical_data_test[categorical_cols]

    clinical_data_test.reset_index(drop=True, inplace=True)
    X_.reset_index(drop=True, inplace=True)
    patient_barcodes.reset_index(drop=True, inplace=True)
    train_mask = train_mask.reset_index(drop=True)
    test_mask = test_mask.reset_index(drop=True)

    X_train_df = X_[train_mask]
    X_test_df = X_[test_mask]
    y_train_event = event_[train_mask]
    y_test_event = event_[test_mask]
    y_train_time = time_[train_mask]
    y_test_time = time_[test_mask]

    patient_barcodes_train = patient_barcodes[train_mask]
    patient_barcodes_test = patient_barcodes[test_mask]

    categorical_columns = list(categorical_cols)
    numerical_columns = num_cols

    def prepare_data(XX, cat_cols, num_cols):
        cat_features = XX[cat_cols] if cat_cols else pd.DataFrame()
        num_features = XX[num_cols] if num_cols else pd.DataFrame()
        return cat_features, num_features

    X_train_cat, X_train_num = prepare_data(X_train_df, categorical_columns, numerical_columns)
    X_test_cat, X_test_num = prepare_data(X_test_df, categorical_columns, numerical_columns)

    def to_tensor(xX, dtype_):
        return torch.tensor(xX.values, dtype=dtype_)

    X_train_cat_tensor = to_tensor(X_train_cat, torch.long) if not X_train_cat.empty else None
    X_train_num_tensor = to_tensor(X_train_num, torch.float32) if not X_train_num.empty else None
    X_test_cat_tensor  = to_tensor(X_test_cat, torch.long) if not X_test_cat.empty else None
    X_test_num_tensor  = to_tensor(X_test_num, torch.float32) if not X_test_num.empty else None

    if categorical_columns:
        categories = [int(max(X_train_cat[col].max(), X_test_cat[col].max()) + 1) for col in categorical_columns]
    else:
        categories = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_continuous = X_train_num_tensor.shape[1] if X_train_num_tensor is not None else 0

    ft_transformer = FTTransformer(
        categories=categories,
        num_continuous=num_continuous,
        dim=192,
        depth=6,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1
    ).to(device)
    ft_transformer.eval()

    def get_embeddings(model_, X_cat_, X_num_):
        model_.eval()
        with torch.no_grad():
            if X_cat_.empty:
                categorical_ = None
            else:
                categorical_ = torch.tensor(X_cat_.values, dtype=torch.long).to(device)
            if X_num_.empty:
                numerical_ = None
            else:
                numerical_ = torch.tensor(X_num_.values, dtype=torch.float32).to(device)
            embeddings_ = model_(categorical_, numerical_, return_embedding=True)
            return embeddings_.cpu().numpy()

    X_full = pd.concat([X_train_df, X_test_df], axis=0).reset_index(drop=True)
    y_event_full = np.concatenate([y_train_event, y_test_event])
    y_time_full  = np.concatenate([y_train_time,  y_test_time])
    patient_barcodes_full = pd.concat([patient_barcodes_train, patient_barcodes_test], ignore_index=True)

    X_full_cat, X_full_num = prepare_data(X_full, categorical_columns, numerical_columns)
    full_embeddings = get_embeddings(ft_transformer, X_full_cat, X_full_num)

    train_embeddings_full = full_embeddings
    test_embeddings = full_embeddings
    events_train_full = y_event_full.astype(np.int64)
    durations_train_full = y_time_full.astype(np.float32)

    num_nodes = [192,128,64]
    out_features = 1
    batch_norm = True
    dropout = 0.4
    output_bias = False

    # -----------------------------
    # WSI LOADING (original folder)
    # -----------------------------
    embeddings_dir = os.path.expanduser('~/scratch/TCGA-HNSC-embeddings-flatten')
    missing_files_ = []

    def load_embedding(file_name_):
        embedding_file_ = file_name_.replace('.svs','_flatten.pt')
        embedding_path_ = os.path.join(embeddings_dir, embedding_file_)
        if not os.path.exists(embedding_path_):
            missing_files_.append(embedding_file_)
            return None
        try:
            embedding_tensor_ = torch.load(embedding_path_)
            embedding_np_ = embedding_tensor_.numpy().astype(np.float32).flatten()
            return embedding_np_
        except Exception:
            missing_files_.append(embedding_file_)
            return None

    def load_embedding_parallel(file_name__):
        embedding_file__ = file_name__.replace('.svs','_flatten.pt')
        embedding_path__ = os.path.join(embeddings_dir, embedding_file__)
        if not os.path.exists(embedding_path__):
            missing_files_.append(embedding_file__)
            return None
        try:
            embedding_tensor__ = torch.load(embedding_path__)
            embedding_np__ = embedding_tensor__.numpy().astype(np.float32).flatten()
            return embedding_np__
        except Exception:
            missing_files_.append(embedding_file__)
            return None

    from concurrent.futures import ThreadPoolExecutor, as_completed
    def load_embeddings_in_parallel(wsi_df_local):
        embeddings_local_ = []
        with ThreadPoolExecutor(max_workers=8) as executor_:
            futures_ = {
                executor_.submit(load_embedding_parallel, row['file_name']): idx_
                for idx_, row in wsi_df_local.iterrows()
            }
            for future_ in tqdm(as_completed(futures_), total=len(futures_)):
                embeddings_local_.append(future_.result())
        return embeddings_local_

    wsi_df = wsi_df[wsi_df['file_name'].str.contains('DX1', case=False, na=False)].reset_index(drop=True)
    wsi_df['embedding'] = load_embeddings_in_parallel(wsi_df)
    wsi_df = wsi_df[wsi_df['embedding'].notnull()].reset_index(drop=True)
    barcode_to_wsi = dict(zip(wsi_df['patient.bcr_patient_barcode'], wsi_df['embedding']))

    wsi_embeddings_list = []
    wsi_mask = []
    for pb_ in patient_barcodes_full:
        if pb_ in barcode_to_wsi and barcode_to_wsi[pb_] is not None:
            wsi_embeddings_list.append(barcode_to_wsi[pb_])
            wsi_mask.append(True)
        else:
            wsi_mask.append(False)

    wsi_embeddings_array = np.array(wsi_embeddings_list, dtype=np.float32)
    wsi_mask = np.array(wsi_mask)

    # ---------------------------------------------------------------
    # Nested Cross-Validation
    # ---------------------------------------------------------------
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    outer_test_c_indexes = []
    outer_test_c_indexes_wsi_original = []
    outer_test_c_indexes_wsi_refined = []

    with open(split_json_path, 'r') as f_:
        orig_split_entries_ = json.load(f_)
    orig_split_df_ = pd.DataFrame(orig_split_entries_)
    orig_split_df_['patient.bcr_patient_barcode'] = orig_split_df_['file_name'].apply(extract_barcode)
    updated_split_df = orig_split_df_.copy()

    print("\nStarting Nested Cross-Validation (5 Outer Folds, each with 5 Inner Folds)...\n")
    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer_kf.split(train_embeddings_full), start=1):
        print(f"Outer Fold {outer_fold}/5")

        X_outer_train = train_embeddings_full[outer_train_idx]
        y_outer_train_events = events_train_full[outer_train_idx]
        y_outer_train_durations = durations_train_full[outer_train_idx]

        X_outer_val = train_embeddings_full[outer_val_idx]
        y_outer_val_events = events_train_full[outer_val_idx]
        y_outer_val_durations = durations_train_full[outer_val_idx]

        inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
        inner_c_indexes = []

        patient_barcodes_outer_train = patient_barcodes_full.iloc[outer_train_idx].reset_index(drop=True)
        patient_barcodes_outer_val   = patient_barcodes_full.iloc[outer_val_idx].reset_index(drop=True)

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_kf.split(X_outer_train), start=1):
            print(f"  Inner Fold {inner_fold}/5 of Outer Fold {outer_fold}")
            X_inner_train = X_outer_train[inner_train_idx]
            y_inner_train_dur = y_outer_train_durations[inner_train_idx]
            y_inner_train_evt = y_outer_train_events[inner_train_idx]

            X_inner_val = X_outer_train[inner_val_idx]
            y_inner_val_dur = y_outer_train_durations[inner_val_idx]
            y_inner_val_evt = y_outer_train_events[inner_val_idx]

            net_inner = tt.practical.MLPVanilla(
                in_features=X_inner_train.shape[1],
                num_nodes=num_nodes,
                out_features=out_features,
                batch_norm=batch_norm,
                dropout=dropout,
                output_bias=output_bias
            )
            optimizer_inner = tt.optim.Adam(lr=1e-3)
            cox_ph_inner = CoxPH(net_inner, optimizer_inner)
            cox_ph_inner.fit(
                X_inner_train,
                (y_inner_train_evt, y_inner_train_dur),
                batch_size=64,
                epochs=100,
                verbose=False
            )
            risk_val_inner = cox_ph_inner.predict(X_inner_val).squeeze()

            from sksurv.metrics import concordance_index_censored
            c_index_inner = concordance_index_censored(
                y_inner_val_evt.astype(bool),
                y_inner_val_dur,
                -risk_val_inner
            )[0]
            print(f"  Inner Fold {inner_fold}, Outer Fold {outer_fold} C-Index (Clinical Only): {c_index_inner:.4f}")
            inner_c_indexes.append(c_index_inner)

            # Original WSI c-index at inner fold
            wsi_inner_train_idx = [ii for ii in inner_train_idx if wsi_mask[outer_train_idx[ii]]]
            wsi_inner_val_idx   = [ii for ii in inner_val_idx   if wsi_mask[outer_train_idx[ii]]]
            if len(wsi_inner_val_idx) > 0:
                X_inner_train_wsi_orig = wsi_embeddings_array[[outer_train_idx[ii] for ii in wsi_inner_train_idx]]
                y_inner_train_evt_wsi_orig = y_outer_train_events[wsi_inner_train_idx]
                y_inner_train_dur_wsi_orig = y_outer_train_durations[wsi_inner_train_idx]
                X_inner_val_wsi_orig = wsi_embeddings_array[[outer_train_idx[ii] for ii in wsi_inner_val_idx]]
                y_inner_val_evt_wsi_orig = y_outer_train_events[wsi_inner_val_idx]
                y_inner_val_dur_wsi_orig = y_outer_train_durations[wsi_inner_val_idx]
                if X_inner_train_wsi_orig.shape[0] > 0 and X_inner_val_wsi_orig.shape[0] > 0:
                    net_inner_wsi_orig = tt.practical.MLPVanilla(
                        in_features=X_inner_train_wsi_orig.shape[1],
                        num_nodes=num_nodes,
                        out_features=out_features,
                        batch_norm=batch_norm,
                        dropout=dropout,
                        output_bias=output_bias
                    )
                    optimizer_inner_wsi_orig = tt.optim.Adam(lr=1e-3)
                    cox_ph_inner_wsi_orig = CoxPH(net_inner_wsi_orig, optimizer_inner_wsi_orig)
                    cox_ph_inner_wsi_orig.fit(
                        X_inner_train_wsi_orig,
                        (y_inner_train_evt_wsi_orig, y_inner_train_dur_wsi_orig),
                        batch_size=64,
                        epochs=100,
                        verbose=False
                    )
                    risk_val_inner_wsi_orig = cox_ph_inner_wsi_orig.predict(X_inner_val_wsi_orig).squeeze()
                    c_index_inner_wsi_orig = concordance_index_censored(
                        y_inner_val_evt_wsi_orig.astype(bool),
                        y_inner_val_dur_wsi_orig,
                        -risk_val_inner_wsi_orig
                    )[0]
                    print(f"    Inner Fold {inner_fold}, Outer Fold {outer_fold} Original WSI C-Index: {c_index_inner_wsi_orig:.4f}")

            # Determine threshold
            from lifelines.statistics import logrank_test
            unique_risk_scores_val = np.sort(np.unique(risk_val_inner))
            total_patients_val = len(risk_val_inner)
            min_group_size_val = total_patients_val * 0.25
            best_threshold_val = None
            min_p_value_val = 1.0
            events_val_bool = y_inner_val_evt.astype(bool)
            durations_val = y_inner_val_dur

            for threshold in unique_risk_scores_val[1:-1]:
                is_high_risk = (risk_val_inner <= threshold)
                is_low_risk  = (risk_val_inner >  threshold)
                group_high_size = is_high_risk.sum()
                group_low_size  = is_low_risk.sum()
                if group_high_size < min_group_size_val or group_low_size < min_group_size_val:
                    continue
                if events_val_bool[is_high_risk].sum() == 0 or events_val_bool[is_low_risk].sum() == 0:
                    continue
                results_val = logrank_test(
                    durations_val[is_high_risk],
                    durations_val[is_low_risk],
                    event_observed_A=events_val_bool[is_high_risk],
                    event_observed_B=events_val_bool[is_low_risk]
                )
                p_val = results_val.p_value
                if p_val < min_p_value_val:
                    min_p_value_val = p_val
                    best_threshold_val = threshold

            if best_threshold_val is None:
                classification_val = np.full(len(risk_val_inner), -1)
            else:
                is_high_risk = (risk_val_inner <= best_threshold_val)
                classification_val = np.where(is_high_risk, 1, 0)

            col_name = f"{selected_name}_inner_loop_{inner_fold}_outer_loop_{outer_fold}"
            inner_val_barcodes = patient_barcodes_outer_train.iloc[inner_val_idx].reset_index(drop=True)
            tmp_df_ = pd.DataFrame({
                'patient.bcr_patient_barcode': inner_val_barcodes,
                col_name: classification_val.astype(int)
            })
            if col_name in updated_split_df.columns:
                updated_split_df = updated_split_df.drop(columns=[col_name])
            updated_split_df = updated_split_df.merge(tmp_df_, on='patient.bcr_patient_barcode', how='left')
            updated_split_df[col_name] = updated_split_df[col_name].fillna(-1).astype(int)

            updated_split_entries_2 = updated_split_df.drop(
                columns=['patient.bcr_patient_barcode'], errors='ignore'
            ).to_dict(orient='records')
            with open(split_json_path, 'w') as fsave_:
                json.dump(updated_split_entries_2, fsave_, indent=4)

            # Run Contrastive + Refine
            run_contrastive_script(
                json_file_path=split_json_path,
                embeddings_folder=embeddings_dir,
                label_type=col_name
            )
            refine_embeddings(
                name=col_name,
                embeddings_folder=embeddings_dir
            )

            # Load refined embeddings
            embeddings_dir_refined = os.path.join(".", col_name)
            missing_files_refined = []
            def load_embedding_parallel_refined(file_name_r):
                embedding_file_r = file_name_r.replace('.svs','_flatten.pt')
                embedding_path_r = os.path.join(embeddings_dir_refined, embedding_file_r)
                if not os.path.exists(embedding_path_r):
                    missing_files_refined.append(embedding_file_r)
                    return None
                try:
                    embedding_tensor_r = torch.load(embedding_path_r)
                    embedding_np_r = embedding_tensor_r.numpy().astype(np.float32).flatten()
                    return embedding_np_r
                except Exception:
                    missing_files_refined.append(embedding_file_r)
                    return None

            def load_embeddings_in_parallel_refined(local_wsi_df_r):
                embeddings_ref_ = []
                with ThreadPoolExecutor(max_workers=8) as executor_r_:
                    futures_r_ = {
                        executor_r_.submit(load_embedding_parallel_refined, row['file_name']): idxr_
                        for idxr_, row in local_wsi_df_r.iterrows()
                    }
                    for future_r_ in tqdm(as_completed(futures_r_), total=len(futures_r_)):
                        embeddings_ref_.append(future_r_.result())
                return embeddings_ref_

            wsi_df_refined = wsi_df.copy()
            wsi_df_refined['embedding'] = load_embeddings_in_parallel_refined(wsi_df_refined)
            wsi_df_refined = wsi_df_refined[wsi_df_refined['embedding'].notnull()].reset_index(drop=True)

            barcode_to_wsi_refined = dict(zip(wsi_df_refined['patient.bcr_patient_barcode'], wsi_df_refined['embedding']))
            wsi_embeddings_list_refined = []
            wsi_mask_refined = []
            for pbb_ in patient_barcodes_full:
                if pbb_ in barcode_to_wsi_refined and barcode_to_wsi_refined[pbb_] is not None:
                    wsi_embeddings_list_refined.append(barcode_to_wsi_refined[pbb_])
                    wsi_mask_refined.append(True)
                else:
                    wsi_mask_refined.append(False)

            wsi_embeddings_array_refined = np.array(wsi_embeddings_list_refined, dtype=np.float32)
            wsi_mask_refined = np.array(wsi_mask_refined)

            wsi_inner_train_idx_ref = [ii for ii in inner_train_idx if wsi_mask_refined[outer_train_idx[ii]]]
            wsi_inner_val_idx_ref   = [ii for ii in inner_val_idx   if wsi_mask_refined[outer_train_idx[ii]]]
            if len(wsi_inner_val_idx_ref) > 0:
                X_inner_train_wsi_ref = wsi_embeddings_array_refined[[outer_train_idx[ii] for ii in wsi_inner_train_idx_ref]]
                y_inner_train_evt_wsi_ref = y_outer_train_events[wsi_inner_train_idx_ref]
                y_inner_train_dur_wsi_ref = y_outer_train_durations[wsi_inner_train_idx_ref]

                X_inner_val_wsi_ref = wsi_embeddings_array_refined[[outer_train_idx[ii] for ii in wsi_inner_val_idx_ref]]
                y_inner_val_evt_wsi_ref = y_outer_train_events[wsi_inner_val_idx_ref]
                y_inner_val_dur_wsi_ref = y_outer_train_durations[wsi_inner_val_idx_ref]
                if X_inner_train_wsi_ref.shape[0] > 0 and X_inner_val_wsi_ref.shape[0] > 0:
                    net_inner_wsi_ref = tt.practical.MLPVanilla(
                        in_features=X_inner_train_wsi_ref.shape[1],
                        num_nodes=num_nodes,
                        out_features=out_features,
                        batch_norm=batch_norm,
                        dropout=dropout,
                        output_bias=output_bias
                    )
                    optimizer_inner_wsi_ref = tt.optim.Adam(lr=1e-3)
                    cox_ph_inner_wsi_ref = CoxPH(net_inner_wsi_ref, optimizer_inner_wsi_ref)
                    cox_ph_inner_wsi_ref.fit(
                        X_inner_train_wsi_ref,
                        (y_inner_train_evt_wsi_ref, y_inner_train_dur_wsi_ref),
                        batch_size=64,
                        epochs=100,
                        verbose=False
                    )
                    risk_val_inner_wsi_ref = cox_ph_inner_wsi_ref.predict(X_inner_val_wsi_ref).squeeze()
                    c_index_inner_wsi_ref = concordance_index_censored(
                        y_inner_val_evt_wsi_ref.astype(bool),
                        y_inner_val_dur_wsi_ref,
                        -risk_val_inner_wsi_ref
                    )[0]
                    print(f"    Inner Fold {inner_fold}, Outer Fold {outer_fold} Refined WSI C-Index: {c_index_inner_wsi_ref:.4f}")

        net_outer = tt.practical.MLPVanilla(
            in_features=X_outer_train.shape[1],
            num_nodes=num_nodes,
            out_features=out_features,
            batch_norm=batch_norm,
            dropout=dropout,
            output_bias=output_bias
        )
        optimizer_outer = tt.optim.Adam(lr=1e-3)
        cox_ph_outer = CoxPH(net_outer, optimizer_outer)
        cox_ph_outer.fit(
            X_outer_train,
            (y_outer_train_events, y_outer_train_durations),
            batch_size=64,
            epochs=100,
            verbose=False
        )
        risk_val_outer = cox_ph_outer.predict(X_outer_val).squeeze()
        c_index_outer = concordance_index_censored(
            y_outer_val_events.astype(bool),
            y_outer_val_durations,
            -risk_val_outer
        )[0]
        print(f"Outer Fold {outer_fold} Test C-Index (Clinical Only): {c_index_outer:.4f}\n")
        outer_test_c_indexes.append(c_index_outer)

        # Threshold for outer validation
        unique_risk_scores_outer_val = np.sort(np.unique(risk_val_outer))
        total_patients_outer_val = len(risk_val_outer)
        min_group_size_outer_val = total_patients_outer_val * 0.25
        best_threshold_outer_val = None
        min_p_value_outer_val = 1.0
        events_outer_val_bool = y_outer_val_events.astype(bool)
        durations_outer_val = y_outer_val_durations

        for threshold_ in unique_risk_scores_outer_val[1:-1]:
            is_high_risk_ = (risk_val_outer <= threshold_)
            is_low_risk_  = (risk_val_outer >  threshold_)
            group_high_size_ = is_high_risk_.sum()
            group_low_size_  = is_low_risk_.sum()
            if group_high_size_ < min_group_size_outer_val or group_low_size_ < min_group_size_outer_val:
                continue
            if events_outer_val_bool[is_high_risk_].sum() == 0 or events_outer_val_bool[is_low_risk_].sum() == 0:
                continue
            results_outer_val_ = logrank_test(
                durations_outer_val[is_high_risk_],
                durations_outer_val[is_low_risk_],
                event_observed_A=events_outer_val_bool[is_high_risk_],
                event_observed_B=events_outer_val_bool[is_low_risk_]
            )
            p_val_outer_ = results_outer_val_.p_value
            if p_val_outer_ < min_p_value_outer_val:
                min_p_value_outer_val = p_val_outer_
                best_threshold_outer_val = threshold_

        if best_threshold_outer_val is None:
            classification_outer_val = np.full(len(risk_val_outer), -1)
        else:
            is_high_risk_ = (risk_val_outer <= best_threshold_outer_val)
            classification_outer_val = np.where(is_high_risk_, 1, 0)

        col_name_outer_val = f"{selected_name}_outer_loop_{outer_fold}_final_strat"
        tmp_outer_val_df = pd.DataFrame({
            'patient.bcr_patient_barcode': patient_barcodes_outer_val,
            col_name_outer_val: classification_outer_val.astype(int)
        })
        if col_name_outer_val in updated_split_df.columns:
            updated_split_df = updated_split_df.drop(columns=[col_name_outer_val])
        updated_split_df = updated_split_df.merge(tmp_outer_val_df, on='patient.bcr_patient_barcode', how='left')
        updated_split_df[col_name_outer_val] = updated_split_df[col_name_outer_val].fillna(-1).astype(int)

        updated_split_entries_3 = updated_split_df.drop(
            columns=['patient.bcr_patient_barcode'], errors='ignore'
        ).to_dict(orient='records')
        with open(split_json_path, 'w') as fsave_2:
            json.dump(updated_split_entries_3, fsave_2, indent=4)

        # Evaluate Original WSI on outer fold
        wsi_outer_val_mask = [ii for ii in outer_val_idx if wsi_mask[ii]]
        if len(wsi_outer_val_mask) > 0:
            wsi_outer_train_positions = [pos for pos, idx_ in enumerate(outer_train_idx) if wsi_mask[idx_]]
            wsi_outer_val_positions   = [pos for pos, idx_ in enumerate(outer_val_idx)   if wsi_mask[idx_]]
            X_outer_train_wsi_orig = wsi_embeddings_array[wsi_outer_train_positions]
            y_outer_train_evt_wsi_orig = y_outer_train_events[wsi_outer_train_positions]
            y_outer_train_dur_wsi_orig = y_outer_train_durations[wsi_outer_train_positions]

            X_outer_val_wsi_orig = wsi_embeddings_array[wsi_outer_val_positions]
            y_outer_val_evt_wsi_orig = y_outer_val_events[wsi_outer_val_positions]
            y_outer_val_dur_wsi_orig = y_outer_val_durations[wsi_outer_val_positions]
            if X_outer_train_wsi_orig.shape[0] > 0 and X_outer_val_wsi_orig.shape[0] > 0:
                net_outer_wsi_orig = tt.practical.MLPVanilla(
                    in_features=X_outer_train_wsi_orig.shape[1],
                    num_nodes=num_nodes,
                    out_features=out_features,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    output_bias=output_bias
                )
                optimizer_outer_wsi_orig = tt.optim.Adam(lr=1e-3)
                cox_ph_outer_wsi_orig = CoxPH(net_outer_wsi_orig, optimizer_outer_wsi_orig)
                cox_ph_outer_wsi_orig.fit(
                    X_outer_train_wsi_orig,
                    (y_outer_train_evt_wsi_orig, y_outer_train_dur_wsi_orig),
                    batch_size=64,
                    epochs=100,
                    verbose=False
                )
                risk_val_outer_wsi_orig = cox_ph_outer_wsi_orig.predict(X_outer_val_wsi_orig).squeeze()
                c_index_outer_wsi_orig = concordance_index_censored(
                    y_outer_val_evt_wsi_orig.astype(bool),
                    y_outer_val_dur_wsi_orig,
                    -risk_val_outer_wsi_orig
                )[0]
                print(f"Outer Fold {outer_fold} Test WSI C-Index (Original): {c_index_outer_wsi_orig:.4f}")
                outer_test_c_indexes_wsi_original.append(c_index_outer_wsi_orig)

        # Refine
        run_contrastive_script(
            json_file_path=split_json_path,
            embeddings_folder=embeddings_dir,
            label_type=col_name_outer_val
        )
        refine_embeddings(
            name=col_name_outer_val,
            embeddings_folder=embeddings_dir
        )

        embeddings_dir_refined_outer = os.path.join(".", col_name_outer_val)
        missing_files_refined_outer = []
        def load_embedding_parallel_refined_outer(file_name_ro):
            embedding_file_ro = file_name_ro.replace('.svs','_flatten.pt')
            embedding_path_ro = os.path.join(embeddings_dir_refined_outer, embedding_file_ro)
            if not os.path.exists(embedding_path_ro):
                missing_files_refined_outer.append(embedding_file_ro)
                return None
            try:
                embedding_tensor_ro = torch.load(embedding_path_ro)
                embedding_np_ro = embedding_tensor_ro.numpy().astype(np.float32).flatten()
                return embedding_np_ro
            except Exception:
                missing_files_refined_outer.append(embedding_file_ro)
                return None

        def load_embeddings_in_parallel_refined_outer(local_wsi_df_ro):
            embeddings_ref_ro_ = []
            with ThreadPoolExecutor(max_workers=8) as executor_ro_:
                futures_ro_ = {
                    executor_ro_.submit(load_embedding_parallel_refined_outer, row['file_name']): idx_ro_
                    for idx_ro_, row in local_wsi_df_ro.iterrows()
                }
                for future_ro_ in tqdm(as_completed(futures_ro_), total=len(futures_ro_)):
                    embeddings_ref_ro_.append(future_ro_.result())
            return embeddings_ref_ro_

        wsi_df_refined_outer = wsi_df.copy()
        wsi_df_refined_outer['embedding'] = load_embeddings_in_parallel_refined_outer(wsi_df_refined_outer)
        wsi_df_refined_outer = wsi_df_refined_outer[wsi_df_refined_outer['embedding'].notnull()].reset_index(drop=True)

        barcode_to_wsi_refined_outer = dict(
            zip(wsi_df_refined_outer['patient.bcr_patient_barcode'], wsi_df_refined_outer['embedding'])
        )
        wsi_embeddings_list_refined_outer = []
        wsi_mask_refined_outer = []
        for pb_out_ in patient_barcodes_full:
            if pb_out_ in barcode_to_wsi_refined_outer and barcode_to_wsi_refined_outer[pb_out_] is not None:
                wsi_embeddings_list_refined_outer.append(barcode_to_wsi_refined_outer[pb_out_])
                wsi_mask_refined_outer.append(True)
            else:
                wsi_mask_refined_outer.append(False)

        wsi_embeddings_array_refined_outer = np.array(wsi_embeddings_list_refined_outer, dtype=np.float32)
        wsi_mask_refined_outer = np.array(wsi_mask_refined_outer)

        wsi_outer_val_mask_ref = [ii for ii in outer_val_idx if wsi_mask_refined_outer[ii]]
        if len(wsi_outer_val_mask_ref) > 0:
            wsi_outer_train_positions_ref = [
                pos for pos, idx_2 in enumerate(outer_train_idx) if wsi_mask_refined_outer[idx_2]
            ]
            wsi_outer_val_positions_ref = [
                pos for pos, idx_2 in enumerate(outer_val_idx)   if wsi_mask_refined_outer[idx_2]
            ]
            X_outer_train_wsi_ref = wsi_embeddings_array_refined_outer[wsi_outer_train_positions_ref]
            y_outer_train_evt_wsi_ref = y_outer_train_events[wsi_outer_train_positions_ref]
            y_outer_train_dur_wsi_ref = y_outer_train_durations[wsi_outer_train_positions_ref]

            X_outer_val_wsi_ref = wsi_embeddings_array_refined_outer[wsi_outer_val_positions_ref]
            y_outer_val_evt_wsi_ref = y_outer_val_events[wsi_outer_val_positions_ref]
            y_outer_val_dur_wsi_ref = y_outer_val_durations[wsi_outer_val_positions_ref]
            if X_outer_train_wsi_ref.shape[0] > 0 and X_outer_val_wsi_ref.shape[0] > 0:
                net_outer_wsi_ref = tt.practical.MLPVanilla(
                    in_features=X_outer_train_wsi_ref.shape[1],
                    num_nodes=num_nodes,
                    out_features=out_features,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    output_bias=output_bias
                )
                optimizer_outer_wsi_ref = tt.optim.Adam(lr=1e-3)
                cox_ph_outer_wsi_ref = CoxPH(net_outer_wsi_ref, optimizer_outer_wsi_ref)
                cox_ph_outer_wsi_ref.fit(
                    X_outer_train_wsi_ref,
                    (y_outer_train_evt_wsi_ref, y_outer_train_dur_wsi_ref),
                    batch_size=64,
                    epochs=100,
                    verbose=False
                )
                risk_val_outer_wsi_ref = cox_ph_outer_wsi_ref.predict(X_outer_val_wsi_ref).squeeze()
                c_index_outer_wsi_ref = concordance_index_censored(
                    y_outer_val_evt_wsi_ref.astype(bool),
                    y_outer_val_dur_wsi_ref,
                    -risk_val_outer_wsi_ref
                )[0]
                print(f"Outer Fold {outer_fold} Test WSI C-Index (Refined): {c_index_outer_wsi_ref:.4f}\n")
                outer_test_c_indexes_wsi_refined.append(c_index_outer_wsi_ref)

    c_index_test_mean = np.mean(outer_test_c_indexes)
    c_index_test_mean_wsi_orig = np.mean(outer_test_c_indexes_wsi_original) if len(outer_test_c_indexes_wsi_original)>0 else np.nan
    c_index_test_mean_wsi_ref  = np.mean(outer_test_c_indexes_wsi_refined)  if len(outer_test_c_indexes_wsi_refined)>0  else np.nan

    net_final = tt.practical.MLPVanilla(
        in_features=train_embeddings_full.shape[1],
        num_nodes=num_nodes,
        out_features=out_features,
        batch_norm=batch_norm,
        dropout=dropout,
        output_bias=output_bias
    )
    optimizer_final = tt.optim.Adam(lr=1e-3)
    cox_ph_final = CoxPH(net_final, optimizer_final)
    cox_ph_final.fit(
        train_embeddings_full,
        (events_train_full, durations_train_full),
        batch_size=64,
        epochs=100,
        verbose=False
    )

    risk_train_final = cox_ph_final.predict(train_embeddings_full).squeeze()
    c_index_train = concordance_index_censored(
        events_train_full.astype(bool),
        durations_train_full,
        -risk_train_final
    )[0]

    print("\n====== Nested CV Results ======")
    print(f"Training C-Index (final model on all training): {c_index_train:.4f}")
    print(f"Test C-Index (Clinical Only, Nested CV): {c_index_test_mean:.4f}")
    if not np.isnan(c_index_test_mean_wsi_orig):
        print(f"Test WSI C-Index (Original, Nested CV): {c_index_test_mean_wsi_orig:.4f}")
    else:
        print("No Original WSI test C-index was computed (some folds may have no WSI data).")
    if not np.isnan(c_index_test_mean_wsi_ref):
        print(f"Test WSI C-Index (Refined, Nested CV): {c_index_test_mean_wsi_ref:.4f}")
    else:
        print("No Refined WSI test C-index was computed (some folds may have no WSI data).")

    for col_inner in updated_split_df.columns:
        if col_inner.startswith(selected_name + '_inner_loop_') or col_inner.startswith(selected_name + '_outer_loop_'):
            updated_split_df[col_inner].fillna(-1, inplace=True)
            updated_split_df[col_inner] = updated_split_df[col_inner].astype(int)

    updated_split_entries_fin = updated_split_df.drop(
        columns=['patient.bcr_patient_barcode'], errors='ignore'
    ).to_dict(orient='records')
    with open(split_json_path, 'w') as f__:
        json.dump(updated_split_entries_fin, f__, indent=4)

    # ----------------------------------------------------------------
    #          NEW: BIASED CROSS-VALIDATION (SINGLE LEVEL)
    # ----------------------------------------------------------------
    if run_biased_cv:
        print("\n===== RUNNING BIASED CROSS-VALIDATION (SINGLE-LEVEL) =====\n")
        from sklearn.model_selection import KFold
        biased_outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)

        biased_outer_test_c_indexes = []
        biased_outer_test_c_indexes_wsi_orig = []
        biased_outer_test_c_indexes_wsi_refined = []
        updated_split_df_biased = updated_split_df.copy()

        for fold_i, (train_idx_b, test_idx_b) in enumerate(biased_outer_kf.split(train_embeddings_full), start=1):
            print(f"Biased CV Fold {fold_i}/5")
            X_train_b = train_embeddings_full[train_idx_b]
            y_train_evt_b = events_train_full[train_idx_b]
            y_train_dur_b = durations_train_full[train_idx_b]

            X_test_b = train_embeddings_full[test_idx_b]
            y_test_evt_b = events_train_full[test_idx_b]
            y_test_dur_b = durations_train_full[test_idx_b]

            net_biased = tt.practical.MLPVanilla(
                in_features=X_train_b.shape[1],
                num_nodes=num_nodes,
                out_features=out_features,
                batch_norm=batch_norm,
                dropout=dropout,
                output_bias=output_bias
            )
            optimizer_biased = tt.optim.Adam(lr=1e-3)
            cox_ph_biased = CoxPH(net_biased, optimizer_biased)
            cox_ph_biased.fit(
                X_train_b,
                (y_train_evt_b, y_train_dur_b),
                batch_size=64,
                epochs=100,
                verbose=False
            )

            risk_val_test_b = cox_ph_biased.predict(X_test_b).squeeze()
            c_index_biased = concordance_index_censored(
                y_test_evt_b.astype(bool),
                y_test_dur_b,
                -risk_val_test_b
            )[0]
            print(f"  Biased Fold {fold_i} C-Index (Clinical Only): {c_index_biased:.4f}")
            biased_outer_test_c_indexes.append(c_index_biased)

            # Threshold on the same test fold => very biased
            unique_risk_scores_test_b = np.sort(np.unique(risk_val_test_b))
            total_patients_test_b = len(risk_val_test_b)
            min_group_size_test_b = total_patients_test_b * 0.25
            best_threshold_test_b = None
            min_p_value_test_b = 1.0
            events_test_b_bool = y_test_evt_b.astype(bool)
            durations_test_b = y_test_dur_b

            for threshold_b in unique_risk_scores_test_b[1:-1]:
                is_high_risk_b = (risk_val_test_b <= threshold_b)
                is_low_risk_b  = (risk_val_test_b >  threshold_b)
                group_high_size_b = is_high_risk_b.sum()
                group_low_size_b  = is_low_risk_b.sum()
                if group_high_size_b < min_group_size_test_b or group_low_size_b < min_group_size_test_b:
                    continue
                if events_test_b_bool[is_high_risk_b].sum() == 0 or events_test_b_bool[is_low_risk_b].sum() == 0:
                    continue
                results_test_b = logrank_test(
                    durations_test_b[is_high_risk_b],
                    durations_test_b[is_low_risk_b],
                    event_observed_A=events_test_b_bool[is_high_risk_b],
                    event_observed_B=events_test_b_bool[is_low_risk_b]
                )
                p_val_b = results_test_b.p_value
                if p_val_b < min_p_value_test_b:
                    min_p_value_test_b = p_val_b
                    best_threshold_test_b = threshold_b

            if best_threshold_test_b is None:
                classification_test_b = np.full(len(risk_val_test_b), -1)
            else:
                is_high_risk_b = (risk_val_test_b <= best_threshold_test_b)
                classification_test_b = np.where(is_high_risk_b, 1, 0)

            col_name_biased = f"{selected_name}_biased_fold_{fold_i}"
            barcodes_test_b = patient_barcodes_full.iloc[test_idx_b].reset_index(drop=True)

            tmp_biased_df_ = pd.DataFrame({
                'patient.bcr_patient_barcode': barcodes_test_b,
                col_name_biased: classification_test_b.astype(int)
            })
            if col_name_biased in updated_split_df_biased.columns:
                updated_split_df_biased = updated_split_df_biased.drop(columns=[col_name_biased])
            updated_split_df_biased = updated_split_df_biased.merge(tmp_biased_df_, on='patient.bcr_patient_barcode', how='left')
            updated_split_df_biased[col_name_biased] = updated_split_df_biased[col_name_biased].fillna(-1).astype(int)

            # Evaluate Original WSI in Biased CV
            wsi_test_mask_b = [ix for ix in test_idx_b if wsi_mask[ix]]
            if len(wsi_test_mask_b) > 0:
                wsi_train_positions_b = [pos for pos, idx_ in enumerate(train_idx_b) if wsi_mask[idx_]]
                wsi_test_positions_b  = [pos for pos, idx_ in enumerate(test_idx_b)  if wsi_mask[idx_]]

                X_train_wsi_orig_b = wsi_embeddings_array[wsi_train_positions_b]
                y_train_evt_wsi_orig_b = events_train_full[train_idx_b[wsi_train_positions_b]]
                y_train_dur_wsi_orig_b = durations_train_full[train_idx_b[wsi_train_positions_b]]

                X_test_wsi_orig_b = wsi_embeddings_array[wsi_test_positions_b]
                y_test_evt_wsi_orig_b = events_train_full[test_idx_b[wsi_test_positions_b]]
                y_test_dur_wsi_orig_b = durations_train_full[test_idx_b[wsi_test_positions_b]]

                if X_train_wsi_orig_b.shape[0] > 0 and X_test_wsi_orig_b.shape[0] > 0:
                    net_biased_wsi_orig = tt.practical.MLPVanilla(
                        in_features=X_train_wsi_orig_b.shape[1],
                        num_nodes=num_nodes,
                        out_features=out_features,
                        batch_norm=batch_norm,
                        dropout=dropout,
                        output_bias=output_bias
                    )
                    optimizer_biased_wsi_orig = tt.optim.Adam(lr=1e-3)
                    cox_ph_biased_wsi_orig = CoxPH(net_biased_wsi_orig, optimizer_biased_wsi_orig)
                    cox_ph_biased_wsi_orig.fit(
                        X_train_wsi_orig_b,
                        (y_train_evt_wsi_orig_b, y_train_dur_wsi_orig_b),
                        batch_size=64,
                        epochs=100,
                        verbose=False
                    )
                    risk_val_test_wsi_orig_b = cox_ph_biased_wsi_orig.predict(X_test_wsi_orig_b).squeeze()
                    c_index_biased_wsi_orig = concordance_index_censored(
                        y_test_evt_wsi_orig_b.astype(bool),
                        y_test_dur_wsi_orig_b,
                        -risk_val_test_wsi_orig_b
                    )[0]
                    print(f"    Biased Fold {fold_i} WSI C-Index (Original): {c_index_biased_wsi_orig:.4f}")
                    biased_outer_test_c_indexes_wsi_orig.append(c_index_biased_wsi_orig)

            # Contrastive + Refine for Biased fold
            with open(split_json_path, 'r') as f_bi:
                current_entries_ = json.load(f_bi)
            tmp_split_df_bi = pd.DataFrame(current_entries_)
            if 'patient.bcr_patient_barcode' not in tmp_split_df_bi.columns:
                tmp_split_df_bi['patient.bcr_patient_barcode'] = tmp_split_df_bi['file_name'].apply(extract_barcode)

            if col_name_biased in tmp_split_df_bi.columns:
                tmp_split_df_bi = tmp_split_df_bi.drop(columns=[col_name_biased])
            tmp_biased_df_2 = pd.DataFrame({
                'patient.bcr_patient_barcode': barcodes_test_b,
                col_name_biased: classification_test_b.astype(int)
            })
            tmp_split_df_bi = tmp_split_df_bi.merge(tmp_biased_df_2, on='patient.bcr_patient_barcode', how='left')
            tmp_split_df_bi[col_name_biased] = tmp_split_df_bi[col_name_biased].fillna(-1).astype(int)

            updated_split_entries_biased = tmp_split_df_bi.drop(
                columns=['patient.bcr_patient_barcode'], errors='ignore'
            ).to_dict(orient='records')
            with open(split_json_path, 'w') as f_bi2:
                json.dump(updated_split_entries_biased, f_bi2, indent=4)

            run_contrastive_script(
                json_file_path=split_json_path,
                embeddings_folder=embeddings_dir,
                label_type=col_name_biased
            )
            refine_embeddings(
                name=col_name_biased,
                embeddings_folder=embeddings_dir
            )

            # Re-load refined WSI embeddings
            embeddings_dir_refined_biased = os.path.join(".", col_name_biased)
            missing_files_refined_biased = []
            def load_embedding_parallel_refined_biased(file_name_r2):
                embedding_file_r2 = file_name_r2.replace('.svs','_flatten.pt')
                embedding_path_r2 = os.path.join(embeddings_dir_refined_biased, embedding_file_r2)
                if not os.path.exists(embedding_path_r2):
                    missing_files_refined_biased.append(embedding_file_r2)
                    return None
                try:
                    embedding_tensor_r2 = torch.load(embedding_path_r2)
                    embedding_np_r2 = embedding_tensor_r2.numpy().astype(np.float32).flatten()
                    return embedding_np_r2
                except Exception:
                    missing_files_refined_biased.append(embedding_file_r2)
                    return None

            def load_embeddings_in_parallel_refined_biased(local_wsi_df_r2):
                embeddings_ref_b2_ = []
                with ThreadPoolExecutor(max_workers=8) as executor_r2_:
                    futures_r2_ = {
                        executor_r2_.submit(load_embedding_parallel_refined_biased, row['file_name']): idxr2_
                        for idxr2_, row in local_wsi_df_r2.iterrows()
                    }
                    for future_r2_ in tqdm(as_completed(futures_r2_), total=len(futures_r2_)):
                        embeddings_ref_b2_.append(future_r2_.result())
                return embeddings_ref_b2_

            wsi_df_refined_biased = wsi_df.copy()
            wsi_df_refined_biased['embedding'] = load_embeddings_in_parallel_refined_biased(wsi_df_refined_biased)
            wsi_df_refined_biased = wsi_df_refined_biased[wsi_df_refined_biased['embedding'].notnull()].reset_index(drop=True)

            barcode_to_wsi_refined_biased = dict(zip(
                wsi_df_refined_biased['patient.bcr_patient_barcode'],
                wsi_df_refined_biased['embedding']
            ))
            wsi_embeddings_list_refined_biased = []
            wsi_mask_refined_biased = []
            for pbb2_ in patient_barcodes_full:
                if pbb2_ in barcode_to_wsi_refined_biased and barcode_to_wsi_refined_biased[pbb2_] is not None:
                    wsi_embeddings_list_refined_biased.append(barcode_to_wsi_refined_biased[pbb2_])
                    wsi_mask_refined_biased.append(True)
                else:
                    wsi_mask_refined_biased.append(False)

            wsi_embeddings_array_refined_biased = np.array(wsi_embeddings_list_refined_biased, dtype=np.float32)
            wsi_mask_refined_biased = np.array(wsi_mask_refined_biased)

            wsi_test_mask_ref_biased = [jj for jj in test_idx_b if wsi_mask_refined_biased[jj]]
            if len(wsi_test_mask_ref_biased) > 0:
                wsi_train_positions_ref_biased = [pos_ for pos_, idx_ in enumerate(train_idx_b) if wsi_mask_refined_biased[idx_]]
                wsi_test_positions_ref_biased  = [pos_ for pos_, idx_ in enumerate(test_idx_b)  if wsi_mask_refined_biased[idx_]]

                X_train_wsi_ref_biased = wsi_embeddings_array_refined_biased[wsi_train_positions_ref_biased]
                y_train_evt_wsi_ref_biased = events_train_full[train_idx_b[wsi_train_positions_ref_biased]]
                y_train_dur_wsi_ref_biased = durations_train_full[train_idx_b[wsi_train_positions_ref_biased]]

                X_test_wsi_ref_biased = wsi_embeddings_array_refined_biased[wsi_test_positions_ref_biased]
                y_test_evt_wsi_ref_biased = events_train_full[test_idx_b[wsi_test_positions_ref_biased]]
                y_test_dur_wsi_ref_biased = durations_train_full[test_idx_b[wsi_test_positions_ref_biased]]

                if X_train_wsi_ref_biased.shape[0]>0 and X_test_wsi_ref_biased.shape[0]>0:
                    net_biased_wsi_ref = tt.practical.MLPVanilla(
                        in_features=X_train_wsi_ref_biased.shape[1],
                        num_nodes=num_nodes,
                        out_features=out_features,
                        batch_norm=batch_norm,
                        dropout=dropout,
                        output_bias=output_bias
                    )
                    optimizer_biased_wsi_ref = tt.optim.Adam(lr=1e-3)
                    cox_ph_biased_wsi_ref = CoxPH(net_biased_wsi_ref, optimizer_biased_wsi_ref)
                    cox_ph_biased_wsi_ref.fit(
                        X_train_wsi_ref_biased,
                        (y_train_evt_wsi_ref_biased, y_train_dur_wsi_ref_biased),
                        batch_size=64,
                        epochs=100,
                        verbose=False
                    )
                    risk_val_test_wsi_ref_biased = cox_ph_biased_wsi_ref.predict(X_test_wsi_ref_biased).squeeze()
                    c_index_biased_wsi_ref = concordance_index_censored(
                        y_test_evt_wsi_ref_biased.astype(bool),
                        y_test_dur_wsi_ref_biased,
                        -risk_val_test_wsi_ref_biased
                    )[0]
                    print(f"    Biased Fold {fold_i} WSI C-Index (Refined): {c_index_biased_wsi_ref:.4f}")
                    biased_outer_test_c_indexes_wsi_refined.append(c_index_biased_wsi_ref)

        c_index_biased_mean = np.mean(biased_outer_test_c_indexes)
        c_index_biased_mean_wsi_orig = np.mean(biased_outer_test_c_indexes_wsi_orig) if len(biased_outer_test_c_indexes_wsi_orig)>0 else np.nan
        c_index_biased_mean_wsi_ref  = np.mean(biased_outer_test_c_indexes_wsi_refined) if len(biased_outer_test_c_indexes_wsi_refined)>0 else np.nan


        print("\n====== Nested CV Results ======")
        print(f"Training C-Index (final model on all training): {c_index_train:.4f}")
        print(f"Test C-Index (Clinical Only, Nested CV): {c_index_test_mean:.4f}")
        if not np.isnan(c_index_test_mean_wsi_orig):
            print(f"Test WSI C-Index (Original, Nested CV): {c_index_test_mean_wsi_orig:.4f}")
        else:
            print("No Original WSI test C-index was computed (some folds may have no WSI data).")
        if not np.isnan(c_index_test_mean_wsi_ref):
            print(f"Test WSI C-Index (Refined, Nested CV): {c_index_test_mean_wsi_ref:.4f}")
        else:
            print("No Refined WSI test C-index was computed (some folds may have no WSI data).")


        print("\n====== Biased CV Results ======")
        print(f"Biased CV Mean Test C-Index (Clinical Only): {c_index_biased_mean:.4f}")
        if not np.isnan(c_index_biased_mean_wsi_orig):
            print(f"Biased CV Mean Test WSI C-Index (Original): {c_index_biased_mean_wsi_orig:.4f}")
        else:
            print("No Original WSI test C-index in biased CV (some folds may have no WSI data).")

        if not np.isnan(c_index_biased_mean_wsi_ref):
            print(f"Biased CV Mean Test WSI C-Index (Refined): {c_index_biased_mean_wsi_ref:.4f}")
        else:
            print("No Refined WSI test C-index in biased CV (some folds may have no WSI data).")

        # Optionally save updated_split_df_biased
        for col_b in updated_split_df_biased.columns:
            if col_b.startswith(selected_name + '_biased_fold_'):
                updated_split_df_biased[col_b].fillna(-1, inplace=True)
                updated_split_df_biased[col_b] = updated_split_df_biased[col_b].astype(int)

        updated_split_entries_biased_ = updated_split_df_biased.drop(
            columns=['patient.bcr_patient_barcode'], errors='ignore'
        ).to_dict(orient='records')
        with open(split_json_path, 'w') as f_bi_final:
            json.dump(updated_split_entries_biased_, f_bi_final, indent=4)

    print("\n===== Script Completed =====")
    return c_index_train, c_index_test_mean, c_index_test_mean_wsi_orig, c_index_test_mean_wsi_ref

run_deep_learning_strat_contrastive(
    run_selected=False, 
    selected=None, 
    selected_name='run_cv',
    run_biased_cv=True
)
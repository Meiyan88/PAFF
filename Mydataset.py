"""
PyTorch Dataset class for the dataset. A placeholder. Users should implement their own
 dataset class.

Author: Ahmed H. Shahin
Date: 31/08/2023
"""
import json
import multiprocessing as mp
import os
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random
# from utils import histogram_matching_sitk, load_scan
from sklearn.model_selection import StratifiedKFold, KFold
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    RandRotated,
    RandFlipd,
    RandZoomd,
    RandAffined,
    ToTensord,
    Lambdad,
    ScaleIntensityd
)
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
    Lambda
)
from pycox.models import DeepHitSingle
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from sklearn.preprocessing import StandardScaler


class LiverDataset_disentangle(Dataset):
    def __init__(self, image_files, labels, censoreds, durations_discrete, events, ids, text, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms
        self.ids = ids
        self.durations_discrete = durations_discrete
        self.events = events
        self.censoreds = censoreds.astype(int)
        self.text = text

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        sample = {}
        # image = self.transforms(self.image_files[index])
        sample['image'] = str(np.char.replace(self.image_files[index], 'tumor2', 'liverspleen'))
        sample['liver'] = str(np.char.replace(self.image_files[index], 'tumor2', 'liver'))
        sample['tumor'] = str(np.char.replace(self.image_files[index], 'tumor2/data', 'tumor2/data2'))
        # sample['tumor'] = self.image_files[index]
        sample = self.transforms(sample)
        return (sample, self.labels[index], self.censoreds[index], self.durations_discrete[index] - 1,
                self.events[index], self.ids[index], self.text[index])

class LiverDataset_disentangle_cir(Dataset):
    def __init__(self, image_files, image_cir_files, labels, censoreds, durations_discrete, events, ids, text, transforms):
        self.image_files = image_files
        self.image_cir_files = image_cir_files
        self.labels = labels
        self.transforms = transforms
        self.ids = ids
        self.durations_discrete = durations_discrete
        self.events = events
        self.censoreds = censoreds.astype(int)
        self.text = text

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        sample = {}
        # image = self.transforms(self.image_files[index])
        sample['image'] = str(np.char.replace(self.image_files[index], 'tumor2', 'liverspleen'))
        sample['liver'] = str(np.char.replace(self.image_files[index], 'tumor2', 'liver'))
        sample['tumor'] = str(np.char.replace(self.image_files[index], 'tumor2/data', 'tumor2/data2'))
        sample['cir_liver'] = self.image_cir_files[index]
        # sample['tumor'] = self.image_files[index]
        sample = self.transforms(sample)
        return (sample, self.labels[index], self.censoreds[index], self.durations_discrete[index] - 1,
                self.events[index], self.ids[index], self.text[index])

class LiverDataset(Dataset):

    def __init__(self, image_files, labels, clinical_data, censoreds, ids, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms
        self.clinical_data = clinical_data.astype(np.float32)
        self.ids = ids
        self.censoreds = censoreds.astype(int)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = self.transforms(self.image_files[index])
        # xx = image.data.cpu().numpy()
        # xx = sitk.GetImageFromArray(xx[0])
        # xx.SetSpacing([1,1,1])
        # sitk.WriteImage(xx, 'example/monai.nii.gz')
        return image, self.clinical_data[index], self.censoreds[index], self.ids[index], self.labels[index]

class LiverDataset_disentangle_cir_feat(Dataset):
    def __init__(self, image_files, image_cir_files, feat_cir,
                 labels, censoreds, durations_discrete, events, ids, text, transforms):
        self.image_files = image_files
        self.image_cir_files = image_cir_files
        self.feat_cir = feat_cir
        self.labels = labels
        self.transforms = transforms
        self.ids = ids
        self.durations_discrete = durations_discrete
        self.events = events
        self.censoreds = censoreds.astype(int)
        self.text = text

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        sample = {}
        # image = self.transforms(self.image_files[index])
        sample['image'] = str(np.char.replace(self.image_files[index], 'tumor2', 'liverspleen'))
        sample['liver'] = str(np.char.replace(self.image_files[index], 'tumor2', 'liver'))
        sample['tumor'] = str(np.char.replace(self.image_files[index], 'tumor2/data', 'tumor2/data2'))
        sample['cir_liver'] = self.image_cir_files[index]
        # sample['tumor'] = self.image_files[index]
        sample = self.transforms(sample)
        return (sample, self.feat_cir[index], self.labels[index], self.censoreds[index], self.durations_discrete[index] - 1,
                self.events[index], self.ids[index], self.text[index])

class LiverDataset_disentangle_cir_feat_cli(Dataset):
    def __init__(self, image_files, image_cir_files, clinics, feat_cir,
                 labels, censoreds, durations_discrete, events, ids, text, transforms):
        self.image_files = image_files
        self.image_cir_files = image_cir_files
        self.feat_cir = feat_cir
        self.labels = labels
        self.transforms = transforms
        self.clinics = clinics
        self.ids = ids
        self.durations_discrete = durations_discrete
        self.events = events
        self.censoreds = censoreds.astype(int)
        self.text = text

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        sample = {}
        # image = self.transforms(self.image_files[index])
        sample['image'] = str(np.char.replace(self.image_files[index], 'tumor2', 'liverspleen'))
        sample['liver'] = str(np.char.replace(self.image_files[index], 'tumor2', 'liver'))
        sample['tumor'] = str(np.char.replace(self.image_files[index], 'tumor2/data', 'tumor2/data2'))
        sample['cir_liver'] = self.image_cir_files[index]
        # sample['tumor'] = self.image_files[index]
        sample = self.transforms(sample)
        return (sample, self.clinics[index], self.feat_cir[index], self.labels[index], self.censoreds[index], self.durations_discrete[index] - 1,
                self.events[index], self.ids[index], self.text[index])

class LiverDataset_disentangle_cir_feat_cli_radio(Dataset):
    def __init__(self, image_files, image_cir_files, clinics, feat_cir,
                 labels, censoreds, durations_discrete, events, ids, text, radio, transforms):
        self.image_files = image_files
        self.image_cir_files = image_cir_files
        self.feat_cir = feat_cir
        self.labels = labels
        self.transforms = transforms
        self.radio = radio
        self.clinics = clinics
        self.ids = ids
        self.durations_discrete = durations_discrete
        self.events = events
        self.censoreds = censoreds.astype(int)
        self.text = text

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        sample = {}
        # image = self.transforms(self.image_files[index])
        sample['image'] = str(np.char.replace(self.image_files[index], 'tumor2', 'liverspleen'))
        sample['liver'] = str(np.char.replace(self.image_files[index], 'tumor2', 'liver'))
        sample['tumor'] = str(np.char.replace(self.image_files[index], 'tumor2/data', 'tumor2/data2'))
        sample['cir_liver'] = self.image_cir_files[index]
        # sample['tumor'] = self.image_files[index]
        sample = self.transforms(sample)
        return (sample, self.clinics[index], self.feat_cir[index], self.labels[index], self.censoreds[index], self.durations_discrete[index] - 1,
                self.events[index], self.ids[index], self.text[index], self.radio[index])


class LiverDataset_disentangle_cir_feat_cli_txt(Dataset):
    def __init__(self, image_files, image_cir_files, clinics, txt_feats, feat_cir,
                 labels, censoreds, durations_discrete, events, ids, text, transforms):
        self.image_files = image_files
        self.image_cir_files = image_cir_files
        self.feat_cir = feat_cir
        self.labels = labels
        self.transforms = transforms
        self.clinics = clinics
        self.ids = ids
        self.durations_discrete = durations_discrete
        self.events = events
        self.censoreds = censoreds.astype(int)
        self.text = text
        self.txt_feats = txt_feats

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        sample = {}
        # image = self.transforms(self.image_files[index])
        sample['image'] = str(np.char.replace(self.image_files[index], 'tumor2', 'liverspleen'))
        sample['liver'] = str(np.char.replace(self.image_files[index], 'tumor2', 'liver'))
        sample['tumor'] = str(np.char.replace(self.image_files[index], 'tumor2/data', 'tumor2/data2'))
        sample['cir_liver'] = self.image_cir_files[index]
        # sample['tumor'] = self.image_files[index]
        sample = self.transforms(sample)
        return (sample, self.clinics[index], self.txt_feats[index], self.feat_cir[index], self.labels[index],
                self.censoreds[index], self.durations_discrete[index] - 1,
                self.events[index], self.ids[index], self.text[index])

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
def create_dataset_ext_week_disentangle(root_dir, fold, num_durations=6, batch_size=6, val_batch_size=4, seed=1998):
    # print_config()
    data_dir1 = root_dir
    hospital_names = (
        sorted([x for x in os.listdir(data_dir1) if os.path.isdir(os.path.join(data_dir1, x))]))
    image_files = [[os.path.join(data_dir1, class_name, x)
                     for x in os.listdir(os.path.join(data_dir1, class_name))]
                    for class_name in hospital_names]

    image_files_need = []
    info = pd.read_excel(r"/home/wangtao/project/Cirros_HCC/data/table/PFS_ZYJ_20240319_PD_CP_text.xlsx")
    ids = info['序号']
    hospitals = info['病例来源']

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']


    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        # continuous_clinicl_hospital = []
        # discrete_clinicl_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        for image_file in image_files_hospital:
            ID = image_file.split('/')[-1].split('.')[0]
            pos = np.where(ids.astype(str) == ID)
            if len(pos[0]) != 0:
                surv_label = surv_labels[pos][0]
                if_censored = if_censoreds[pos][0]
                text_now = gpt_text[pos][0]
                image_files_need_hospital.append(image_file)
                ids_hospital.append(ID)
                text_hospital.append(text_now)
                labels_hospital.append(surv_label)
                censored_hospital.append(if_censored)


        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        # continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        # discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        # continuous_clinicl_sum.append(continuous_clinicl_hospital)
        # discrete_clinicl_sum.append(discrete_clinicl_hospital)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    train_val_image_files = []
    train_val_labels = []
    train_val_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []

    train_image_files = []
    train_labels = []
    train_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []

    valid_image_files = []
    valid_labels = []
    valid_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []

    test_image_files = []
    test_labels = []
    test_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        labels_split = labels[hospital]
        # continuous_clinics_split = continuous_clinicl_sum[hospital]
        # discrete_clinics_split = discrete_clinicl_sum[hospital]
        # clinics_split = np.concatenate([continuous_clinics_split, discrete_clinics_split], axis=-1)
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        if hospital_name_split in ['Center2', 'Center3', 'Center4', 'Center8', 'Center9', 'Center5', 'Center11']:  # 0420_1
            train_val_image_files.append(image_files_split)
            train_val_labels.append(labels_split)
            # train_val_clinics.append(clinics_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
        else:
            test_image_files.append(image_files_split)
            test_labels.append(labels_split)
            # test_clinics.append(clinics_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)

    for hospital in range(len(train_val_image_files)):
        image_files_split = np.array(train_val_image_files[hospital])
        labels_split = train_val_labels[hospital]
        # clinics_split = train_val_clinics[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        censoreds_split = train_val_censoreds[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_labels.append(labels_split)
            # train_clinics.append(clinics_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
        else:
            for train_index, valid_index in kf.split(image_files_split, censoreds_split):
                _fold = _fold + 1
                if _fold == fold:
                    train_image_files_split = image_files_split[train_index]
                    train_image_files.append(train_image_files_split)
                    train_labels_split = labels_split[train_index]
                    train_labels.append(train_labels_split)
                    # train_clinics_split = clinics_split[train_index]
                    # train_clinics.append(train_clinics_split)
                    train_ids_split = ids_split[train_index]
                    train_ids.append(train_ids_split)
                    train_censoreds_split = censoreds_split[train_index]
                    train_censoreds.append(train_censoreds_split)
                    train_text_split = text_split[train_index]
                    train_text.append(train_text_split)

                    valid_image_files_split = image_files_split[valid_index]
                    valid_image_files.append(valid_image_files_split)
                    valid_labels_split = labels_split[valid_index]
                    valid_labels.append(valid_labels_split)
                    # valid_clinics_split = clinics_split[valid_index]
                    # valid_clinics.append(valid_clinics_split)
                    valid_ids_split = ids_split[valid_index]
                    valid_ids.append(valid_ids_split)
                    valid_censoreds_split = censoreds_split[valid_index]
                    valid_censoreds.append(valid_censoreds_split)
                    valid_text_split = text_split[valid_index]
                    valid_text.append(valid_text_split)

            _fold = 0

    # Scaler = StandardScaler()
    # OHencoder = OneHotEncoder(categories='auto', sparse=False)
    train_image_files = np.concatenate(train_image_files, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    # train_clinics = np.concatenate(train_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    train_text = np.concatenate(train_text, axis=0)

    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    # valid_clinics = np.concatenate(valid_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    valid_text = np.concatenate(valid_text, axis=0)

    test_image_files = np.concatenate(test_image_files, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    # test_clinics = np.concatenate(test_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    test_text = np.concatenate(test_text, axis=0)
    # test_clinics_con = test_clinics[:, :2]
    # test_clinics_dis = test_clinics[:, 2:]
    # test_clinics_con = Scaler.transform(test_clinics_con)
    # test_clinics_dis = OHencoder.transform(test_clinics_dis)
    # test_clinics = np.concatenate([test_clinics_con, test_clinics_dis], axis=-1)

    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)

    train_text = train_text.tolist()
    valid_text = valid_text.tolist()
    test_text = test_text.tolist()

    train_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor"], ),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor"], ),
        RandRotated(keys=["image", "liver", "tumor"], range_x=15, prob=0.5, keep_size=True),
        RandFlipd(keys=["image", "liver", "tumor"], spatial_axis=0, prob=0.2),
        RandZoomd(keys=["image", "liver", "tumor"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensord(keys=["image", "liver", "tumor"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor"]),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor"], ),
        ToTensord(keys=["image", "liver", "tumor"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    train_dataset = LiverDataset_disentangle(train_image_files, train_labels, train_censoreds, train_durations_discrete,
                                             train_events, train_ids, train_text, train_transforms)
    valid_dataset = LiverDataset_disentangle(valid_image_files, valid_labels, valid_censoreds, valid_durations_discrete,
                                             valid_events, valid_ids, valid_text, val_transforms)
    test_dataset = LiverDataset_disentangle(test_image_files, test_labels, test_censoreds, test_durations_discrete,
                                            test_events, test_ids, test_text, val_transforms)

    # train_infos = np.concatenate([train_ids[:, None], train_image_files[:, None], train_labels[:, None],
    #                          train_censoreds[:, None], train_clinics, np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], valid_image_files[:, None], valid_labels[:, None],
    #                               valid_censoreds[:, None], valid_clinics, np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], test_image_files[:, None], test_labels[:, None],
    #                               test_censoreds[:, None], test_clinics, np.array(test_text)[:, None]], axis=1)
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_infos.csv', train_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_infos.csv', valid_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_infos.csv', test_infos, fmt='%s',
    #            delimiter=',')
    # train_infos = np.concatenate([train_ids[:, None], np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], np.array(test_text)[:, None]], axis=1)
    # train_infos = pd.DataFrame(train_infos, columns=['ids', 'text'])
    # valid_infos = pd.DataFrame(valid_infos, columns=['ids', 'text'])
    # test_infos = pd.DataFrame(test_infos, columns=['ids', 'text'])
    # train_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_text.xlsx', index=False)
    # valid_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_text.xlsx', index=False)
    # test_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_text.xlsx', index=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=2)
    x_image_files, x_labels, x_censoreds, x_durations_discrete, \
        x_events, x_ids, x_text = next(iter(train_dataset))
    # for i, batch_data in enumerate(train_loader):
    #     s = 0
    # valid_sample = next(iter(valid_dataset))
    return train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader

def create_dataset_ext_week_disentangle_nocp3(root_dir, fold, num_durations=6, batch_size=6, val_batch_size=4, seed=1998):
    # print_config()
    data_dir1 = root_dir
    hospital_names = (
        sorted([x for x in os.listdir(data_dir1) if os.path.isdir(os.path.join(data_dir1, x))]))
    image_files = [[os.path.join(data_dir1, class_name, x)
                     for x in os.listdir(os.path.join(data_dir1, class_name))]
                    for class_name in hospital_names]

    image_files_need = []
    info = pd.read_excel(r"/home/wangtao/project/Cirros_HCC/data/table/PFS_ZYJ_20240319_PD_CP_text.xlsx")
    ids = info['序号']
    hospitals = info['病例来源']

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']


    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        # continuous_clinicl_hospital = []
        # discrete_clinicl_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        for image_file in image_files_hospital:
            ID = image_file.split('/')[-1].split('.')[0]
            pos = np.where(ids.astype(str) == ID)
            if len(pos[0]) != 0:
                cp_now = cp_labels[pos][0]
                if cp_now != 3 and np.isnan(cp_now) == False:
                    surv_label = surv_labels[pos][0]
                    if_censored = if_censoreds[pos][0]
                    text_now = gpt_text[pos][0]
                    image_files_need_hospital.append(image_file)
                    ids_hospital.append(ID)
                    text_hospital.append(text_now)
                    labels_hospital.append(surv_label)
                    censored_hospital.append(if_censored)

        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        # continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        # discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        # continuous_clinicl_sum.append(continuous_clinicl_hospital)
        # discrete_clinicl_sum.append(discrete_clinicl_hospital)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    train_val_image_files = []
    train_val_labels = []
    train_val_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []

    train_image_files = []
    train_labels = []
    train_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []

    valid_image_files = []
    valid_labels = []
    valid_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []

    test_image_files = []
    test_labels = []
    test_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        labels_split = labels[hospital]
        # continuous_clinics_split = continuous_clinicl_sum[hospital]
        # discrete_clinics_split = discrete_clinicl_sum[hospital]
        # clinics_split = np.concatenate([continuous_clinics_split, discrete_clinics_split], axis=-1)
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        if hospital_name_split in ['Center2', 'Center3', 'Center4', 'Center8', 'Center9', 'Center5', 'Center11']:  # 0420_1
            train_val_image_files.append(image_files_split)
            train_val_labels.append(labels_split)
            # train_val_clinics.append(clinics_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
        else:
            test_image_files.append(image_files_split)
            test_labels.append(labels_split)
            # test_clinics.append(clinics_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)

    for hospital in range(len(train_val_image_files)):
        image_files_split = np.array(train_val_image_files[hospital])
        labels_split = train_val_labels[hospital]
        # clinics_split = train_val_clinics[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        censoreds_split = train_val_censoreds[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_labels.append(labels_split)
            # train_clinics.append(clinics_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
        else:
            for train_index, valid_index in kf.split(image_files_split, censoreds_split):
                _fold = _fold + 1
                if _fold == fold:
                    train_image_files_split = image_files_split[train_index]
                    train_image_files.append(train_image_files_split)
                    train_labels_split = labels_split[train_index]
                    train_labels.append(train_labels_split)
                    # train_clinics_split = clinics_split[train_index]
                    # train_clinics.append(train_clinics_split)
                    train_ids_split = ids_split[train_index]
                    train_ids.append(train_ids_split)
                    train_censoreds_split = censoreds_split[train_index]
                    train_censoreds.append(train_censoreds_split)
                    train_text_split = text_split[train_index]
                    train_text.append(train_text_split)

                    valid_image_files_split = image_files_split[valid_index]
                    valid_image_files.append(valid_image_files_split)
                    valid_labels_split = labels_split[valid_index]
                    valid_labels.append(valid_labels_split)
                    # valid_clinics_split = clinics_split[valid_index]
                    # valid_clinics.append(valid_clinics_split)
                    valid_ids_split = ids_split[valid_index]
                    valid_ids.append(valid_ids_split)
                    valid_censoreds_split = censoreds_split[valid_index]
                    valid_censoreds.append(valid_censoreds_split)
                    valid_text_split = text_split[valid_index]
                    valid_text.append(valid_text_split)

            _fold = 0

    # Scaler = StandardScaler()
    # OHencoder = OneHotEncoder(categories='auto', sparse=False)
    train_image_files = np.concatenate(train_image_files, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    # train_clinics = np.concatenate(train_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    train_text = np.concatenate(train_text, axis=0)

    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    # valid_clinics = np.concatenate(valid_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    valid_text = np.concatenate(valid_text, axis=0)

    test_image_files = np.concatenate(test_image_files, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    # test_clinics = np.concatenate(test_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    test_text = np.concatenate(test_text, axis=0)
    # test_clinics_con = test_clinics[:, :2]
    # test_clinics_dis = test_clinics[:, 2:]
    # test_clinics_con = Scaler.transform(test_clinics_con)
    # test_clinics_dis = OHencoder.transform(test_clinics_dis)
    # test_clinics = np.concatenate([test_clinics_con, test_clinics_dis], axis=-1)

    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)

    train_text = train_text.tolist()
    valid_text = valid_text.tolist()
    test_text = test_text.tolist()

    train_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor"], ),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor"], ),
        RandRotated(keys=["image", "liver", "tumor"], range_x=15, prob=0.5, keep_size=True),
        RandFlipd(keys=["image", "liver", "tumor"], spatial_axis=0, prob=0.2),
        RandZoomd(keys=["image", "liver", "tumor"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensord(keys=["image", "liver", "tumor"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor"]),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor"], ),
        ToTensord(keys=["image", "liver", "tumor"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    train_dataset = LiverDataset_disentangle(train_image_files, train_labels, train_censoreds, train_durations_discrete,
                                             train_events, train_ids, train_text, train_transforms)
    valid_dataset = LiverDataset_disentangle(valid_image_files, valid_labels, valid_censoreds, valid_durations_discrete,
                                             valid_events, valid_ids, valid_text, val_transforms)
    test_dataset = LiverDataset_disentangle(test_image_files, test_labels, test_censoreds, test_durations_discrete,
                                            test_events, test_ids, test_text, val_transforms)

    # train_infos = np.concatenate([train_ids[:, None], train_image_files[:, None], train_labels[:, None],
    #                          train_censoreds[:, None], train_clinics, np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], valid_image_files[:, None], valid_labels[:, None],
    #                               valid_censoreds[:, None], valid_clinics, np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], test_image_files[:, None], test_labels[:, None],
    #                               test_censoreds[:, None], test_clinics, np.array(test_text)[:, None]], axis=1)
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_infos.csv', train_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_infos.csv', valid_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_infos.csv', test_infos, fmt='%s',
    #            delimiter=',')
    # train_infos = np.concatenate([train_ids[:, None], np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], np.array(test_text)[:, None]], axis=1)
    # train_infos = pd.DataFrame(train_infos, columns=['ids', 'text'])
    # valid_infos = pd.DataFrame(valid_infos, columns=['ids', 'text'])
    # test_infos = pd.DataFrame(test_infos, columns=['ids', 'text'])
    # train_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_text.xlsx', index=False)
    # valid_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_text.xlsx', index=False)
    # test_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_text.xlsx', index=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=2)
    x_image_files, x_labels, x_censoreds, x_durations_discrete, \
        x_events, x_ids, x_text = next(iter(train_dataset))
    # for i, batch_data in enumerate(train_loader):
    #     s = 0
    # valid_sample = next(iter(valid_dataset))
    return train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader

def create_dataset_ext_week_persudo_pair(matched_feature_file1, matched_feature_file2, matched_ids_file1, matched_ids_file2,
                                matched_censor_file1, matched_censor_file2, matched_surv_file1, matched_surv_file2,
                                         root_dir, fold, num_durations=6, batch_size=6, val_batch_size=4, seed=1998):
    features_cp0 = np.load(matched_feature_file1)
    features_cp1 = np.load(matched_feature_file2)[:, 0, ...]
    ids_cp0 = np.load(matched_ids_file1)
    ids_cp1 = np.load(matched_ids_file2)[:, 0, ...]
    censors_cp0 = np.load(matched_censor_file1)
    censors_cp1 = np.load(matched_censor_file2)
    survs_cp0 = np.load(matched_surv_file1)
    survs_cp1 = np.load(matched_surv_file2)

    match_ids_sum = np.concatenate([ids_cp0, ids_cp1])
    match_features_sum = np.concatenate([features_cp0, features_cp1])
    match_censors_sum = np.concatenate([censors_cp0, censors_cp1])
    match_survs_sum = np.concatenate([survs_cp0, survs_cp1])
    # print_config()
    data_dir1 = root_dir
    hospital_names = (
        sorted([x for x in os.listdir(data_dir1) if os.path.isdir(os.path.join(data_dir1, x))]))
    image_files = [[os.path.join(data_dir1, class_name, x)
                     for x in os.listdir(os.path.join(data_dir1, class_name))]
                    for class_name in hospital_names]

    image_files_need = []
    image_cir_files_need = []
    info = pd.read_excel(r"/home/wangtao/project/Cirros_HCC/data/table/PFS_ZYJ_20240319_PD_CP_text.xlsx")
    ids = info['序号']
    hospitals = info['病例来源']

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']


    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        # continuous_clinicl_hospital = []
        # discrete_clinicl_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        image_cir_files_need_hospital = []
        for image_file in image_files_hospital:
            ID = image_file.split('/')[-1].split('.')[0]
            center_now = image_file.split('/')[-2]
            pos = np.where(ids.astype(str) == ID)
            pos_match = np.where(match_ids_sum[:, -1] == center_now + '_' + ID)
            if len(pos[0]) != 0:
                cp_now = cp_labels[pos][0]
                if cp_now != 3 and np.isnan(cp_now) == False:
                    id_cir_now = match_ids_sum[pos_match[0], 0][0]
                    image_cir_file = os.path.join(r'/home/wangtao/project/Survival/data_cir/preprocess/Liver/data',
                                                  id_cir_now.split('_')[0], id_cir_now + '.nii.gz')
                    surv_label = surv_labels[pos][0]
                    if_censored = if_censoreds[pos][0]
                    text_now = gpt_text[pos][0]
                    image_files_need_hospital.append(image_file)
                    ids_hospital.append(ID)
                    text_hospital.append(text_now)
                    labels_hospital.append(surv_label)
                    censored_hospital.append(if_censored)
                    image_cir_files_need_hospital.append(image_cir_file)


        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        image_cir_files_need_hospital = np.array(image_cir_files_need_hospital)
        # continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        # discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        image_cir_files_need.append(image_cir_files_need_hospital)
        # continuous_clinicl_sum.append(continuous_clinicl_hospital)
        # discrete_clinicl_sum.append(discrete_clinicl_hospital)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    train_val_image_files = []
    train_val_image_cir_files = []
    train_val_labels = []
    train_val_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []

    train_image_files = []
    train_image_cir_files = []
    train_labels = []
    train_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []

    valid_image_files = []
    valid_image_cir_files = []
    valid_labels = []
    valid_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []

    test_image_files = []
    test_image_cir_files = []
    test_labels = []
    test_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        image_cir_files_split = np.array(image_cir_files_need[hospital])
        labels_split = labels[hospital]
        # continuous_clinics_split = continuous_clinicl_sum[hospital]
        # discrete_clinics_split = discrete_clinicl_sum[hospital]
        # clinics_split = np.concatenate([continuous_clinics_split, discrete_clinics_split], axis=-1)
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        if hospital_name_split in ['Center2', 'Center3', 'Center4', 'Center8', 'Center9', 'Center5', 'Center11']:  # 0420_1
            train_val_image_files.append(image_files_split)
            train_val_image_cir_files.append(image_cir_files_split)
            train_val_labels.append(labels_split)
            # train_val_clinics.append(clinics_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
        else:
            test_image_files.append(image_files_split)
            test_image_cir_files.append(image_cir_files_split)
            test_labels.append(labels_split)
            # test_clinics.append(clinics_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)

    for hospital in range(len(train_val_image_files)):
        image_files_split = np.array(train_val_image_files[hospital])
        image_cir_files_split = np.array(train_val_image_cir_files[hospital])
        labels_split = train_val_labels[hospital]
        # clinics_split = train_val_clinics[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        censoreds_split = train_val_censoreds[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_image_cir_files.append(image_cir_files_split)
            train_labels.append(labels_split)
            # train_clinics.append(clinics_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
        else:
            for train_index, valid_index in kf.split(image_files_split, censoreds_split):
                _fold = _fold + 1
                if _fold == fold:
                    train_image_files_split = image_files_split[train_index]
                    train_image_cir_files_split = image_cir_files_split[train_index]
                    train_image_files.append(train_image_files_split)
                    train_image_cir_files.append(train_image_cir_files_split)
                    train_labels_split = labels_split[train_index]
                    train_labels.append(train_labels_split)
                    # train_clinics_split = clinics_split[train_index]
                    # train_clinics.append(train_clinics_split)
                    train_ids_split = ids_split[train_index]
                    train_ids.append(train_ids_split)
                    train_censoreds_split = censoreds_split[train_index]
                    train_censoreds.append(train_censoreds_split)
                    train_text_split = text_split[train_index]
                    train_text.append(train_text_split)

                    valid_image_files_split = image_files_split[valid_index]
                    valid_image_cir_files_split = image_cir_files_split[valid_index]
                    valid_image_files.append(valid_image_files_split)
                    valid_image_cir_files.append(valid_image_cir_files_split)
                    valid_labels_split = labels_split[valid_index]
                    valid_labels.append(valid_labels_split)
                    # valid_clinics_split = clinics_split[valid_index]
                    # valid_clinics.append(valid_clinics_split)
                    valid_ids_split = ids_split[valid_index]
                    valid_ids.append(valid_ids_split)
                    valid_censoreds_split = censoreds_split[valid_index]
                    valid_censoreds.append(valid_censoreds_split)
                    valid_text_split = text_split[valid_index]
                    valid_text.append(valid_text_split)

            _fold = 0

    # Scaler = StandardScaler()
    # OHencoder = OneHotEncoder(categories='auto', sparse=False)
    train_image_files = np.concatenate(train_image_files, axis=0)
    train_image_cir_files = np.concatenate(train_image_cir_files, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    # train_clinics = np.concatenate(train_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    train_text = np.concatenate(train_text, axis=0)

    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_image_cir_files = np.concatenate(valid_image_cir_files, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    # valid_clinics = np.concatenate(valid_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    valid_text = np.concatenate(valid_text, axis=0)

    test_image_files = np.concatenate(test_image_files, axis=0)
    test_image_cir_files = np.concatenate(test_image_cir_files, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    # test_clinics = np.concatenate(test_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    test_text = np.concatenate(test_text, axis=0)
    # test_clinics_con = test_clinics[:, :2]
    # test_clinics_dis = test_clinics[:, 2:]
    # test_clinics_con = Scaler.transform(test_clinics_con)
    # test_clinics_dis = OHencoder.transform(test_clinics_dis)
    # test_clinics = np.concatenate([test_clinics_con, test_clinics_dis], axis=-1)

    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)

    train_text = train_text.tolist()
    valid_text = valid_text.tolist()
    test_text = test_text.tolist()

    train_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"], ),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        RandRotated(keys=["image", "liver", "tumor", "cir_liver"], range_x=15, prob=0.5, keep_size=True),
        RandFlipd(keys=["image", "liver", "tumor", "cir_liver"], spatial_axis=0, prob=0.2),
        RandZoomd(keys=["image", "liver", "tumor", "cir_liver"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    train_dataset = LiverDataset_disentangle_cir(train_image_files, train_image_cir_files, train_labels, train_censoreds,
                                                 train_durations_discrete, train_events, train_ids, train_text,
                                                 train_transforms)
    valid_dataset = LiverDataset_disentangle_cir(valid_image_files, valid_image_cir_files, valid_labels, valid_censoreds,
                                                 valid_durations_discrete, valid_events, valid_ids, valid_text,
                                                 val_transforms)
    test_dataset = LiverDataset_disentangle_cir(test_image_files, test_image_cir_files, test_labels, test_censoreds,
                                                test_durations_discrete, test_events, test_ids, test_text,
                                                val_transforms)

    # train_infos = np.concatenate([train_ids[:, None], train_image_files[:, None], train_labels[:, None],
    #                          train_censoreds[:, None], train_clinics, np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], valid_image_files[:, None], valid_labels[:, None],
    #                               valid_censoreds[:, None], valid_clinics, np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], test_image_files[:, None], test_labels[:, None],
    #                               test_censoreds[:, None], test_clinics, np.array(test_text)[:, None]], axis=1)
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_infos.csv', train_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_infos.csv', valid_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_infos.csv', test_infos, fmt='%s',
    #            delimiter=',')
    # train_infos = np.concatenate([train_ids[:, None], np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], np.array(test_text)[:, None]], axis=1)
    # train_infos = pd.DataFrame(train_infos, columns=['ids', 'text'])
    # valid_infos = pd.DataFrame(valid_infos, columns=['ids', 'text'])
    # test_infos = pd.DataFrame(test_infos, columns=['ids', 'text'])
    # train_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_text.xlsx', index=False)
    # valid_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_text.xlsx', index=False)
    # test_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_text.xlsx', index=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=2)
    x_image_files, x_labels, x_censoreds, x_durations_discrete, \
        x_events, x_ids, x_text = next(iter(train_loader))
    # for i, batch_data in enumerate(train_loader):
    #     s = 0
    # train_sample = next(iter(train_loader))
    return train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader


def create_dataset_ext_week_persudo_pair_addcirf(matched_feature_file1, matched_feature_file2, matched_ids_file1, matched_ids_file2,
                                matched_censor_file1, matched_censor_file2, matched_surv_file1, matched_surv_file2,
                                         root_dir, fold, num_durations=6, batch_size=6, val_batch_size=4, seed=1998):
    features_cp0 = np.load(matched_feature_file1)
    features_cp1 = np.load(matched_feature_file2)[:, 0, ...]
    ids_cp0 = np.load(matched_ids_file1)
    ids_cp1 = np.load(matched_ids_file2)[:, 0, ...]
    censors_cp0 = np.load(matched_censor_file1)
    censors_cp1 = np.load(matched_censor_file2)
    survs_cp0 = np.load(matched_surv_file1)
    survs_cp1 = np.load(matched_surv_file2)

    match_ids_sum = np.concatenate([ids_cp0, ids_cp1])
    match_features_sum = np.concatenate([features_cp0, features_cp1])
    match_censors_sum = np.concatenate([censors_cp0, censors_cp1])
    match_survs_sum = np.concatenate([survs_cp0, survs_cp1])
    # print_config()
    data_dir1 = root_dir
    hospital_names = (
        sorted([x for x in os.listdir(data_dir1) if os.path.isdir(os.path.join(data_dir1, x))]))
    image_files = [[os.path.join(data_dir1, class_name, x)
                     for x in os.listdir(os.path.join(data_dir1, class_name))]
                    for class_name in hospital_names]

    image_files_need = []
    image_cir_files_need = []
    feat_cir_need = []
    info = pd.read_excel(r"/home/wangtao/project/Cirros_HCC/data/table/PFS_ZYJ_20240319_PD_CP_text.xlsx")
    ids = info['序号']
    hospitals = info['病例来源']

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']


    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        # continuous_clinicl_hospital = []
        # discrete_clinicl_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        image_cir_files_need_hospital = []
        feat_cir_need_hospital = []
        for image_file in image_files_hospital:
            ID = image_file.split('/')[-1].split('.')[0]
            center_now = image_file.split('/')[-2]
            pos = np.where(ids.astype(str) == ID)
            pos_match = np.where(match_ids_sum[:, -1] == center_now + '_' + ID)
            if len(pos[0]) != 0:
                cp_now = cp_labels[pos][0]
                if cp_now != 3 and np.isnan(cp_now) == False:
                    id_cir_now = match_ids_sum[pos_match[0], 0][0]
                    feat_cir_now = match_features_sum[pos_match[0], 0][0][None, :]
                    image_cir_file = os.path.join(r'/home/wangtao/project/Survival/data_cir/preprocess/Liver/data',
                                                  id_cir_now.split('_')[0], id_cir_now + '.nii.gz')
                    surv_label = surv_labels[pos][0]
                    if_censored = if_censoreds[pos][0]
                    text_now = gpt_text[pos][0]
                    image_files_need_hospital.append(image_file)
                    ids_hospital.append(ID)
                    text_hospital.append(text_now)
                    labels_hospital.append(surv_label)
                    censored_hospital.append(if_censored)
                    image_cir_files_need_hospital.append(image_cir_file)
                    feat_cir_need_hospital.append(feat_cir_now)


        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        image_cir_files_need_hospital = np.array(image_cir_files_need_hospital)
        feat_cir_need_hospital = np.concatenate(feat_cir_need_hospital)
        # continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        # discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        image_cir_files_need.append(image_cir_files_need_hospital)
        # continuous_clinicl_sum.append(continuous_clinicl_hospital)
        # discrete_clinicl_sum.append(discrete_clinicl_hospital)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)
        feat_cir_need.append(feat_cir_need_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    train_val_image_files = []
    train_val_image_cir_files = []
    train_val_feat_cir = []
    train_val_labels = []
    train_val_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []

    train_image_files = []
    train_image_cir_files = []
    train_feat_cir = []
    train_labels = []
    train_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []

    valid_image_files = []
    valid_image_cir_files = []
    valid_feat_cir = []
    valid_labels = []
    valid_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []

    test_image_files = []
    test_image_cir_files = []
    test_feat_cir = []
    test_labels = []
    test_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        image_cir_files_split = np.array(image_cir_files_need[hospital])
        feat_cir_split = feat_cir_need[hospital]
        labels_split = labels[hospital]
        # continuous_clinics_split = continuous_clinicl_sum[hospital]
        # discrete_clinics_split = discrete_clinicl_sum[hospital]
        # clinics_split = np.concatenate([continuous_clinics_split, discrete_clinics_split], axis=-1)
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        if hospital_name_split in ['Center2', 'Center3', 'Center4', 'Center8', 'Center9', 'Center5', 'Center11']:  # 0420_1
            train_val_image_files.append(image_files_split)
            train_val_image_cir_files.append(image_cir_files_split)
            train_val_labels.append(labels_split)
            # train_val_clinics.append(clinics_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
            train_val_feat_cir.append(feat_cir_split)
        else:
            test_image_files.append(image_files_split)
            test_image_cir_files.append(image_cir_files_split)
            test_labels.append(labels_split)
            # test_clinics.append(clinics_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)
            test_feat_cir.append(feat_cir_split)

    for hospital in range(len(train_val_image_files)):
        image_files_split = np.array(train_val_image_files[hospital])
        image_cir_files_split = np.array(train_val_image_cir_files[hospital])
        labels_split = train_val_labels[hospital]
        feat_cir_split = train_val_feat_cir[hospital]
        # clinics_split = train_val_clinics[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        censoreds_split = train_val_censoreds[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_image_cir_files.append(image_cir_files_split)
            train_labels.append(labels_split)
            # train_clinics.append(clinics_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
            train_feat_cir.append(feat_cir_split)
        else:
            for train_index, valid_index in kf.split(image_files_split, censoreds_split):
                _fold = _fold + 1
                if _fold == fold:
                    train_image_files_split = image_files_split[train_index]
                    train_image_cir_files_split = image_cir_files_split[train_index]
                    train_image_files.append(train_image_files_split)
                    train_image_cir_files.append(train_image_cir_files_split)
                    train_feat_cir_split = feat_cir_split[train_index]
                    train_feat_cir.append(train_feat_cir_split)
                    train_labels_split = labels_split[train_index]
                    train_labels.append(train_labels_split)
                    # train_clinics_split = clinics_split[train_index]
                    # train_clinics.append(train_clinics_split)
                    train_ids_split = ids_split[train_index]
                    train_ids.append(train_ids_split)
                    train_censoreds_split = censoreds_split[train_index]
                    train_censoreds.append(train_censoreds_split)
                    train_text_split = text_split[train_index]
                    train_text.append(train_text_split)

                    valid_image_files_split = image_files_split[valid_index]
                    valid_image_cir_files_split = image_cir_files_split[valid_index]
                    valid_image_files.append(valid_image_files_split)
                    valid_image_cir_files.append(valid_image_cir_files_split)
                    valid_feat_cir_split = feat_cir_split[valid_index]
                    valid_feat_cir.append(valid_feat_cir_split)
                    valid_labels_split = labels_split[valid_index]
                    valid_labels.append(valid_labels_split)
                    # valid_clinics_split = clinics_split[valid_index]
                    # valid_clinics.append(valid_clinics_split)
                    valid_ids_split = ids_split[valid_index]
                    valid_ids.append(valid_ids_split)
                    valid_censoreds_split = censoreds_split[valid_index]
                    valid_censoreds.append(valid_censoreds_split)
                    valid_text_split = text_split[valid_index]
                    valid_text.append(valid_text_split)

            _fold = 0

    # Scaler = StandardScaler()
    # OHencoder = OneHotEncoder(categories='auto', sparse=False)
    train_image_files = np.concatenate(train_image_files, axis=0)
    train_image_cir_files = np.concatenate(train_image_cir_files, axis=0)
    train_feat_cir = np.concatenate(train_feat_cir, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    # train_clinics = np.concatenate(train_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    train_text = np.concatenate(train_text, axis=0)

    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_image_cir_files = np.concatenate(valid_image_cir_files, axis=0)
    valid_feat_cir = np.concatenate(valid_feat_cir, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    # valid_clinics = np.concatenate(valid_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    valid_text = np.concatenate(valid_text, axis=0)

    test_image_files = np.concatenate(test_image_files, axis=0)
    test_image_cir_files = np.concatenate(test_image_cir_files, axis=0)
    test_feat_cir = np.concatenate(test_feat_cir, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    # test_clinics = np.concatenate(test_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    test_text = np.concatenate(test_text, axis=0)
    # test_clinics_con = test_clinics[:, :2]
    # test_clinics_dis = test_clinics[:, 2:]
    # test_clinics_con = Scaler.transform(test_clinics_con)
    # test_clinics_dis = OHencoder.transform(test_clinics_dis)
    # test_clinics = np.concatenate([test_clinics_con, test_clinics_dis], axis=-1)

    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)

    train_text = train_text.tolist()
    valid_text = valid_text.tolist()
    test_text = test_text.tolist()

    train_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"], ),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        RandRotated(keys=["image", "liver", "tumor", "cir_liver"], range_x=15, prob=0.5, keep_size=True),
        RandFlipd(keys=["image", "liver", "tumor", "cir_liver"], spatial_axis=0, prob=0.2),
        RandZoomd(keys=["image", "liver", "tumor", "cir_liver"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    train_dataset = LiverDataset_disentangle_cir_feat(train_image_files, train_image_cir_files, train_feat_cir,
                                                      train_labels, train_censoreds, train_durations_discrete,
                                                      train_events, train_ids, train_text, train_transforms)
    valid_dataset = LiverDataset_disentangle_cir_feat(valid_image_files, valid_image_cir_files, valid_feat_cir,
                                                      valid_labels, valid_censoreds, valid_durations_discrete,
                                                      valid_events, valid_ids, valid_text,  val_transforms)
    test_dataset = LiverDataset_disentangle_cir_feat(test_image_files, test_image_cir_files, test_feat_cir,
                                                     test_labels, test_censoreds, test_durations_discrete,
                                                     test_events, test_ids, test_text, val_transforms)

    # train_infos = np.concatenate([train_ids[:, None], train_image_files[:, None], train_labels[:, None],
    #                          train_censoreds[:, None], train_clinics, np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], valid_image_files[:, None], valid_labels[:, None],
    #                               valid_censoreds[:, None], valid_clinics, np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], test_image_files[:, None], test_labels[:, None],
    #                               test_censoreds[:, None], test_clinics, np.array(test_text)[:, None]], axis=1)
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_infos.csv', train_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_infos.csv', valid_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_infos.csv', test_infos, fmt='%s',
    #            delimiter=',')
    # train_infos = np.concatenate([train_ids[:, None], np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], np.array(test_text)[:, None]], axis=1)
    # train_infos = pd.DataFrame(train_infos, columns=['ids', 'text'])
    # valid_infos = pd.DataFrame(valid_infos, columns=['ids', 'text'])
    # test_infos = pd.DataFrame(test_infos, columns=['ids', 'text'])
    # train_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_text.xlsx', index=False)
    # valid_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_text.xlsx', index=False)
    # test_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_text.xlsx', index=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=2)
    x_image_files, x_cir_feat, x_labels, x_censoreds, x_durations_discrete, \
        x_events, x_ids, x_text = next(iter(train_loader))
    # for i, batch_data in enumerate(train_loader):
    #     s = 0
    # train_sample = next(iter(train_loader))
    return train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader


def create_dataset_ext_week_persudo_pair_addcirf(matched_feature_file1, matched_feature_file2, matched_ids_file1, matched_ids_file2,
                                matched_censor_file1, matched_censor_file2, matched_surv_file1, matched_surv_file2,
                                         root_dir, fold, num_durations=6, batch_size=6, val_batch_size=4, seed=1998):
    features_cp0 = np.load(matched_feature_file1)
    features_cp1 = np.load(matched_feature_file2)[:, 0, ...]
    ids_cp0 = np.load(matched_ids_file1)
    ids_cp1 = np.load(matched_ids_file2)[:, 0, ...]
    censors_cp0 = np.load(matched_censor_file1)
    censors_cp1 = np.load(matched_censor_file2)
    survs_cp0 = np.load(matched_surv_file1)
    survs_cp1 = np.load(matched_surv_file2)

    match_ids_sum = np.concatenate([ids_cp0, ids_cp1])
    match_features_sum = np.concatenate([features_cp0, features_cp1])
    match_censors_sum = np.concatenate([censors_cp0, censors_cp1])
    match_survs_sum = np.concatenate([survs_cp0, survs_cp1])
    # print_config()
    data_dir1 = root_dir
    hospital_names = (
        sorted([x for x in os.listdir(data_dir1) if os.path.isdir(os.path.join(data_dir1, x))]))
    image_files = [[os.path.join(data_dir1, class_name, x)
                     for x in os.listdir(os.path.join(data_dir1, class_name))]
                    for class_name in hospital_names]

    image_files_need = []
    image_cir_files_need = []
    feat_cir_need = []
    info = pd.read_excel(r"/home/wangtao/project/Cirros_HCC/data/table/PFS_ZYJ_20240319_PD_CP_text.xlsx")
    ids = info['序号']
    hospitals = info['病例来源']

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']


    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        # continuous_clinicl_hospital = []
        # discrete_clinicl_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        image_cir_files_need_hospital = []
        feat_cir_need_hospital = []
        for image_file in image_files_hospital:
            ID = image_file.split('/')[-1].split('.')[0]
            center_now = image_file.split('/')[-2]
            pos = np.where(ids.astype(str) == ID)
            pos_match = np.where(match_ids_sum[:, -1] == center_now + '_' + ID)
            if len(pos[0]) != 0:
                cp_now = cp_labels[pos][0]
                if cp_now != 3 and np.isnan(cp_now) == False:
                    id_cir_now = match_ids_sum[pos_match[0], 0][0]
                    feat_cir_now = match_features_sum[pos_match[0], 0][0][None, :]
                    image_cir_file = os.path.join(r'/home/wangtao/project/Survival/data_cir/preprocess/Liver/data',
                                                  id_cir_now.split('_')[0], id_cir_now + '.nii.gz')
                    surv_label = surv_labels[pos][0]
                    if_censored = if_censoreds[pos][0]
                    text_now = gpt_text[pos][0]
                    image_files_need_hospital.append(image_file)
                    ids_hospital.append(ID)
                    text_hospital.append(text_now)
                    labels_hospital.append(surv_label)
                    censored_hospital.append(if_censored)
                    image_cir_files_need_hospital.append(image_cir_file)
                    feat_cir_need_hospital.append(feat_cir_now)


        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        image_cir_files_need_hospital = np.array(image_cir_files_need_hospital)
        feat_cir_need_hospital = np.concatenate(feat_cir_need_hospital)
        # continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        # discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        image_cir_files_need.append(image_cir_files_need_hospital)
        # continuous_clinicl_sum.append(continuous_clinicl_hospital)
        # discrete_clinicl_sum.append(discrete_clinicl_hospital)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)
        feat_cir_need.append(feat_cir_need_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    train_val_image_files = []
    train_val_image_cir_files = []
    train_val_feat_cir = []
    train_val_labels = []
    train_val_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []

    train_image_files = []
    train_image_cir_files = []
    train_feat_cir = []
    train_labels = []
    train_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []

    valid_image_files = []
    valid_image_cir_files = []
    valid_feat_cir = []
    valid_labels = []
    valid_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []

    test_image_files = []
    test_image_cir_files = []
    test_feat_cir = []
    test_labels = []
    test_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        image_cir_files_split = np.array(image_cir_files_need[hospital])
        feat_cir_split = feat_cir_need[hospital]
        labels_split = labels[hospital]
        # continuous_clinics_split = continuous_clinicl_sum[hospital]
        # discrete_clinics_split = discrete_clinicl_sum[hospital]
        # clinics_split = np.concatenate([continuous_clinics_split, discrete_clinics_split], axis=-1)
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        # if hospital_name_split not in ['Center1', 'Center6', 'Center8']:  #1
        if hospital_name_split not in ['Center5', 'Center9']:  #
            # if hospital_name_split in ['Center2', 'Center3', 'Center4', 'Center5', 'Center8', 'Center9', 'Center11']:  # 0420_1 - Center1 Center6 Center10

            train_val_image_files.append(image_files_split)
            train_val_image_cir_files.append(image_cir_files_split)
            train_val_labels.append(labels_split)
            # train_val_clinics.append(clinics_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
            train_val_feat_cir.append(feat_cir_split)
        else:
            test_image_files.append(image_files_split)
            test_image_cir_files.append(image_cir_files_split)
            test_labels.append(labels_split)
            # test_clinics.append(clinics_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)
            test_feat_cir.append(feat_cir_split)

    for hospital in range(len(train_val_image_files)):
        image_files_split = np.array(train_val_image_files[hospital])
        image_cir_files_split = np.array(train_val_image_cir_files[hospital])
        labels_split = train_val_labels[hospital]
        feat_cir_split = train_val_feat_cir[hospital]
        # clinics_split = train_val_clinics[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        censoreds_split = train_val_censoreds[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_image_cir_files.append(image_cir_files_split)
            train_labels.append(labels_split)
            # train_clinics.append(clinics_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
            train_feat_cir.append(feat_cir_split)
        else:
            for train_index, valid_index in kf.split(image_files_split, censoreds_split):
                _fold = _fold + 1
                if _fold == fold:
                    train_image_files_split = image_files_split[train_index]
                    train_image_cir_files_split = image_cir_files_split[train_index]
                    train_image_files.append(train_image_files_split)
                    train_image_cir_files.append(train_image_cir_files_split)
                    train_feat_cir_split = feat_cir_split[train_index]
                    train_feat_cir.append(train_feat_cir_split)
                    train_labels_split = labels_split[train_index]
                    train_labels.append(train_labels_split)
                    # train_clinics_split = clinics_split[train_index]
                    # train_clinics.append(train_clinics_split)
                    train_ids_split = ids_split[train_index]
                    train_ids.append(train_ids_split)
                    train_censoreds_split = censoreds_split[train_index]
                    train_censoreds.append(train_censoreds_split)
                    train_text_split = text_split[train_index]
                    train_text.append(train_text_split)

                    valid_image_files_split = image_files_split[valid_index]
                    valid_image_cir_files_split = image_cir_files_split[valid_index]
                    valid_image_files.append(valid_image_files_split)
                    valid_image_cir_files.append(valid_image_cir_files_split)
                    valid_feat_cir_split = feat_cir_split[valid_index]
                    valid_feat_cir.append(valid_feat_cir_split)
                    valid_labels_split = labels_split[valid_index]
                    valid_labels.append(valid_labels_split)
                    # valid_clinics_split = clinics_split[valid_index]
                    # valid_clinics.append(valid_clinics_split)
                    valid_ids_split = ids_split[valid_index]
                    valid_ids.append(valid_ids_split)
                    valid_censoreds_split = censoreds_split[valid_index]
                    valid_censoreds.append(valid_censoreds_split)
                    valid_text_split = text_split[valid_index]
                    valid_text.append(valid_text_split)
            # np.savetxt()

            _fold = 0

    # Scaler = StandardScaler()
    # OHencoder = OneHotEncoder(categories='auto', sparse=False)
    train_image_files = np.concatenate(train_image_files, axis=0)
    train_image_cir_files = np.concatenate(train_image_cir_files, axis=0)
    train_feat_cir = np.concatenate(train_feat_cir, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    # train_clinics = np.concatenate(train_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    train_text = np.concatenate(train_text, axis=0)

    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_image_cir_files = np.concatenate(valid_image_cir_files, axis=0)
    valid_feat_cir = np.concatenate(valid_feat_cir, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    # valid_clinics = np.concatenate(valid_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    valid_text = np.concatenate(valid_text, axis=0)

    test_image_files = np.concatenate(test_image_files, axis=0)
    test_image_cir_files = np.concatenate(test_image_cir_files, axis=0)
    test_feat_cir = np.concatenate(test_feat_cir, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    # test_clinics = np.concatenate(test_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    test_text = np.concatenate(test_text, axis=0)
    # test_clinics_con = test_clinics[:, :2]
    # test_clinics_dis = test_clinics[:, 2:]
    # test_clinics_con = Scaler.transform(test_clinics_con)
    # test_clinics_dis = OHencoder.transform(test_clinics_dis)
    # test_clinics = np.concatenate([test_clinics_con, test_clinics_dis], axis=-1)

    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)

    train_text = train_text.tolist()
    valid_text = valid_text.tolist()
    test_text = test_text.tolist()

    train_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"], ),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        RandRotated(keys=["image", "liver", "tumor", "cir_liver"], range_x=15, prob=0.5, keep_size=True),
        RandFlipd(keys=["image", "liver", "tumor", "cir_liver"], spatial_axis=0, prob=0.2),
        RandZoomd(keys=["image", "liver", "tumor", "cir_liver"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    train_dataset = LiverDataset_disentangle_cir_feat(train_image_files, train_image_cir_files, train_feat_cir,
                                                      train_labels, train_censoreds, train_durations_discrete,
                                                      train_events, train_ids, train_text, train_transforms)
    valid_dataset = LiverDataset_disentangle_cir_feat(valid_image_files, valid_image_cir_files, valid_feat_cir,
                                                      valid_labels, valid_censoreds, valid_durations_discrete,
                                                      valid_events, valid_ids, valid_text,  val_transforms)
    test_dataset = LiverDataset_disentangle_cir_feat(test_image_files, test_image_cir_files, test_feat_cir,
                                                     test_labels, test_censoreds, test_durations_discrete,
                                                     test_events, test_ids, test_text, val_transforms)

    # train_infos = np.concatenate([train_ids[:, None], train_image_files[:, None], train_labels[:, None],
    #                          train_censoreds[:, None], train_clinics, np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], valid_image_files[:, None], valid_labels[:, None],
    #                               valid_censoreds[:, None], valid_clinics, np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], test_image_files[:, None], test_labels[:, None],
    #                               test_censoreds[:, None], test_clinics, np.array(test_text)[:, None]], axis=1)
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_infos.csv', train_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_infos.csv', valid_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_infos.csv', test_infos, fmt='%s',
    #            delimiter=',')
    # train_infos = np.concatenate([train_ids[:, None], np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], np.array(test_text)[:, None]], axis=1)
    # train_infos = pd.DataFrame(train_infos, columns=['ids', 'text'])
    # valid_infos = pd.DataFrame(valid_infos, columns=['ids', 'text'])
    # test_infos = pd.DataFrame(test_infos, columns=['ids', 'text'])
    # train_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_text.xlsx', index=False)
    # valid_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_text.xlsx', index=False)
    # test_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_text.xlsx', index=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=2)
    x_image_files, x_cir_feat, x_labels, x_censoreds, x_durations_discrete, \
        x_events, x_ids, x_text = next(iter(train_loader))
    # for i, batch_data in enumerate(train_loader):
    #     s = 0
    # train_sample = next(iter(train_loader))
    return train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader

def create_dataset_ext_week_persudo_pair_addcirf_fixid(matched_feature_file1, matched_feature_file2, matched_ids_file1, matched_ids_file2,
                                matched_censor_file1, matched_censor_file2, matched_surv_file1, matched_surv_file2,
                                         root_dir, fold, num_durations=6, batch_size=6, val_batch_size=4, seed=1998):
    features_cp0 = np.load(matched_feature_file1)
    features_cp1 = np.load(matched_feature_file2)[:, 0, ...]
    ids_cp0 = np.load(matched_ids_file1)
    ids_cp1 = np.load(matched_ids_file2)[:, 0, ...]
    censors_cp0 = np.load(matched_censor_file1)
    censors_cp1 = np.load(matched_censor_file2)
    survs_cp0 = np.load(matched_surv_file1)
    survs_cp1 = np.load(matched_surv_file2)

    match_ids_sum = np.concatenate([ids_cp0, ids_cp1])
    match_features_sum = np.concatenate([features_cp0, features_cp1])
    match_censors_sum = np.concatenate([censors_cp0, censors_cp1])
    match_survs_sum = np.concatenate([survs_cp0, survs_cp1])
    # print_config()
    data_dir1 = root_dir
    hospital_names = (
        sorted([x for x in os.listdir(data_dir1) if os.path.isdir(os.path.join(data_dir1, x))]))
    image_files = [[os.path.join(data_dir1, class_name, x)
                     for x in os.listdir(os.path.join(data_dir1, class_name))]
                    for class_name in hospital_names]

    image_files_need = []
    image_cir_files_need = []
    feat_cir_need = []
    info = pd.read_excel(r"/home/wangtao/project/Cirros_HCC/data/table/PFS_ZYJ_20240319_PD_CP_text.xlsx")
    ids = info['序号']
    hospitals = info['病例来源']

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']

    continuous_clinicl = info.iloc[:, [5, 10]]
    discrete_clinicl = info.iloc[:, [6, 7, 8, 9, 11, 12, 23]]
    continuous_clinicl = np.asarray(continuous_clinicl)
    discrete_clinicl = np.asarray(discrete_clinicl)
    discrete_clinicl[np.where(np.isnan(discrete_clinicl)==True)] = 1
    OHencoder = OneHotEncoder(categories='auto', sparse_output=False)
    discrete_clinicl = OHencoder.fit_transform(discrete_clinicl)

    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        continuous_clinicl_hospital = []
        discrete_clinicl_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        image_cir_files_need_hospital = []
        feat_cir_need_hospital = []
        for image_file in image_files_hospital:
            ID = image_file.split('/')[-1].split('.')[0]
            center_now = image_file.split('/')[-2]
            pos = np.where(ids.astype(str) == ID)
            pos_match = np.where(match_ids_sum[:, -1] == center_now + '_' + ID)
            if len(pos[0]) != 0:
                cp_now = cp_labels[pos][0]
                if cp_now != 3 and np.isnan(cp_now) == False:
                    id_cir_now = match_ids_sum[pos_match[0], 0][0]
                    feat_cir_now = match_features_sum[pos_match[0], 0][0][None, :]
                    image_cir_file = os.path.join(r'/home/wangtao/project/Survival/data_cir/preprocess/Liver/data',
                                                  id_cir_now.split('_')[0], id_cir_now + '.nii.gz')
                    surv_label = surv_labels[pos][0]
                    if_censored = if_censoreds[pos][0]
                    text_now = gpt_text[pos][0]

                    continuous_clinicl_now = continuous_clinicl[pos][0]
                    discrete_clinicl_now = discrete_clinicl[pos][0]
                    image_files_need_hospital.append(image_file)
                    ids_hospital.append(ID)
                    text_hospital.append(text_now)
                    labels_hospital.append(surv_label)
                    censored_hospital.append(if_censored)
                    image_cir_files_need_hospital.append(image_cir_file)
                    feat_cir_need_hospital.append(feat_cir_now)
                    continuous_clinicl_hospital.append(continuous_clinicl_now)
                    discrete_clinicl_hospital.append(discrete_clinicl_now)


        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        image_cir_files_need_hospital = np.array(image_cir_files_need_hospital)
        feat_cir_need_hospital = np.concatenate(feat_cir_need_hospital)
        continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        image_cir_files_need.append(image_cir_files_need_hospital)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)
        feat_cir_need.append(feat_cir_need_hospital)
        continuous_clinicl_sum.append(continuous_clinicl_hospital)
        discrete_clinicl_sum.append(discrete_clinicl_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    train_val_image_files = []
    train_val_image_cir_files = []
    train_val_feat_cir = []
    train_val_labels = []
    train_val_con_clinics = []
    train_val_dis_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []

    train_image_files = []
    train_image_cir_files = []
    train_feat_cir = []
    train_labels = []
    train_con_clinics = []
    train_dis_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []

    valid_image_files = []
    valid_image_cir_files = []
    valid_feat_cir = []
    valid_labels = []
    valid_con_clinics = []
    valid_dis_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []

    test_image_files = []
    test_image_cir_files = []
    test_feat_cir = []
    test_labels = []
    test_con_clinics = []
    test_dis_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        image_cir_files_split = np.array(image_cir_files_need[hospital])
        feat_cir_split = feat_cir_need[hospital]
        labels_split = labels[hospital]
        # continuous_clinics_split = continuous_clinicl_sum[hospital]
        # discrete_clinics_split = discrete_clinicl_sum[hospital]
        # clinics_split = np.concatenate([continuous_clinics_split, discrete_clinics_split], axis=-1)
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        continuous_clinicl_split = continuous_clinicl_sum[hospital]
        discrete_clinicl_split = discrete_clinicl_sum[hospital]
        # if hospital_name_split in ['Center2', 'Center3', 'Center4', 'Center8', 'Center9', 'Center5', 'Center11']:  # 0420_1
        if hospital_name_split not in ['Center5', 'Center9']:
            train_val_image_files.append(image_files_split)
            train_val_image_cir_files.append(image_cir_files_split)
            train_val_labels.append(labels_split)
            # train_val_clinics.append(clinics_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
            train_val_feat_cir.append(feat_cir_split)
            train_val_dis_clinics.append(discrete_clinicl_split)
            train_val_con_clinics.append(continuous_clinicl_split)
        else:
            test_image_files.append(image_files_split)
            test_image_cir_files.append(image_cir_files_split)
            test_labels.append(labels_split)
            # test_clinics.append(clinics_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)
            test_feat_cir.append(feat_cir_split)
            test_dis_clinics.append(discrete_clinicl_split)
            test_con_clinics.append(continuous_clinicl_split)

    for hospital in range(len(train_val_image_files)):
        image_files_split = np.array(train_val_image_files[hospital])
        image_cir_files_split = np.array(train_val_image_cir_files[hospital])
        labels_split = train_val_labels[hospital]
        feat_cir_split = train_val_feat_cir[hospital]
        # clinics_split = train_val_clinics[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        censoreds_split = train_val_censoreds[hospital]
        continuous_clinicl_split = train_val_con_clinics[hospital]
        discrete_clinicl_split = train_val_dis_clinics[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_image_cir_files.append(image_cir_files_split)
            train_labels.append(labels_split)
            train_dis_clinics.append(discrete_clinicl_split)
            train_con_clinics.append(continuous_clinicl_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
            train_feat_cir.append(feat_cir_split)
        else:
            # for train_index, valid_index in kf.split(image_files_split, censoreds_split):
            #     _fold = _fold + 1
            #     if _fold == fold:
            train_index = np.loadtxt(r'/home/wangtao/project/Survival/split_ids/train_index{}_fold{}_nt59.csv'.format(hospital, fold),
                                     delimiter=',', dtype=str)
            train_index = train_index.astype(int)
            valid_index = np.loadtxt(r'/home/wangtao/project/Survival/split_ids/valid_index{}_fold{}_nt59.csv'.format(hospital, fold),
                                     delimiter=',', dtype=str)
            valid_index = valid_index.astype(int)
            train_image_files_split = image_files_split[train_index]
            train_image_cir_files_split = image_cir_files_split[train_index]
            train_image_files.append(train_image_files_split)
            train_image_cir_files.append(train_image_cir_files_split)
            train_feat_cir_split = feat_cir_split[train_index]
            train_feat_cir.append(train_feat_cir_split)
            train_labels_split = labels_split[train_index]
            train_labels.append(train_labels_split)

            train_ids_split = ids_split[train_index]
            train_ids.append(train_ids_split)
            train_censoreds_split = censoreds_split[train_index]
            train_censoreds.append(train_censoreds_split)
            train_text_split = text_split[train_index]
            train_text.append(train_text_split)

            train_discrete_clinicl_split = discrete_clinicl_split[train_index]
            train_dis_clinics.append(train_discrete_clinicl_split)

            train_continuous_clinicl_split = continuous_clinicl_split[train_index]
            train_con_clinics.append(train_continuous_clinicl_split)

            valid_image_files_split = image_files_split[valid_index]
            valid_image_cir_files_split = image_cir_files_split[valid_index]
            valid_image_files.append(valid_image_files_split)
            valid_image_cir_files.append(valid_image_cir_files_split)
            valid_feat_cir_split = feat_cir_split[valid_index]
            valid_feat_cir.append(valid_feat_cir_split)
            valid_labels_split = labels_split[valid_index]
            valid_labels.append(valid_labels_split)
            # valid_clinics_split = clinics_split[valid_index]
            # valid_clinics.append(valid_clinics_split)
            valid_ids_split = ids_split[valid_index]
            valid_ids.append(valid_ids_split)
            valid_censoreds_split = censoreds_split[valid_index]
            valid_censoreds.append(valid_censoreds_split)
            valid_text_split = text_split[valid_index]
            valid_text.append(valid_text_split)

            valid_discrete_clinicl_split = discrete_clinicl_split[valid_index]
            valid_dis_clinics.append(valid_discrete_clinicl_split)

            valid_continuous_clinicl_split = continuous_clinicl_split[valid_index]
            valid_con_clinics.append(valid_continuous_clinicl_split)

    Scaler = StandardScaler()
    # OHencoder = OneHotEncoder(categories='auto', sparse_output=False)


    train_image_files = np.concatenate(train_image_files, axis=0)
    train_image_cir_files = np.concatenate(train_image_cir_files, axis=0)
    train_feat_cir = np.concatenate(train_feat_cir, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_con_clinics = np.concatenate(train_con_clinics, axis=0)
    train_dis_clinics = np.concatenate(train_dis_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    train_text = np.concatenate(train_text, axis=0)
    train_con_clinics_ts = Scaler.fit_transform(train_con_clinics)
    # train_dis_clinics_ts = OHencoder.fit_transform(train_dis_clinics)
    train_clinics = np.concatenate([train_con_clinics_ts, train_dis_clinics], axis=-1)

    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_image_cir_files = np.concatenate(valid_image_cir_files, axis=0)
    valid_feat_cir = np.concatenate(valid_feat_cir, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    valid_con_clinics = np.concatenate(valid_con_clinics, axis=0)
    valid_dis_clinics = np.concatenate(valid_dis_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    valid_text = np.concatenate(valid_text, axis=0)
    valid_con_clinics_ts = Scaler.transform(valid_con_clinics)
    # valid_dis_clinics_ts = OHencoder.fit_transform(valid_dis_clinics)
    valid_clinics = np.concatenate([valid_con_clinics_ts, valid_dis_clinics], axis=-1)

    test_image_files = np.concatenate(test_image_files, axis=0)
    test_image_cir_files = np.concatenate(test_image_cir_files, axis=0)
    test_feat_cir = np.concatenate(test_feat_cir, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_con_clinics = np.concatenate(test_con_clinics, axis=0)
    test_dis_clinics = np.concatenate(test_dis_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    test_text = np.concatenate(test_text, axis=0)
    test_con_clinics_ts = Scaler.transform(test_con_clinics)
    # test_dis_clinics_ts = OHencoder.fit_transform(test_dis_clinics)
    test_clinics = np.concatenate([test_con_clinics_ts, test_dis_clinics], axis=-1)

    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)

    train_text = train_text.tolist()
    valid_text = valid_text.tolist()
    test_text = test_text.tolist()

    train_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"], ),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        RandRotated(keys=["image", "liver", "tumor", "cir_liver"], range_x=15, prob=0.5, keep_size=True),
        RandFlipd(keys=["image", "liver", "tumor", "cir_liver"], spatial_axis=0, prob=0.2),
        RandZoomd(keys=["image", "liver", "tumor", "cir_liver"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    train_dataset = LiverDataset_disentangle_cir_feat(train_image_files, train_image_cir_files, train_feat_cir,
                                                      train_labels, train_censoreds, train_durations_discrete,
                                                      train_events, train_ids, train_text, train_transforms)
    valid_dataset = LiverDataset_disentangle_cir_feat(valid_image_files, valid_image_cir_files, valid_feat_cir,
                                                      valid_labels, valid_censoreds, valid_durations_discrete,
                                                      valid_events, valid_ids, valid_text,  val_transforms)
    test_dataset = LiverDataset_disentangle_cir_feat(test_image_files, test_image_cir_files, test_feat_cir,
                                                     test_labels, test_censoreds, test_durations_discrete,
                                                     test_events, test_ids, test_text, val_transforms)

    # train_infos = np.concatenate([train_ids[:, None], train_image_files[:, None], train_labels[:, None],
    #                          train_censoreds[:, None], train_clinics, np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], valid_image_files[:, None], valid_labels[:, None],
    #                               valid_censoreds[:, None], valid_clinics, np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], test_image_files[:, None], test_labels[:, None],
    #                               test_censoreds[:, None], test_clinics, np.array(test_text)[:, None]], axis=1)
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_infos.csv', train_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_infos.csv', valid_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_infos.csv', test_infos, fmt='%s',
    #            delimiter=',')
    # train_infos = np.concatenate([train_ids[:, None], np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], np.array(test_text)[:, None]], axis=1)
    # train_infos = pd.DataFrame(train_infos, columns=['ids', 'text'])
    # valid_infos = pd.DataFrame(valid_infos, columns=['ids', 'text'])
    # test_infos = pd.DataFrame(test_infos, columns=['ids', 'text'])
    # train_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_text.xlsx', index=False)
    # valid_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_text.xlsx', index=False)
    # test_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_text.xlsx', index=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=2)
    x_image_files, x_cir_feat, x_labels, x_censoreds, x_durations_discrete, \
        x_events, x_ids, x_text = next(iter(train_loader))
    # for i, batch_data in enumerate(train_loader):
    #     s = 0
    # train_sample = next(iter(train_loader))
    return train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader

def create_dataset_ext_week_persudo_pair_addcirf_fixid2(matched_feature_file1, matched_feature_file2, matched_ids_file1, matched_ids_file2,
                                matched_censor_file1, matched_censor_file2, matched_surv_file1, matched_surv_file2,
                                         root_dir, fold, num_durations=6, batch_size=6, val_batch_size=4, seed=1998):
    features_cp0 = np.load(matched_feature_file1)
    features_cp1 = np.load(matched_feature_file2)[:, 0, ...]
    ids_cp0 = np.load(matched_ids_file1)
    ids_cp1 = np.load(matched_ids_file2)[:, 0, ...]
    censors_cp0 = np.load(matched_censor_file1)
    censors_cp1 = np.load(matched_censor_file2)
    survs_cp0 = np.load(matched_surv_file1)
    survs_cp1 = np.load(matched_surv_file2)

    match_ids_sum = np.concatenate([ids_cp0, ids_cp1])
    match_features_sum = np.concatenate([features_cp0, features_cp1])
    match_censors_sum = np.concatenate([censors_cp0, censors_cp1])
    match_survs_sum = np.concatenate([survs_cp0, survs_cp1])
    # print_config()
    data_dir1 = root_dir
    hospital_names = (
        sorted([x for x in os.listdir(data_dir1) if os.path.isdir(os.path.join(data_dir1, x))]))
    image_files = [[os.path.join(data_dir1, class_name, x)
                     for x in os.listdir(os.path.join(data_dir1, class_name))]
                    for class_name in hospital_names]

    image_files_need = []
    image_cir_files_need = []
    feat_cir_need = []
    info = pd.read_excel(r"/home/wangtao/project/Cirros_HCC/data/table/PFS_ZYJ_20240319_PD_CP_text.xlsx")
    ids = info['序号']
    hospitals = info['病例来源']

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']
    clinical_tile = ['Age', 'Max_d', '']
    continuous_clinicl = info.iloc[:, [5, 10]]
    discrete_clinicl = info.iloc[:, [6, 7, 8, 9, 11, 12, 23]]
    # discrete_clinicl = info.iloc[:, [7, 8, 9]]
    continuous_clinicl = np.asarray(continuous_clinicl)
    discrete_clinicl = np.asarray(discrete_clinicl)
    discrete_clinicl[np.where(np.isnan(discrete_clinicl)==True)] = 1
    OHencoder = OneHotEncoder(categories='auto', sparse_output=False)
    discrete_clinicl = OHencoder.fit_transform(discrete_clinicl)

    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        continuous_clinicl_hospital = []
        discrete_clinicl_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        image_cir_files_need_hospital = []
        feat_cir_need_hospital = []
        for image_file in image_files_hospital:
            ID = image_file.split('/')[-1].split('.')[0]
            center_now = image_file.split('/')[-2]
            pos = np.where(ids.astype(str) == ID)
            pos_match = np.where(match_ids_sum[:, -1] == center_now + '_' + ID)
            if len(pos[0]) != 0:
                cp_now = cp_labels[pos][0]
                if cp_now != 3 and np.isnan(cp_now) == False:
                    id_cir_now = match_ids_sum[pos_match[0], 0][0]
                    feat_cir_now = match_features_sum[pos_match[0], 0][0][None, :]
                    image_cir_file = os.path.join(r'/home/wangtao/project/Survival/data_cir/preprocess/Liver/data',
                                                  id_cir_now.split('_')[0], id_cir_now + '.nii.gz')
                    surv_label = surv_labels[pos][0]
                    if_censored = if_censoreds[pos][0]
                    text_now = gpt_text[pos][0]

                    continuous_clinicl_now = continuous_clinicl[pos][0]
                    discrete_clinicl_now = discrete_clinicl[pos][0]
                    image_files_need_hospital.append(image_file)
                    ids_hospital.append(ID)
                    text_hospital.append(text_now)
                    labels_hospital.append(surv_label)
                    censored_hospital.append(if_censored)
                    image_cir_files_need_hospital.append(image_cir_file)
                    feat_cir_need_hospital.append(feat_cir_now)
                    continuous_clinicl_hospital.append(continuous_clinicl_now)
                    discrete_clinicl_hospital.append(discrete_clinicl_now)


        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        image_cir_files_need_hospital = np.array(image_cir_files_need_hospital)
        feat_cir_need_hospital = np.concatenate(feat_cir_need_hospital)
        continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        image_cir_files_need.append(image_cir_files_need_hospital)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)
        feat_cir_need.append(feat_cir_need_hospital)
        continuous_clinicl_sum.append(continuous_clinicl_hospital)
        discrete_clinicl_sum.append(discrete_clinicl_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    train_val_image_files = []
    train_val_image_cir_files = []
    train_val_feat_cir = []
    train_val_labels = []
    train_val_con_clinics = []
    train_val_dis_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []

    train_image_files = []
    train_image_cir_files = []
    train_feat_cir = []
    train_labels = []
    train_con_clinics = []
    train_dis_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []

    valid_image_files = []
    valid_image_cir_files = []
    valid_feat_cir = []
    valid_labels = []
    valid_con_clinics = []
    valid_dis_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []

    test_image_files = []
    test_image_cir_files = []
    test_feat_cir = []
    test_labels = []
    test_con_clinics = []
    test_dis_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        image_cir_files_split = np.array(image_cir_files_need[hospital])
        feat_cir_split = feat_cir_need[hospital]
        labels_split = labels[hospital]
        # continuous_clinics_split = continuous_clinicl_sum[hospital]
        # discrete_clinics_split = discrete_clinicl_sum[hospital]
        # clinics_split = np.concatenate([continuous_clinics_split, discrete_clinics_split], axis=-1)
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        continuous_clinicl_split = continuous_clinicl_sum[hospital]
        discrete_clinicl_split = discrete_clinicl_sum[hospital]
        # if hospital_name_split in ['Center2', 'Center3', 'Center4', 'Center8', 'Center9', 'Center5', 'Center11']:  # 0420_1
        if hospital_name_split not in ['Center5', 'Center9']:
            train_val_image_files.append(image_files_split)
            train_val_image_cir_files.append(image_cir_files_split)
            train_val_labels.append(labels_split)
            # train_val_clinics.append(clinics_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
            train_val_feat_cir.append(feat_cir_split)
            train_val_dis_clinics.append(discrete_clinicl_split)
            train_val_con_clinics.append(continuous_clinicl_split)
        else:
            test_image_files.append(image_files_split)
            test_image_cir_files.append(image_cir_files_split)
            test_labels.append(labels_split)
            # test_clinics.append(clinics_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)
            test_feat_cir.append(feat_cir_split)
            test_dis_clinics.append(discrete_clinicl_split)
            test_con_clinics.append(continuous_clinicl_split)

    for hospital in range(len(train_val_image_files)):
        image_files_split = np.array(train_val_image_files[hospital])
        image_cir_files_split = np.array(train_val_image_cir_files[hospital])
        labels_split = train_val_labels[hospital]
        feat_cir_split = train_val_feat_cir[hospital]
        # clinics_split = train_val_clinics[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        censoreds_split = train_val_censoreds[hospital]
        continuous_clinicl_split = train_val_con_clinics[hospital]
        discrete_clinicl_split = train_val_dis_clinics[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_image_cir_files.append(image_cir_files_split)
            train_labels.append(labels_split)
            train_dis_clinics.append(discrete_clinicl_split)
            train_con_clinics.append(continuous_clinicl_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
            train_feat_cir.append(feat_cir_split)
        else:
            # for train_index, valid_index in kf.split(image_files_split, censoreds_split):
            #     _fold = _fold + 1
            #     if _fold == fold:
            train_index = np.loadtxt(r'/home/wangtao/project/Survival/split_ids/train_index{}_fold{}_nt59.csv'.format(hospital, fold),
                                     delimiter=',', dtype=str)
            train_index = train_index.astype(int)
            valid_index = np.loadtxt(r'/home/wangtao/project/Survival/split_ids/valid_index{}_fold{}_nt59.csv'.format(hospital, fold),
                                     delimiter=',', dtype=str)
            valid_index = valid_index.astype(int)
            train_image_files_split = image_files_split[train_index]
            train_image_cir_files_split = image_cir_files_split[train_index]
            train_image_files.append(train_image_files_split)
            train_image_cir_files.append(train_image_cir_files_split)
            train_feat_cir_split = feat_cir_split[train_index]
            train_feat_cir.append(train_feat_cir_split)
            train_labels_split = labels_split[train_index]
            train_labels.append(train_labels_split)

            train_ids_split = ids_split[train_index]
            train_ids.append(train_ids_split)
            train_censoreds_split = censoreds_split[train_index]
            train_censoreds.append(train_censoreds_split)
            train_text_split = text_split[train_index]
            train_text.append(train_text_split)

            train_discrete_clinicl_split = discrete_clinicl_split[train_index]
            train_dis_clinics.append(train_discrete_clinicl_split)

            train_continuous_clinicl_split = continuous_clinicl_split[train_index]
            train_con_clinics.append(train_continuous_clinicl_split)

            valid_image_files_split = image_files_split[valid_index]
            valid_image_cir_files_split = image_cir_files_split[valid_index]
            valid_image_files.append(valid_image_files_split)
            valid_image_cir_files.append(valid_image_cir_files_split)
            valid_feat_cir_split = feat_cir_split[valid_index]
            valid_feat_cir.append(valid_feat_cir_split)
            valid_labels_split = labels_split[valid_index]
            valid_labels.append(valid_labels_split)
            # valid_clinics_split = clinics_split[valid_index]
            # valid_clinics.append(valid_clinics_split)
            valid_ids_split = ids_split[valid_index]
            valid_ids.append(valid_ids_split)
            valid_censoreds_split = censoreds_split[valid_index]
            valid_censoreds.append(valid_censoreds_split)
            valid_text_split = text_split[valid_index]
            valid_text.append(valid_text_split)

            valid_discrete_clinicl_split = discrete_clinicl_split[valid_index]
            valid_dis_clinics.append(valid_discrete_clinicl_split)

            valid_continuous_clinicl_split = continuous_clinicl_split[valid_index]
            valid_con_clinics.append(valid_continuous_clinicl_split)

    Scaler = StandardScaler()
    # OHencoder = OneHotEncoder(categories='auto', sparse_output=False)


    train_image_files = np.concatenate(train_image_files, axis=0)
    train_image_cir_files = np.concatenate(train_image_cir_files, axis=0)
    train_feat_cir = np.concatenate(train_feat_cir, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_con_clinics = np.concatenate(train_con_clinics, axis=0)
    train_dis_clinics = np.concatenate(train_dis_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    train_text = np.concatenate(train_text, axis=0)
    train_con_clinics_ts = Scaler.fit_transform(train_con_clinics)
    # train_dis_clinics_ts = OHencoder.fit_transform(train_dis_clinics)
    train_clinics = np.concatenate([train_con_clinics_ts, train_dis_clinics], axis=-1)

    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_image_cir_files = np.concatenate(valid_image_cir_files, axis=0)
    valid_feat_cir = np.concatenate(valid_feat_cir, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    valid_con_clinics = np.concatenate(valid_con_clinics, axis=0)
    valid_dis_clinics = np.concatenate(valid_dis_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    valid_text = np.concatenate(valid_text, axis=0)
    valid_con_clinics_ts = Scaler.transform(valid_con_clinics)
    # valid_dis_clinics_ts = OHencoder.fit_transform(valid_dis_clinics)
    valid_clinics = np.concatenate([valid_con_clinics_ts, valid_dis_clinics], axis=-1)

    test_image_files = np.concatenate(test_image_files, axis=0)
    test_image_cir_files = np.concatenate(test_image_cir_files, axis=0)
    test_feat_cir = np.concatenate(test_feat_cir, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_con_clinics = np.concatenate(test_con_clinics, axis=0)
    test_dis_clinics = np.concatenate(test_dis_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    test_text = np.concatenate(test_text, axis=0)
    test_con_clinics_ts = Scaler.transform(test_con_clinics)
    # test_dis_clinics_ts = OHencoder.fit_transform(test_dis_clinics)
    test_clinics = np.concatenate([test_con_clinics_ts, test_dis_clinics], axis=-1)

    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)

    train_text = train_text.tolist()
    valid_text = valid_text.tolist()
    test_text = test_text.tolist()

    train_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"], ),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        RandRotated(keys=["image", "liver", "tumor", "cir_liver"], range_x=15, prob=0.5, keep_size=True),
        RandFlipd(keys=["image", "liver", "tumor", "cir_liver"], spatial_axis=0, prob=0.2),
        RandZoomd(keys=["image", "liver", "tumor", "cir_liver"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    train_dataset = LiverDataset_disentangle_cir_feat_cli(train_image_files, train_image_cir_files, train_clinics, train_feat_cir,
                                                      train_labels, train_censoreds, train_durations_discrete,
                                                      train_events, train_ids, train_text, train_transforms)
    valid_dataset = LiverDataset_disentangle_cir_feat_cli(valid_image_files, valid_image_cir_files, valid_clinics, valid_feat_cir,
                                                      valid_labels, valid_censoreds, valid_durations_discrete,
                                                      valid_events, valid_ids, valid_text,  val_transforms)
    test_dataset = LiverDataset_disentangle_cir_feat_cli(test_image_files, test_image_cir_files, test_clinics, test_feat_cir,
                                                     test_labels, test_censoreds, test_durations_discrete,
                                                     test_events, test_ids, test_text, val_transforms)

    # train_infos = np.concatenate([train_ids[:, None], train_image_files[:, None], train_labels[:, None],
    #                          train_censoreds[:, None], train_clinics, np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], valid_image_files[:, None], valid_labels[:, None],
    #                               valid_censoreds[:, None], valid_clinics, np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], test_image_files[:, None], test_labels[:, None],
    #                               test_censoreds[:, None], test_clinics, np.array(test_text)[:, None]], axis=1)
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_infos.csv', train_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_infos.csv', valid_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_infos.csv', test_infos, fmt='%s',
    #            delimiter=',')
    # train_infos = np.concatenate([train_ids[:, None], np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], np.array(test_text)[:, None]], axis=1)
    # train_infos = pd.DataFrame(train_infos, columns=['ids', 'text'])
    # valid_infos = pd.DataFrame(valid_infos, columns=['ids', 'text'])
    # test_infos = pd.DataFrame(test_infos, columns=['ids', 'text'])
    # train_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_text.xlsx', index=False)
    # valid_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_text.xlsx', index=False)
    # test_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_text.xlsx', index=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=2)
    x_image_files, x_clinics, x_cir_feat, x_labels, x_censoreds, x_durations_discrete, \
        x_events, x_ids, x_text = next(iter(train_loader))
    # for i, batch_data in enumerate(train_loader):
    #     s = 0
    # train_sample = next(iter(train_loader))
    return train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader

def create_dataset_ext_week_persudo_pair_addcirf_fixid2r(matched_feature_file1, matched_feature_file2, matched_ids_file1, matched_ids_file2,
                                matched_censor_file1, matched_censor_file2, matched_surv_file1, matched_surv_file2,
                                         root_dir, fold, num_durations=6, batch_size=6, val_batch_size=4, seed=1998):
    set_seeds(seed)
    features_cp0 = np.load(matched_feature_file1)
    features_cp1 = np.load(matched_feature_file2)[:, 0, ...]
    ids_cp0 = np.load(matched_ids_file1)
    ids_cp1 = np.load(matched_ids_file2)[:, 0, ...]
    censors_cp0 = np.load(matched_censor_file1)
    censors_cp1 = np.load(matched_censor_file2)
    survs_cp0 = np.load(matched_surv_file1)
    survs_cp1 = np.load(matched_surv_file2)

    match_ids_sum = np.concatenate([ids_cp0, ids_cp1])
    match_features_sum = np.concatenate([features_cp0, features_cp1])
    match_censors_sum = np.concatenate([censors_cp0, censors_cp1])
    match_survs_sum = np.concatenate([survs_cp0, survs_cp1])
    # print_config()
    data_dir1 = root_dir
    hospital_names = (
        sorted([x for x in os.listdir(data_dir1) if os.path.isdir(os.path.join(data_dir1, x))]))
    image_files = [[os.path.join(data_dir1, class_name, x)
                     for x in os.listdir(os.path.join(data_dir1, class_name))]
                    for class_name in hospital_names]

    image_files_need = []
    image_cir_files_need = []
    feat_cir_need = []
    info = pd.read_excel(r"/home/wangtao/project/Cirros_HCC/data/table/PFS_ZYJ_20240319_PD_CP_text.xlsx")
    ids = info['序号']
    hospitals = info['病例来源']

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']

    continuous_clinicl = info.iloc[:, [5, 10]]
    discrete_clinicl = info.iloc[:, [6, 7, 8, 9, 11, 12, 23]]
    # discrete_clinicl = info.iloc[:, [7, 8, 9]]
    continuous_clinicl = np.asarray(continuous_clinicl)
    discrete_clinicl = np.asarray(discrete_clinicl)
    discrete_clinicl[np.where(np.isnan(discrete_clinicl)==True)] = 1
    OHencoder = OneHotEncoder(categories='auto', sparse_output=False)
    discrete_clinicl = OHencoder.fit_transform(discrete_clinicl)
    joblib.dump(OHencoder, 'onehot.pkl')
    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        continuous_clinicl_hospital = []
        discrete_clinicl_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        image_cir_files_need_hospital = []
        feat_cir_need_hospital = []
        for image_file in image_files_hospital:
            ID = image_file.split('/')[-1].split('.')[0]
            center_now = image_file.split('/')[-2]
            pos = np.where(ids.astype(str) == ID)
            pos_match = np.where(match_ids_sum[:, -1] == center_now + '_' + ID)
            if len(pos[0]) != 0:
                cp_now = cp_labels[pos][0]
                if cp_now != 3 and np.isnan(cp_now) == False:
                    id_cir_now = match_ids_sum[pos_match[0], 0][0]
                    feat_cir_now = match_features_sum[pos_match[0], 0][0][None, :]
                    image_cir_file = os.path.join(r'/home/wangtao/project/Survival/data_cir/preprocess/Liver/data',
                                                  id_cir_now.split('_')[0], id_cir_now + '.nii.gz')
                    surv_label = surv_labels[pos][0]
                    if_censored = if_censoreds[pos][0]
                    text_now = gpt_text[pos][0]

                    continuous_clinicl_now = continuous_clinicl[pos][0]
                    discrete_clinicl_now = discrete_clinicl[pos][0]
                    image_files_need_hospital.append(image_file)
                    ids_hospital.append(ID)
                    text_hospital.append(text_now)
                    labels_hospital.append(surv_label)
                    censored_hospital.append(if_censored)
                    image_cir_files_need_hospital.append(image_cir_file)
                    feat_cir_need_hospital.append(feat_cir_now)
                    continuous_clinicl_hospital.append(continuous_clinicl_now)
                    discrete_clinicl_hospital.append(discrete_clinicl_now)


        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        image_cir_files_need_hospital = np.array(image_cir_files_need_hospital)
        feat_cir_need_hospital = np.concatenate(feat_cir_need_hospital)
        continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        image_cir_files_need.append(image_cir_files_need_hospital)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)
        feat_cir_need.append(feat_cir_need_hospital)
        continuous_clinicl_sum.append(continuous_clinicl_hospital)
        discrete_clinicl_sum.append(discrete_clinicl_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1998)

    train_val_image_files = []
    train_val_image_cir_files = []
    train_val_feat_cir = []
    train_val_labels = []
    train_val_con_clinics = []
    train_val_dis_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []

    train_image_files = []
    train_image_cir_files = []
    train_feat_cir = []
    train_labels = []
    train_con_clinics = []
    train_dis_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []

    valid_image_files = []
    valid_image_cir_files = []
    valid_feat_cir = []
    valid_labels = []
    valid_con_clinics = []
    valid_dis_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []

    test_image_files = []
    test_image_cir_files = []
    test_feat_cir = []
    test_labels = []
    test_con_clinics = []
    test_dis_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        image_cir_files_split = np.array(image_cir_files_need[hospital])
        feat_cir_split = feat_cir_need[hospital]
        labels_split = labels[hospital]
        # continuous_clinics_split = continuous_clinicl_sum[hospital]
        # discrete_clinics_split = discrete_clinicl_sum[hospital]
        # clinics_split = np.concatenate([continuous_clinics_split, discrete_clinics_split], axis=-1)
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        continuous_clinicl_split = continuous_clinicl_sum[hospital]
        discrete_clinicl_split = discrete_clinicl_sum[hospital]
        # if hospital_name_split in ['Center2', 'Center3', 'Center4', 'Center8', 'Center9', 'Center5', 'Center11']:  # 0420_1
        if hospital_name_split not in ['Center5', 'Center9']:
            train_val_image_files.append(image_files_split)
            train_val_image_cir_files.append(image_cir_files_split)
            train_val_labels.append(labels_split)
            # train_val_clinics.append(clinics_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
            train_val_feat_cir.append(feat_cir_split)
            train_val_dis_clinics.append(discrete_clinicl_split)
            train_val_con_clinics.append(continuous_clinicl_split)
        else:
            test_image_files.append(image_files_split)
            test_image_cir_files.append(image_cir_files_split)
            test_labels.append(labels_split)
            # test_clinics.append(clinics_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)
            test_feat_cir.append(feat_cir_split)
            test_dis_clinics.append(discrete_clinicl_split)
            test_con_clinics.append(continuous_clinicl_split)

    for hospital in range(len(train_val_image_files)):
        image_files_split = np.array(train_val_image_files[hospital])
        image_cir_files_split = np.array(train_val_image_cir_files[hospital])
        labels_split = train_val_labels[hospital]
        feat_cir_split = train_val_feat_cir[hospital]
        # clinics_split = train_val_clinics[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        censoreds_split = train_val_censoreds[hospital]
        continuous_clinicl_split = train_val_con_clinics[hospital]
        discrete_clinicl_split = train_val_dis_clinics[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_image_cir_files.append(image_cir_files_split)
            train_labels.append(labels_split)
            train_dis_clinics.append(discrete_clinicl_split)
            train_con_clinics.append(continuous_clinicl_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
            train_feat_cir.append(feat_cir_split)
        else:
            # for train_index, valid_index in kf.split(image_files_split, censoreds_split):
            #     _fold = _fold + 1
            #     if _fold == fold:
            train_index = np.loadtxt(r'/home/wangtao/project/Survival/split_ids/train_index{}_fold{}_nt59.csv'.format(hospital, fold),
                                     delimiter=',', dtype=str)
            train_index = train_index.astype(int)
            valid_index = np.loadtxt(r'/home/wangtao/project/Survival/split_ids/valid_index{}_fold{}_nt59.csv'.format(hospital, fold),
                                     delimiter=',', dtype=str)
            valid_index = valid_index.astype(int)
            train_image_files_split = image_files_split[train_index]
            train_image_cir_files_split = image_cir_files_split[train_index]
            train_image_files.append(train_image_files_split)
            train_image_cir_files.append(train_image_cir_files_split)
            train_feat_cir_split = feat_cir_split[train_index]
            train_feat_cir.append(train_feat_cir_split)
            train_labels_split = labels_split[train_index]
            train_labels.append(train_labels_split)

            train_ids_split = ids_split[train_index]
            train_ids.append(train_ids_split)
            train_censoreds_split = censoreds_split[train_index]
            train_censoreds.append(train_censoreds_split)
            train_text_split = text_split[train_index]
            train_text.append(train_text_split)

            train_discrete_clinicl_split = discrete_clinicl_split[train_index]
            train_dis_clinics.append(train_discrete_clinicl_split)

            train_continuous_clinicl_split = continuous_clinicl_split[train_index]
            train_con_clinics.append(train_continuous_clinicl_split)

            valid_image_files_split = image_files_split[valid_index]
            valid_image_cir_files_split = image_cir_files_split[valid_index]
            valid_image_files.append(valid_image_files_split)
            valid_image_cir_files.append(valid_image_cir_files_split)
            valid_feat_cir_split = feat_cir_split[valid_index]
            valid_feat_cir.append(valid_feat_cir_split)
            valid_labels_split = labels_split[valid_index]
            valid_labels.append(valid_labels_split)
            # valid_clinics_split = clinics_split[valid_index]
            # valid_clinics.append(valid_clinics_split)
            valid_ids_split = ids_split[valid_index]
            valid_ids.append(valid_ids_split)
            valid_censoreds_split = censoreds_split[valid_index]
            valid_censoreds.append(valid_censoreds_split)
            valid_text_split = text_split[valid_index]
            valid_text.append(valid_text_split)

            valid_discrete_clinicl_split = discrete_clinicl_split[valid_index]
            valid_dis_clinics.append(valid_discrete_clinicl_split)

            valid_continuous_clinicl_split = continuous_clinicl_split[valid_index]
            valid_con_clinics.append(valid_continuous_clinicl_split)

    Scaler = StandardScaler()
    # OHencoder = OneHotEncoder(categories='auto', sparse_output=False)


    train_image_files = np.concatenate(train_image_files, axis=0)
    train_image_cir_files = np.concatenate(train_image_cir_files, axis=0)
    train_feat_cir = np.concatenate(train_feat_cir, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_con_clinics = np.concatenate(train_con_clinics, axis=0)
    train_dis_clinics = np.concatenate(train_dis_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    train_text = np.concatenate(train_text, axis=0)
    train_con_clinics_ts = Scaler.fit_transform(train_con_clinics)
    # train_dis_clinics_ts = OHencoder.fit_transform(train_dis_clinics)
    train_clinics = np.concatenate([train_con_clinics_ts, train_dis_clinics], axis=-1)
    # ### train_image_files, train_image_cir_files, train_clinics, train_feat_cir,
    #                                                   train_labels, train_censoreds, train_durations_discrete,
    #                                                   train_events, train_ids, train_text, train_transforms
    train_indices = list(range(len(train_image_files)))
    train_selected_indices = random.sample(train_indices, len(train_image_files) - 40)
    train_image_files = train_image_files[train_selected_indices]
    train_image_cir_files = train_image_cir_files[train_selected_indices]
    train_feat_cir = train_feat_cir[train_selected_indices]
    train_labels = train_labels[train_selected_indices]
    train_ids = train_ids[train_selected_indices]
    train_censoreds = train_censoreds[train_selected_indices]
    train_text = train_text[train_selected_indices]
    train_clinics = train_clinics[train_selected_indices]
    joblib.dump(Scaler, 'scaler.pkl')

    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_image_cir_files = np.concatenate(valid_image_cir_files, axis=0)
    valid_feat_cir = np.concatenate(valid_feat_cir, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    valid_con_clinics = np.concatenate(valid_con_clinics, axis=0)
    valid_dis_clinics = np.concatenate(valid_dis_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    valid_text = np.concatenate(valid_text, axis=0)
    valid_con_clinics_ts = Scaler.transform(valid_con_clinics)
    # valid_dis_clinics_ts = OHencoder.fit_transform(valid_dis_clinics)
    valid_clinics = np.concatenate([valid_con_clinics_ts, valid_dis_clinics], axis=-1)
    
    valid_indices = list(range(len(valid_image_files)))
    valid_selected_indices = random.sample(valid_indices, len(valid_image_files) - 20)
    valid_image_files = valid_image_files[valid_selected_indices]
    valid_image_cir_files = valid_image_cir_files[valid_selected_indices]
    valid_feat_cir = valid_feat_cir[valid_selected_indices]
    valid_labels = valid_labels[valid_selected_indices]
    valid_ids = valid_ids[valid_selected_indices]
    valid_censoreds = valid_censoreds[valid_selected_indices]
    valid_text = valid_text[valid_selected_indices]
    valid_clinics = valid_clinics[valid_selected_indices]

    test_image_files = np.concatenate(test_image_files, axis=0)
    test_image_cir_files = np.concatenate(test_image_cir_files, axis=0)
    test_feat_cir = np.concatenate(test_feat_cir, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_con_clinics = np.concatenate(test_con_clinics, axis=0)
    test_dis_clinics = np.concatenate(test_dis_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    test_text = np.concatenate(test_text, axis=0)
    test_con_clinics_ts = Scaler.transform(test_con_clinics)
    # test_dis_clinics_ts = OHencoder.fit_transform(test_dis_clinics)
    test_clinics = np.concatenate([test_con_clinics_ts, test_dis_clinics], axis=-1)

    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)
    joblib.dump(labtrans, 'labtrans.pkl')

    train_text = train_text.tolist()
    valid_text = valid_text.tolist()
    test_text = test_text.tolist()

    train_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"], ),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        RandRotated(keys=["image", "liver", "tumor", "cir_liver"], range_x=15, prob=0.5, keep_size=True),
        RandFlipd(keys=["image", "liver", "tumor", "cir_liver"], spatial_axis=0, prob=0.2),
        RandZoomd(keys=["image", "liver", "tumor", "cir_liver"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    train_dataset = LiverDataset_disentangle_cir_feat_cli(train_image_files, train_image_cir_files, train_clinics, train_feat_cir,
                                                      train_labels, train_censoreds, train_durations_discrete,
                                                      train_events, train_ids, train_text, train_transforms)
    valid_dataset = LiverDataset_disentangle_cir_feat_cli(valid_image_files, valid_image_cir_files, valid_clinics, valid_feat_cir,
                                                      valid_labels, valid_censoreds, valid_durations_discrete,
                                                      valid_events, valid_ids, valid_text,  val_transforms)
    test_dataset = LiverDataset_disentangle_cir_feat_cli(test_image_files, test_image_cir_files, test_clinics, test_feat_cir,
                                                     test_labels, test_censoreds, test_durations_discrete,
                                                     test_events, test_ids, test_text, val_transforms)

    # train_infos = np.concatenate([train_ids[:, None], train_image_files[:, None], train_labels[:, None],
    #                          train_censoreds[:, None], train_clinics, np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], valid_image_files[:, None], valid_labels[:, None],
    #                               valid_censoreds[:, None], valid_clinics, np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], test_image_files[:, None], test_labels[:, None],
    #                               test_censoreds[:, None], test_clinics, np.array(test_text)[:, None]], axis=1)
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_infos.csv', train_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_infos.csv', valid_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_infos.csv', test_infos, fmt='%s',
    #            delimiter=',')
    # train_infos = np.concatenate([train_ids[:, None], np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], np.array(test_text)[:, None]], axis=1)
    # train_infos = pd.DataFrame(train_infos, columns=['ids', 'text'])
    # valid_infos = pd.DataFrame(valid_infos, columns=['ids', 'text'])
    # test_infos = pd.DataFrame(test_infos, columns=['ids', 'text'])
    # train_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_text.xlsx', index=False)
    # valid_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_text.xlsx', index=False)
    # test_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_text.xlsx', index=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=2)
    x_image_files, x_clinics, x_cir_feat, x_labels, x_censoreds, x_durations_discrete, \
        x_events, x_ids, x_text = next(iter(train_loader))
    # for i, batch_data in enumerate(train_loader):
    #     s = 0
    # train_sample = next(iter(train_loader))
    return train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader

def all_files_path(rootDir):
    filepaths = []
    for root, dirs, files in os.walk(rootDir):     # 分别代表根目录、文件夹、文件
        for file in files:                         # 遍历文件
            file_path = os.path.join(root, file)   # 获取文件绝对路径
            filepaths.append(file_path)            # 将文件路径添加进列表
        for dir in dirs:                           # 遍历目录下的子目录
            dir_path = os.path.join(root, dir)     # 获取子目录路径
            all_files_path(dir_path)
    return filepaths

def create_dataset_ext_week_persudo_pair_addcirf_fixid2r_LIHC(matched_feature_file1, matched_feature_file2,
                                                         matched_ids_file1, matched_ids_file2,
                                                         matched_censor_file1, matched_censor_file2, matched_surv_file1,
                                                         matched_surv_file2,
                                                         root_dir, fold, num_durations=6, batch_size=6,
                                                         val_batch_size=4, seed=1998):
    set_seeds(seed)
    features_cp0 = np.load(matched_feature_file1)
    features_cp1 = np.load(matched_feature_file2)[:, 0, ...]
    ids_cp0 = np.load(matched_ids_file1)
    ids_cp1 = np.load(matched_ids_file2)[:, 0, ...]
    censors_cp0 = np.load(matched_censor_file1)
    censors_cp1 = np.load(matched_censor_file2)
    survs_cp0 = np.load(matched_surv_file1)
    survs_cp1 = np.load(matched_surv_file2)

    match_ids_sum = np.concatenate([ids_cp0, ids_cp1])
    match_features_sum = np.concatenate([features_cp0, features_cp1])
    match_censors_sum = np.concatenate([censors_cp0, censors_cp1])
    match_survs_sum = np.concatenate([survs_cp0, survs_cp1])
    # print_config()
    # data_dir1 = root_dir

    image_files = all_files_path(root_dir)

    image_files_need = []
    image_cir_files_need = []
    feat_cir_need = []
    info = pd.read_excel(r"/home/wangtao/project/Survival/data_LIHC/TCGA_LIHC_PD_text.xlsx")
    ids = info['患者姓名']
    hospitals = info['病例来源']

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']

    continuous_clinicl = info.iloc[:, [5, 10]]
    discrete_clinicl = info.iloc[:, [6, 7, 8, 9, 11, 12, 23]]
    # discrete_clinicl = info.iloc[:, [7, 8, 9]]
    continuous_clinicl = np.asarray(continuous_clinicl)

    discrete_clinicl = np.asarray(discrete_clinicl)
    discrete_clinicl[np.where(np.isnan(discrete_clinicl) == True)] = 1
    # OHencoder = OneHotEncoder(categories='auto', sparse_output=False)
    OHencoder = joblib.load('onehot.pkl')
    discrete_clinicl = OHencoder.transform(discrete_clinicl)

    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    # for image_files_hospital in image_files:
    labels_hospital = []
    image_files_need_hospital = []
    continuous_clinicl_hospital = []
    discrete_clinicl_hospital = []
    ids_hospital = []
    censored_hospital = []
    text_hospital = []
    image_cir_files_need_hospital = []
    feat_cir_need_hospital = []
    for image_file in image_files:
        ID = image_file.split('/')[-1].split('.')[0]
        center_now = image_file.split('/')[-2]
        pos = np.where(ids.astype(str) == ID)
        pos_match = np.where(match_ids_sum[:, -1] == center_now + '_' + ID)
        if len(pos[0]) != 0 and len(pos_match[0]) != 0:
            cp_now = cp_labels[pos][0]
            if cp_now != 3 and np.isnan(cp_now) == False:
                id_cir_now = match_ids_sum[pos_match[0], 0][0]
                feat_cir_now = match_features_sum[pos_match[0], 0][0][None, :]
                image_cir_file = os.path.join(r'/home/wangtao/project/Survival/data_cir/preprocess/Liver/data',
                                              id_cir_now.split('_')[0], id_cir_now + '.nii.gz')
                surv_label = surv_labels[pos][0]
                if_censored = if_censoreds[pos][0]
                text_now = gpt_text[pos][0]

                continuous_clinicl_now = continuous_clinicl[pos][0]
                discrete_clinicl_now = discrete_clinicl[pos][0]
                image_files_need_hospital.append(image_file)
                ids_hospital.append(ID)
                text_hospital.append(text_now)
                labels_hospital.append(surv_label)
                censored_hospital.append(if_censored)
                image_cir_files_need_hospital.append(image_cir_file)
                feat_cir_need_hospital.append(feat_cir_now)
                continuous_clinicl_hospital.append(continuous_clinicl_now)
                discrete_clinicl_hospital.append(discrete_clinicl_now)

    labels_hospital = np.array(labels_hospital)
    censored_hospital = np.array(censored_hospital)
    image_files_need_hospital = np.array(image_files_need_hospital)
    image_cir_files_need_hospital = np.array(image_cir_files_need_hospital)
    feat_cir_need_hospital = np.concatenate(feat_cir_need_hospital)
    continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
    discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
    ids_hospital = np.array(ids_hospital)
    text_hospital = np.array(text_hospital)
    # labels.append(labels_hospital)
    # image_files_need.append(image_files_need_hospital)
    # image_cir_files_need.append(image_cir_files_need_hospital)
    # ids_sum.append(ids_hospital)
    # censoreds_sum.append(censored_hospital)
    # text_sum.append(text_hospital)
    # feat_cir_need.append(feat_cir_need_hospital)
    # continuous_clinicl_sum.append(continuous_clinicl_hospital)
    # discrete_clinicl_sum.append(discrete_clinicl_hospital)


    test_image_files = image_files_need_hospital
    test_image_cir_files = image_cir_files_need_hospital
    test_feat_cir = feat_cir_need_hospital
    test_labels = labels_hospital
    test_con_clinics = continuous_clinicl_hospital
    test_dis_clinics = discrete_clinicl_hospital
    test_ids = ids_hospital
    test_text = text_hospital
    test_censoreds = censored_hospital


    Scaler = joblib.load('scaler.pkl')
    # OHencoder = OneHotEncoder(categories='auto', sparse_output=False)


    # test_image_files = np.concatenate(test_image_files, axis=0)
    # test_image_cir_files = np.concatenate(test_image_cir_files, axis=0)
    # test_feat_cir = np.concatenate(test_feat_cir, axis=0)
    # test_labels = np.concatenate(test_labels, axis=0)
    # test_con_clinics = np.concatenate(test_con_clinics, axis=0)
    # test_dis_clinics = np.concatenate(test_dis_clinics, axis=0)
    # test_ids = np.concatenate(test_ids, axis=0)
    # test_censoreds = np.concatenate(test_censoreds, axis=0)
    # test_text = np.concatenate(test_text, axis=0)
    test_con_clinics_ts = Scaler.transform(test_con_clinics)
    test_con_clinics_ts[:, -1] = 0
    # test_dis_clinics_ts = OHencoder.fit_transform(test_dis_clinics)
    test_clinics = np.concatenate([test_con_clinics_ts, test_dis_clinics], axis=-1)

    labtrans = joblib.load('labtrans.pkl')
    # labtrans.fit(train_labels, train_censoreds)
    # train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    # valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)

    # train_text = train_text.tolist()
    # valid_text = valid_text.tolist()
    test_text = test_text.tolist()

    # train_transforms = Compose([
    #     LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
    #     EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"], ),
    #     # Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #     ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
    #     RandRotated(keys=["image", "liver", "tumor", "cir_liver"], range_x=15, prob=0.5, keep_size=True),
    #     RandFlipd(keys=["image", "liver", "tumor", "cir_liver"], spatial_axis=0, prob=0.2),
    #     RandZoomd(keys=["image", "liver", "tumor", "cir_liver"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
    #     ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
    #     # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    # ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    # train_dataset = LiverDataset_disentangle_cir_feat_cli(train_image_files, train_image_cir_files, train_clinics,
    #                                                       train_feat_cir,
    #                                                       train_labels, train_censoreds, train_durations_discrete,
    #                                                       train_events, train_ids, train_text, train_transforms)
    # valid_dataset = LiverDataset_disentangle_cir_feat_cli(valid_image_files, valid_image_cir_files, valid_clinics,
    #                                                       valid_feat_cir,
    #                                                       valid_labels, valid_censoreds, valid_durations_discrete,
    #                                                       valid_events, valid_ids, valid_text, val_transforms)
    test_dataset = LiverDataset_disentangle_cir_feat_cli(test_image_files, test_image_cir_files, test_clinics,
                                                         test_feat_cir,
                                                         test_labels, test_censoreds, test_durations_discrete,
                                                         test_events, test_ids, test_text, val_transforms)

    # train_infos = np.concatenate([train_ids[:, None], train_image_files[:, None], train_labels[:, None],
    #                          train_censoreds[:, None], train_clinics, np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], valid_image_files[:, None], valid_labels[:, None],
    #                               valid_censoreds[:, None], valid_clinics, np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], test_image_files[:, None], test_labels[:, None],
    #                               test_censoreds[:, None], test_clinics, np.array(test_text)[:, None]], axis=1)
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_infos.csv', train_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_infos.csv', valid_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_infos.csv', test_infos, fmt='%s',
    #            delimiter=',')
    # train_infos = np.concatenate([train_ids[:, None], np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], np.array(test_text)[:, None]], axis=1)
    # train_infos = pd.DataFrame(train_infos, columns=['ids', 'text'])
    # valid_infos = pd.DataFrame(valid_infos, columns=['ids', 'text'])
    # test_infos = pd.DataFrame(test_infos, columns=['ids', 'text'])
    # train_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_text.xlsx', index=False)
    # valid_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_text.xlsx', index=False)
    # test_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_text.xlsx', index=False)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    # val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=2)
    x_image_files, x_clinics, x_cir_feat, x_labels, x_censoreds, x_durations_discrete, \
        x_events, x_ids, x_text = next(iter(test_loader))
    # for i, batch_data in enumerate(train_loader):
    #     s = 0
    # train_sample = next(iter(train_loader))
    return test_dataset, test_loader

def create_dataset_ext_week_persudo_pair_addcirf_fixid2r_radio(matched_feature_file1, matched_feature_file2,
                                                         matched_ids_file1, matched_ids_file2,
                                                         matched_censor_file1, matched_censor_file2, matched_surv_file1,
                                                         matched_surv_file2,
                                                         root_dir, fold, num_durations=6, batch_size=6,
                                                         val_batch_size=4, seed=1998):
    set_seeds(seed)
    features_cp0 = np.load(matched_feature_file1)
    features_cp1 = np.load(matched_feature_file2)[:, 0, ...]
    ids_cp0 = np.load(matched_ids_file1)
    ids_cp1 = np.load(matched_ids_file2)[:, 0, ...]
    censors_cp0 = np.load(matched_censor_file1)
    censors_cp1 = np.load(matched_censor_file2)
    survs_cp0 = np.load(matched_surv_file1)
    survs_cp1 = np.load(matched_surv_file2)

    match_ids_sum = np.concatenate([ids_cp0, ids_cp1])
    match_features_sum = np.concatenate([features_cp0, features_cp1])
    match_censors_sum = np.concatenate([censors_cp0, censors_cp1])
    match_survs_sum = np.concatenate([survs_cp0, survs_cp1])
    # print_config()
    data_dir1 = root_dir
    hospital_names = (
        sorted([x for x in os.listdir(data_dir1) if os.path.isdir(os.path.join(data_dir1, x))]))
    image_files = [[os.path.join(data_dir1, class_name, x)
                    for x in os.listdir(os.path.join(data_dir1, class_name))]
                   for class_name in hospital_names]

    image_files_need = []
    image_cir_files_need = []
    feat_cir_need = []
    info = pd.read_excel(r"/home/wangtao/project/Cirros_HCC/data/table/PFS_ZYJ_20240319_PD_CP_text.xlsx")
    ids = info['序号']
    hospitals = info['病例来源']
    radio_info = np.load(r'/home/wangtao/project/Survival/radio_filter.npy')

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']

    continuous_clinicl = info.iloc[:, [5, 10]]
    discrete_clinicl = info.iloc[:, [6, 7, 8, 9, 11, 12, 23]]
    # discrete_clinicl = info.iloc[:, [7, 8, 9]]
    continuous_clinicl = np.asarray(continuous_clinicl)
    discrete_clinicl = np.asarray(discrete_clinicl)
    discrete_clinicl[np.where(np.isnan(discrete_clinicl) == True)] = 1
    OHencoder = OneHotEncoder(categories='auto', sparse_output=False)
    discrete_clinicl = OHencoder.fit_transform(discrete_clinicl)

    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    radio_sum = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        continuous_clinicl_hospital = []
        discrete_clinicl_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        image_cir_files_need_hospital = []
        feat_cir_need_hospital = []
        radio_hospital = []
        for image_file in image_files_hospital:
            ID = image_file.split('/')[-1].split('.')[0]
            center_now = image_file.split('/')[-2]
            pos = np.where(ids.astype(str) == ID)
            pos_match = np.where(match_ids_sum[:, -1] == center_now + '_' + ID)
            if len(pos[0]) != 0:
                cp_now = cp_labels[pos][0]
                if cp_now != 3 and np.isnan(cp_now) == False:
                    pos_radio = np.where(radio_info[:, 0].astype(int).astype(str) == ID)
                    if len(pos_radio[0]) != 0:
                        feat_radio = radio_info[pos_radio[0], 1:-2]
                    else:
                        feat_radio = np.zeros([1, 20])
                    id_cir_now = match_ids_sum[pos_match[0], 0][0]
                    feat_cir_now = match_features_sum[pos_match[0], 0][0][None, :]
                    image_cir_file = os.path.join(r'/home/wangtao/project/Survival/data_cir/preprocess/Liver/data',
                                                  id_cir_now.split('_')[0], id_cir_now + '.nii.gz')
                    surv_label = surv_labels[pos][0]
                    if_censored = if_censoreds[pos][0]
                    text_now = gpt_text[pos][0]

                    continuous_clinicl_now = continuous_clinicl[pos][0]
                    discrete_clinicl_now = discrete_clinicl[pos][0]
                    image_files_need_hospital.append(image_file)
                    ids_hospital.append(ID)
                    text_hospital.append(text_now)
                    labels_hospital.append(surv_label)
                    censored_hospital.append(if_censored)
                    image_cir_files_need_hospital.append(image_cir_file)
                    feat_cir_need_hospital.append(feat_cir_now)
                    continuous_clinicl_hospital.append(continuous_clinicl_now)
                    discrete_clinicl_hospital.append(discrete_clinicl_now)
                    radio_hospital.append(feat_radio)

        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        image_cir_files_need_hospital = np.array(image_cir_files_need_hospital)
        feat_cir_need_hospital = np.concatenate(feat_cir_need_hospital)
        continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        radio_hospital = np.array(radio_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        image_cir_files_need.append(image_cir_files_need_hospital)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)
        feat_cir_need.append(feat_cir_need_hospital)
        continuous_clinicl_sum.append(continuous_clinicl_hospital)
        discrete_clinicl_sum.append(discrete_clinicl_hospital)
        radio_sum.append(radio_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1998)

    train_val_image_files = []
    train_val_image_cir_files = []
    train_val_feat_cir = []
    train_val_labels = []
    train_val_con_clinics = []
    train_val_dis_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []
    train_val_radio = []

    train_image_files = []
    train_image_cir_files = []
    train_feat_cir = []
    train_labels = []
    train_con_clinics = []
    train_dis_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []
    train_radio = []

    valid_image_files = []
    valid_image_cir_files = []
    valid_feat_cir = []
    valid_labels = []
    valid_con_clinics = []
    valid_dis_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []
    valid_radio = []

    test_image_files = []
    test_image_cir_files = []
    test_feat_cir = []
    test_labels = []
    test_con_clinics = []
    test_dis_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []
    test_radio = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        image_cir_files_split = np.array(image_cir_files_need[hospital])
        feat_cir_split = feat_cir_need[hospital]
        labels_split = labels[hospital]
        # continuous_clinics_split = continuous_clinicl_sum[hospital]
        # discrete_clinics_split = discrete_clinicl_sum[hospital]
        # clinics_split = np.concatenate([continuous_clinics_split, discrete_clinics_split], axis=-1)
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        continuous_clinicl_split = continuous_clinicl_sum[hospital]
        radio_split = radio_sum[hospital]
        discrete_clinicl_split = discrete_clinicl_sum[hospital]
        # if hospital_name_split in ['Center2', 'Center3', 'Center4', 'Center8', 'Center9', 'Center5', 'Center11']:  # 0420_1
        if hospital_name_split not in ['Center5', 'Center9']:
            train_val_image_files.append(image_files_split)
            train_val_image_cir_files.append(image_cir_files_split)
            train_val_labels.append(labels_split)
            # train_val_clinics.append(clinics_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
            train_val_feat_cir.append(feat_cir_split)
            train_val_dis_clinics.append(discrete_clinicl_split)
            train_val_con_clinics.append(continuous_clinicl_split)
            train_val_radio.append(radio_split)
        else:
            test_image_files.append(image_files_split)
            test_image_cir_files.append(image_cir_files_split)
            test_labels.append(labels_split)
            # test_clinics.append(clinics_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)
            test_feat_cir.append(feat_cir_split)
            test_dis_clinics.append(discrete_clinicl_split)
            test_con_clinics.append(continuous_clinicl_split)
            test_radio.append(radio_split)

    for hospital in range(len(train_val_image_files)):
        image_files_split = np.array(train_val_image_files[hospital])
        image_cir_files_split = np.array(train_val_image_cir_files[hospital])
        labels_split = train_val_labels[hospital]
        feat_cir_split = train_val_feat_cir[hospital]
        # clinics_split = train_val_clinics[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        censoreds_split = train_val_censoreds[hospital]
        continuous_clinicl_split = train_val_con_clinics[hospital]
        discrete_clinicl_split = train_val_dis_clinics[hospital]
        radio_split = train_val_radio[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_image_cir_files.append(image_cir_files_split)
            train_labels.append(labels_split)
            train_dis_clinics.append(discrete_clinicl_split)
            train_con_clinics.append(continuous_clinicl_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
            train_feat_cir.append(feat_cir_split)
            train_radio.append(radio_split)
        else:
            # for train_index, valid_index in kf.split(image_files_split, censoreds_split):
            #     _fold = _fold + 1
            #     if _fold == fold:
            train_index = np.loadtxt(
                r'/home/wangtao/project/Survival/split_ids/train_index{}_fold{}_nt59.csv'.format(hospital, fold),
                delimiter=',', dtype=str)
            train_index = train_index.astype(int)
            valid_index = np.loadtxt(
                r'/home/wangtao/project/Survival/split_ids/valid_index{}_fold{}_nt59.csv'.format(hospital, fold),
                delimiter=',', dtype=str)
            valid_index = valid_index.astype(int)
            train_image_files_split = image_files_split[train_index]
            train_image_cir_files_split = image_cir_files_split[train_index]
            train_image_files.append(train_image_files_split)
            train_image_cir_files.append(train_image_cir_files_split)
            train_feat_cir_split = feat_cir_split[train_index]
            train_feat_cir.append(train_feat_cir_split)
            train_labels_split = labels_split[train_index]
            train_labels.append(train_labels_split)
            train_radio_split = radio_split[train_index]
            train_radio.append(train_radio_split)

            train_ids_split = ids_split[train_index]
            train_ids.append(train_ids_split)
            train_censoreds_split = censoreds_split[train_index]
            train_censoreds.append(train_censoreds_split)
            train_text_split = text_split[train_index]
            train_text.append(train_text_split)

            train_discrete_clinicl_split = discrete_clinicl_split[train_index]
            train_dis_clinics.append(train_discrete_clinicl_split)

            train_continuous_clinicl_split = continuous_clinicl_split[train_index]
            train_con_clinics.append(train_continuous_clinicl_split)

            valid_image_files_split = image_files_split[valid_index]
            valid_image_cir_files_split = image_cir_files_split[valid_index]
            valid_image_files.append(valid_image_files_split)
            valid_image_cir_files.append(valid_image_cir_files_split)
            valid_feat_cir_split = feat_cir_split[valid_index]
            valid_feat_cir.append(valid_feat_cir_split)
            valid_labels_split = labels_split[valid_index]
            valid_labels.append(valid_labels_split)
            # valid_clinics_split = clinics_split[valid_index]
            # valid_clinics.append(valid_clinics_split)
            valid_ids_split = ids_split[valid_index]
            valid_ids.append(valid_ids_split)
            valid_censoreds_split = censoreds_split[valid_index]
            valid_censoreds.append(valid_censoreds_split)
            valid_text_split = text_split[valid_index]
            valid_text.append(valid_text_split)
            valid_radio_split = radio_split[valid_index]
            valid_radio.append(valid_radio_split)

            valid_discrete_clinicl_split = discrete_clinicl_split[valid_index]
            valid_dis_clinics.append(valid_discrete_clinicl_split)

            valid_continuous_clinicl_split = continuous_clinicl_split[valid_index]
            valid_con_clinics.append(valid_continuous_clinicl_split)

    Scaler = StandardScaler()
    # OHencoder = OneHotEncoder(categories='auto', sparse_output=False)

    train_image_files = np.concatenate(train_image_files, axis=0)
    train_image_cir_files = np.concatenate(train_image_cir_files, axis=0)
    train_feat_cir = np.concatenate(train_feat_cir, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_con_clinics = np.concatenate(train_con_clinics, axis=0)
    train_dis_clinics = np.concatenate(train_dis_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    train_text = np.concatenate(train_text, axis=0)
    train_radio = np.concatenate(train_radio, axis=0)[:, 0, :]
    train_con_clinics_ts = Scaler.fit_transform(train_con_clinics)
    # train_dis_clinics_ts = OHencoder.fit_transform(train_dis_clinics)
    train_clinics = np.concatenate([train_con_clinics_ts, train_dis_clinics], axis=-1)


    # train_indices = list(range(len(train_image_files)))
    # train_selected_indices = random.sample(train_indices, len(train_image_files) - 40)
    # train_image_files = train_image_files[train_selected_indices]
    # train_image_cir_files = train_image_cir_files[train_selected_indices]
    # train_feat_cir = train_feat_cir[train_selected_indices]
    # train_labels = train_labels[train_selected_indices]
    # train_ids = train_ids[train_selected_indices]
    # train_censoreds = train_censoreds[train_selected_indices]
    # train_text = train_text[train_selected_indices]
    # train_clinics = train_clinics[train_selected_indices]
    # train_radio = train_radio[train_selected_indices]

    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_image_cir_files = np.concatenate(valid_image_cir_files, axis=0)
    valid_feat_cir = np.concatenate(valid_feat_cir, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    valid_con_clinics = np.concatenate(valid_con_clinics, axis=0)
    valid_dis_clinics = np.concatenate(valid_dis_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    valid_text = np.concatenate(valid_text, axis=0)
    valid_con_clinics_ts = Scaler.transform(valid_con_clinics)
    # valid_dis_clinics_ts = OHencoder.fit_transform(valid_dis_clinics)
    valid_clinics = np.concatenate([valid_con_clinics_ts, valid_dis_clinics], axis=-1)
    valid_radio = np.concatenate(valid_radio, axis=0)[:, 0, :]

    # valid_indices = list(range(len(valid_image_files)))
    # valid_selected_indices = random.sample(valid_indices, len(valid_image_files) - 20)
    # valid_image_files = valid_image_files[valid_selected_indices]
    # valid_image_cir_files = valid_image_cir_files[valid_selected_indices]
    # valid_feat_cir = valid_feat_cir[valid_selected_indices]
    # valid_labels = valid_labels[valid_selected_indices]
    # valid_ids = valid_ids[valid_selected_indices]
    # valid_censoreds = valid_censoreds[valid_selected_indices]
    # valid_text = valid_text[valid_selected_indices]
    # valid_clinics = valid_clinics[valid_selected_indices]
    # valid_radio = valid_radio[valid_selected_indices]

    test_image_files = np.concatenate(test_image_files, axis=0)
    test_image_cir_files = np.concatenate(test_image_cir_files, axis=0)
    test_feat_cir = np.concatenate(test_feat_cir, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_con_clinics = np.concatenate(test_con_clinics, axis=0)
    test_dis_clinics = np.concatenate(test_dis_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    test_text = np.concatenate(test_text, axis=0)
    test_con_clinics_ts = Scaler.transform(test_con_clinics)
    # test_dis_clinics_ts = OHencoder.fit_transform(test_dis_clinics)
    test_clinics = np.concatenate([test_con_clinics_ts, test_dis_clinics], axis=-1)
    test_radio = np.concatenate(test_radio, axis=0)[:, 0, :]

    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)

    train_text = train_text.tolist()
    valid_text = valid_text.tolist()
    test_text = test_text.tolist()

    train_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"], ),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        RandRotated(keys=["image", "liver", "tumor", "cir_liver"], range_x=15, prob=0.5, keep_size=True),
        RandFlipd(keys=["image", "liver", "tumor", "cir_liver"], spatial_axis=0, prob=0.2),
        RandZoomd(keys=["image", "liver", "tumor", "cir_liver"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    train_dataset = LiverDataset_disentangle_cir_feat_cli_radio(train_image_files, train_image_cir_files, train_clinics,
                                                          train_feat_cir, train_labels, train_censoreds,
                                                          train_durations_discrete, train_events, train_ids, train_text,
                                                          train_radio, train_transforms)
    valid_dataset = LiverDataset_disentangle_cir_feat_cli_radio(valid_image_files, valid_image_cir_files, valid_clinics,
                                                          valid_feat_cir,
                                                          valid_labels, valid_censoreds, valid_durations_discrete,
                                                          valid_events, valid_ids, valid_text, valid_radio,
                                                                val_transforms,)
    test_dataset = LiverDataset_disentangle_cir_feat_cli_radio(test_image_files, test_image_cir_files, test_clinics,
                                                         test_feat_cir,
                                                         test_labels, test_censoreds, test_durations_discrete,
                                                         test_events, test_ids, test_text, test_radio,
                                                               val_transforms)

    # train_infos = np.concatenate([train_ids[:, None], train_image_files[:, None], train_labels[:, None],
    #                          train_censoreds[:, None], train_clinics, np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], valid_image_files[:, None], valid_labels[:, None],
    #                               valid_censoreds[:, None], valid_clinics, np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], test_image_files[:, None], test_labels[:, None],
    #                               test_censoreds[:, None], test_clinics, np.array(test_text)[:, None]], axis=1)
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_infos.csv', train_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_infos.csv', valid_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_infos.csv', test_infos, fmt='%s',
    #            delimiter=',')
    # train_infos = np.concatenate([train_ids[:, None], np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], np.array(test_text)[:, None]], axis=1)
    # train_infos = pd.DataFrame(train_infos, columns=['ids', 'text'])
    # valid_infos = pd.DataFrame(valid_infos, columns=['ids', 'text'])
    # test_infos = pd.DataFrame(test_infos, columns=['ids', 'text'])
    # train_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_text.xlsx', index=False)
    # valid_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_text.xlsx', index=False)
    # test_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_text.xlsx', index=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=2)
    x_image_files, x_clinics, x_cir_feat, x_labels, x_censoreds, x_durations_discrete, \
        x_events, x_ids, x_text, x_radio = next(iter(train_loader))
    # for i, batch_data in enumerate(train_loader):
    #     s = 0
    # train_sample = next(iter(train_loader))
    return train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader

def create_dataset_ext_week_persudo_pair_addcirf_fixid2r_radio_group(matched_feature_file1, matched_feature_file2,
                                                         matched_ids_file1, matched_ids_file2,
                                                         matched_censor_file1, matched_censor_file2, matched_surv_file1,
                                                         matched_surv_file2,
                                                         root_dir, fold, num_durations=6, batch_size=6,
                                                         val_batch_size=4, seed=1998):
    set_seeds(seed)
    features_cp0 = np.load(matched_feature_file1)
    features_cp1 = np.load(matched_feature_file2)[:, 0, ...]
    ids_cp0 = np.load(matched_ids_file1)
    ids_cp1 = np.load(matched_ids_file2)[:, 0, ...]
    censors_cp0 = np.load(matched_censor_file1)
    censors_cp1 = np.load(matched_censor_file2)
    survs_cp0 = np.load(matched_surv_file1)
    survs_cp1 = np.load(matched_surv_file2)

    match_ids_sum = np.concatenate([ids_cp0, ids_cp1])
    match_features_sum = np.concatenate([features_cp0, features_cp1])
    match_censors_sum = np.concatenate([censors_cp0, censors_cp1])
    match_survs_sum = np.concatenate([survs_cp0, survs_cp1])
    # print_config()
    data_dir1 = root_dir
    hospital_names = (
        sorted([x for x in os.listdir(data_dir1) if os.path.isdir(os.path.join(data_dir1, x))]))
    image_files = [[os.path.join(data_dir1, class_name, x)
                    for x in os.listdir(os.path.join(data_dir1, class_name))]
                   for class_name in hospital_names]

    image_files_need = []
    image_cir_files_need = []
    feat_cir_need = []
    info = pd.read_excel(r"/home/wangtao/project/Cirros_HCC/data/table/PFS_ZYJ_20240319_PD_CP_text.xlsx")
    ids = info['序号']
    hospitals = info['病例来源']
    radio_info = np.load(r'/home/wangtao/project/Survival/radio_filter.npy')

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    treat_labels = info['初始治疗']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']

    continuous_clinicl = info.iloc[:, [5, 10]]
    discrete_clinicl = info.iloc[:, [6, 7, 8, 9, 11, 12, 23]]
    # discrete_clinicl = info.iloc[:, [7, 8, 9]]
    continuous_clinicl = np.asarray(continuous_clinicl)
    discrete_clinicl = np.asarray(discrete_clinicl)
    discrete_clinicl[np.where(np.isnan(discrete_clinicl) == True)] = 1
    OHencoder = OneHotEncoder(categories='auto', sparse_output=False)
    discrete_clinicl = OHencoder.fit_transform(discrete_clinicl)

    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    treat_labels = np.asarray(treat_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    radio_sum = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        continuous_clinicl_hospital = []
        discrete_clinicl_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        image_cir_files_need_hospital = []
        feat_cir_need_hospital = []
        radio_hospital = []
        for image_file in image_files_hospital:
            ID = image_file.split('/')[-1].split('.')[0]
            center_now = image_file.split('/')[-2]
            pos = np.where(ids.astype(str) == ID)
            pos_match = np.where(match_ids_sum[:, -1] == center_now + '_' + ID)
            if len(pos[0]) != 0:
                cp_now = cp_labels[pos][0]
                treat_now = treat_labels[pos][0]
                if cp_now != 3 and np.isnan(cp_now) == False and (treat_now == 2 or treat_now == 3):
                    pos_radio = np.where(radio_info[:, 0].astype(int).astype(str) == ID)
                    if len(pos_radio[0]) != 0:
                        feat_radio = radio_info[pos_radio[0], 1:-2]
                    else:
                        feat_radio = np.zeros([1, 20])
                    id_cir_now = match_ids_sum[pos_match[0], 0][0]
                    feat_cir_now = match_features_sum[pos_match[0], 0][0][None, :]
                    image_cir_file = os.path.join(r'/home/wangtao/project/Survival/data_cir/preprocess/Liver/data',
                                                  id_cir_now.split('_')[0], id_cir_now + '.nii.gz')
                    surv_label = surv_labels[pos][0]
                    if_censored = if_censoreds[pos][0]
                    text_now = gpt_text[pos][0]

                    continuous_clinicl_now = continuous_clinicl[pos][0]
                    discrete_clinicl_now = discrete_clinicl[pos][0]
                    image_files_need_hospital.append(image_file)
                    ids_hospital.append(ID)
                    text_hospital.append(text_now)
                    labels_hospital.append(surv_label)
                    censored_hospital.append(if_censored)
                    image_cir_files_need_hospital.append(image_cir_file)
                    feat_cir_need_hospital.append(feat_cir_now)
                    continuous_clinicl_hospital.append(continuous_clinicl_now)
                    discrete_clinicl_hospital.append(discrete_clinicl_now)
                    radio_hospital.append(feat_radio)

        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        image_cir_files_need_hospital = np.array(image_cir_files_need_hospital)
        feat_cir_need_hospital = np.concatenate(feat_cir_need_hospital)
        continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        radio_hospital = np.array(radio_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        image_cir_files_need.append(image_cir_files_need_hospital)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)
        feat_cir_need.append(feat_cir_need_hospital)
        continuous_clinicl_sum.append(continuous_clinicl_hospital)
        discrete_clinicl_sum.append(discrete_clinicl_hospital)
        radio_sum.append(radio_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1998)

    train_val_image_files = []
    train_val_image_cir_files = []
    train_val_feat_cir = []
    train_val_labels = []
    train_val_con_clinics = []
    train_val_dis_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []
    train_val_radio = []

    train_image_files = []
    train_image_cir_files = []
    train_feat_cir = []
    train_labels = []
    train_con_clinics = []
    train_dis_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []
    train_radio = []

    valid_image_files = []
    valid_image_cir_files = []
    valid_feat_cir = []
    valid_labels = []
    valid_con_clinics = []
    valid_dis_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []
    valid_radio = []

    test_image_files = []
    test_image_cir_files = []
    test_feat_cir = []
    test_labels = []
    test_con_clinics = []
    test_dis_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []
    test_radio = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        image_cir_files_split = np.array(image_cir_files_need[hospital])
        feat_cir_split = feat_cir_need[hospital]
        labels_split = labels[hospital]
        # continuous_clinics_split = continuous_clinicl_sum[hospital]
        # discrete_clinics_split = discrete_clinicl_sum[hospital]
        # clinics_split = np.concatenate([continuous_clinics_split, discrete_clinics_split], axis=-1)
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        continuous_clinicl_split = continuous_clinicl_sum[hospital]
        radio_split = radio_sum[hospital]
        discrete_clinicl_split = discrete_clinicl_sum[hospital]
        # if hospital_name_split in ['Center2', 'Center3', 'Center4', 'Center8', 'Center9', 'Center5', 'Center11']:  # 0420_1
        if hospital_name_split not in ['Center5', 'Center9']:
            train_val_image_files.append(image_files_split)
            train_val_image_cir_files.append(image_cir_files_split)
            train_val_labels.append(labels_split)
            # train_val_clinics.append(clinics_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
            train_val_feat_cir.append(feat_cir_split)
            train_val_dis_clinics.append(discrete_clinicl_split)
            train_val_con_clinics.append(continuous_clinicl_split)
            train_val_radio.append(radio_split)
        else:
            test_image_files.append(image_files_split)
            test_image_cir_files.append(image_cir_files_split)
            test_labels.append(labels_split)
            # test_clinics.append(clinics_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)
            test_feat_cir.append(feat_cir_split)
            test_dis_clinics.append(discrete_clinicl_split)
            test_con_clinics.append(continuous_clinicl_split)
            test_radio.append(radio_split)

    for hospital in range(len(train_val_image_files)):

        image_files_split = np.array(train_val_image_files[hospital])
        image_cir_files_split = np.array(train_val_image_cir_files[hospital])
        labels_split = train_val_labels[hospital]
        feat_cir_split = train_val_feat_cir[hospital]
        # clinics_split = train_val_clinics[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        censoreds_split = train_val_censoreds[hospital]
        continuous_clinicl_split = train_val_con_clinics[hospital]
        discrete_clinicl_split = train_val_dis_clinics[hospital]
        radio_split = train_val_radio[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_image_cir_files.append(image_cir_files_split)
            train_labels.append(labels_split)
            train_dis_clinics.append(discrete_clinicl_split)
            train_con_clinics.append(continuous_clinicl_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
            train_feat_cir.append(feat_cir_split)
            train_radio.append(radio_split)
        else:
            if len(np.where(censoreds_split==0)[0]) >=5 or len(np.where(censoreds_split==1)[0]) >=5:
                for train_index, valid_index in kf.split(image_files_split, censoreds_split):
                    _fold = _fold + 1
                    if _fold == fold:
                        # train_index = np.loadtxt(
                        #     r'/home/wangtao/project/Survival/split_ids/train_index{}_fold{}_nt59.csv'.format(hospital, fold),
                        #     delimiter=',', dtype=str)
                        # train_index = train_index.astype(int)
                        # valid_index = np.loadtxt(
                        #     r'/home/wangtao/project/Survival/split_ids/valid_index{}_fold{}_nt59.csv'.format(hospital, fold),
                        #     delimiter=',', dtype=str)
                        # valid_index = valid_index.astype(int)
                        train_image_files_split = image_files_split[train_index]
                        train_image_cir_files_split = image_cir_files_split[train_index]
                        train_image_files.append(train_image_files_split)
                        train_image_cir_files.append(train_image_cir_files_split)
                        train_feat_cir_split = feat_cir_split[train_index]
                        train_feat_cir.append(train_feat_cir_split)
                        train_labels_split = labels_split[train_index]
                        train_labels.append(train_labels_split)
                        train_radio_split = radio_split[train_index]
                        train_radio.append(train_radio_split)

                        train_ids_split = ids_split[train_index]
                        train_ids.append(train_ids_split)
                        train_censoreds_split = censoreds_split[train_index]
                        train_censoreds.append(train_censoreds_split)
                        train_text_split = text_split[train_index]
                        train_text.append(train_text_split)

                        train_discrete_clinicl_split = discrete_clinicl_split[train_index]
                        train_dis_clinics.append(train_discrete_clinicl_split)

                        train_continuous_clinicl_split = continuous_clinicl_split[train_index]
                        train_con_clinics.append(train_continuous_clinicl_split)

                        valid_image_files_split = image_files_split[valid_index]
                        valid_image_cir_files_split = image_cir_files_split[valid_index]
                        valid_image_files.append(valid_image_files_split)
                        valid_image_cir_files.append(valid_image_cir_files_split)
                        valid_feat_cir_split = feat_cir_split[valid_index]
                        valid_feat_cir.append(valid_feat_cir_split)
                        valid_labels_split = labels_split[valid_index]
                        valid_labels.append(valid_labels_split)
                        # valid_clinics_split = clinics_split[valid_index]
                        # valid_clinics.append(valid_clinics_split)
                        valid_ids_split = ids_split[valid_index]
                        valid_ids.append(valid_ids_split)
                        valid_censoreds_split = censoreds_split[valid_index]
                        valid_censoreds.append(valid_censoreds_split)
                        valid_text_split = text_split[valid_index]
                        valid_text.append(valid_text_split)
                        valid_radio_split = radio_split[valid_index]
                        valid_radio.append(valid_radio_split)

                        valid_discrete_clinicl_split = discrete_clinicl_split[valid_index]
                        valid_dis_clinics.append(valid_discrete_clinicl_split)

                        valid_continuous_clinicl_split = continuous_clinicl_split[valid_index]
                        valid_con_clinics.append(valid_continuous_clinicl_split)
                _fold = 0
            else:
                train_image_files_split = image_files_split
                train_image_cir_files_split = image_cir_files_split
                train_image_files.append(train_image_files_split)
                train_image_cir_files.append(train_image_cir_files_split)
                train_feat_cir_split = feat_cir_split
                train_feat_cir.append(train_feat_cir_split)
                train_labels_split = labels_split
                train_labels.append(train_labels_split)
                train_radio_split = radio_split
                train_radio.append(train_radio_split)

                train_ids_split = ids_split
                train_ids.append(train_ids_split)
                train_censoreds_split = censoreds_split
                train_censoreds.append(train_censoreds_split)
                train_text_split = text_split
                train_text.append(train_text_split)

                train_discrete_clinicl_split = discrete_clinicl_split
                train_dis_clinics.append(train_discrete_clinicl_split)

                train_continuous_clinicl_split = continuous_clinicl_split
                train_con_clinics.append(train_continuous_clinicl_split)

    Scaler = StandardScaler()
    # OHencoder = OneHotEncoder(categories='auto', sparse_output=False)

    train_image_files = np.concatenate(train_image_files, axis=0)
    train_image_cir_files = np.concatenate(train_image_cir_files, axis=0)
    train_feat_cir = np.concatenate(train_feat_cir, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_con_clinics = np.concatenate(train_con_clinics, axis=0)
    train_dis_clinics = np.concatenate(train_dis_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    train_text = np.concatenate(train_text, axis=0)
    train_radio = np.concatenate(train_radio, axis=0)[:, 0, :]
    train_con_clinics_ts = Scaler.fit_transform(train_con_clinics)
    # train_dis_clinics_ts = OHencoder.fit_transform(train_dis_clinics)
    train_clinics = np.concatenate([train_con_clinics_ts, train_dis_clinics], axis=-1)


    # train_indices = list(range(len(train_image_files)))
    # train_selected_indices = random.sample(train_indices, len(train_image_files) - 40)
    # train_image_files = train_image_files[train_selected_indices]
    # train_image_cir_files = train_image_cir_files[train_selected_indices]
    # train_feat_cir = train_feat_cir[train_selected_indices]
    # train_labels = train_labels[train_selected_indices]
    # train_ids = train_ids[train_selected_indices]
    # train_censoreds = train_censoreds[train_selected_indices]
    # train_text = train_text[train_selected_indices]
    # train_clinics = train_clinics[train_selected_indices]
    # train_radio = train_radio[train_selected_indices]

    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_image_cir_files = np.concatenate(valid_image_cir_files, axis=0)
    valid_feat_cir = np.concatenate(valid_feat_cir, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    valid_con_clinics = np.concatenate(valid_con_clinics, axis=0)
    valid_dis_clinics = np.concatenate(valid_dis_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    valid_text = np.concatenate(valid_text, axis=0)
    valid_con_clinics_ts = Scaler.transform(valid_con_clinics)
    # valid_dis_clinics_ts = OHencoder.fit_transform(valid_dis_clinics)
    valid_clinics = np.concatenate([valid_con_clinics_ts, valid_dis_clinics], axis=-1)
    valid_radio = np.concatenate(valid_radio, axis=0)[:, 0, :]

    # valid_indices = list(range(len(valid_image_files)))
    # valid_selected_indices = random.sample(valid_indices, len(valid_image_files) - 20)
    # valid_image_files = valid_image_files[valid_selected_indices]
    # valid_image_cir_files = valid_image_cir_files[valid_selected_indices]
    # valid_feat_cir = valid_feat_cir[valid_selected_indices]
    # valid_labels = valid_labels[valid_selected_indices]
    # valid_ids = valid_ids[valid_selected_indices]
    # valid_censoreds = valid_censoreds[valid_selected_indices]
    # valid_text = valid_text[valid_selected_indices]
    # valid_clinics = valid_clinics[valid_selected_indices]
    # valid_radio = valid_radio[valid_selected_indices]

    test_image_files = np.concatenate(test_image_files, axis=0)
    test_image_cir_files = np.concatenate(test_image_cir_files, axis=0)
    test_feat_cir = np.concatenate(test_feat_cir, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_con_clinics = np.concatenate(test_con_clinics, axis=0)
    test_dis_clinics = np.concatenate(test_dis_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    test_text = np.concatenate(test_text, axis=0)
    test_con_clinics_ts = Scaler.transform(test_con_clinics)
    # test_dis_clinics_ts = OHencoder.fit_transform(test_dis_clinics)
    test_clinics = np.concatenate([test_con_clinics_ts, test_dis_clinics], axis=-1)
    test_radio = np.concatenate(test_radio, axis=0)[:, 0, :]

    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)

    train_text = train_text.tolist()
    valid_text = valid_text.tolist()
    test_text = test_text.tolist()

    train_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"], ),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        RandRotated(keys=["image", "liver", "tumor", "cir_liver"], range_x=15, prob=0.5, keep_size=True),
        RandFlipd(keys=["image", "liver", "tumor", "cir_liver"], spatial_axis=0, prob=0.2),
        RandZoomd(keys=["image", "liver", "tumor", "cir_liver"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    train_dataset = LiverDataset_disentangle_cir_feat_cli_radio(train_image_files, train_image_cir_files, train_clinics,
                                                          train_feat_cir, train_labels, train_censoreds,
                                                          train_durations_discrete, train_events, train_ids, train_text,
                                                          train_radio, train_transforms)
    valid_dataset = LiverDataset_disentangle_cir_feat_cli_radio(valid_image_files, valid_image_cir_files, valid_clinics,
                                                          valid_feat_cir,
                                                          valid_labels, valid_censoreds, valid_durations_discrete,
                                                          valid_events, valid_ids, valid_text, valid_radio,
                                                                val_transforms,)
    test_dataset = LiverDataset_disentangle_cir_feat_cli_radio(test_image_files, test_image_cir_files, test_clinics,
                                                         test_feat_cir,
                                                         test_labels, test_censoreds, test_durations_discrete,
                                                         test_events, test_ids, test_text, test_radio,
                                                               val_transforms)

    # train_infos = np.concatenate([train_ids[:, None], train_image_files[:, None], train_labels[:, None],
    #                          train_censoreds[:, None], train_clinics, np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], valid_image_files[:, None], valid_labels[:, None],
    #                               valid_censoreds[:, None], valid_clinics, np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], test_image_files[:, None], test_labels[:, None],
    #                               test_censoreds[:, None], test_clinics, np.array(test_text)[:, None]], axis=1)
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_infos.csv', train_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_infos.csv', valid_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_infos.csv', test_infos, fmt='%s',
    #            delimiter=',')
    # train_infos = np.concatenate([train_ids[:, None], np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], np.array(test_text)[:, None]], axis=1)
    # train_infos = pd.DataFrame(train_infos, columns=['ids', 'text'])
    # valid_infos = pd.DataFrame(valid_infos, columns=['ids', 'text'])
    # test_infos = pd.DataFrame(test_infos, columns=['ids', 'text'])
    # train_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_text.xlsx', index=False)
    # valid_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_text.xlsx', index=False)
    # test_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_text.xlsx', index=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=2)
    x_image_files, x_clinics, x_cir_feat, x_labels, x_censoreds, x_durations_discrete, \
        x_events, x_ids, x_text, x_radio = next(iter(train_loader))
    # for i, batch_data in enumerate(train_loader):
    #     s = 0
    # train_sample = next(iter(train_loader))
    return train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader

def create_dataset_ext_week_persudo_pair_addcirf_fixid2r_gnn(matched_feature_file1, matched_feature_file2,
                                                         matched_ids_file1, matched_ids_file2,
                                                         matched_censor_file1, matched_censor_file2, matched_surv_file1,
                                                         matched_surv_file2,
                                                         root_dir, fold, num_durations=6, batch_size=6,
                                                         val_batch_size=4, seed=1998):
    set_seeds(seed)
    features_cp0 = np.load(matched_feature_file1)
    features_cp1 = np.load(matched_feature_file2)[:, 0, ...]
    ids_cp0 = np.load(matched_ids_file1)
    ids_cp1 = np.load(matched_ids_file2)[:, 0, ...]
    censors_cp0 = np.load(matched_censor_file1)
    censors_cp1 = np.load(matched_censor_file2)
    survs_cp0 = np.load(matched_surv_file1)
    survs_cp1 = np.load(matched_surv_file2)

    match_ids_sum = np.concatenate([ids_cp0, ids_cp1])
    match_features_sum = np.concatenate([features_cp0, features_cp1])
    match_censors_sum = np.concatenate([censors_cp0, censors_cp1])
    match_survs_sum = np.concatenate([survs_cp0, survs_cp1])
    # print_config()
    data_dir1 = root_dir
    hospital_names = (
        sorted([x for x in os.listdir(data_dir1) if os.path.isdir(os.path.join(data_dir1, x))]))
    image_files = [[os.path.join(data_dir1, class_name, x)
                    for x in os.listdir(os.path.join(data_dir1, class_name))]
                   for class_name in hospital_names]

    image_files_need = []
    image_cir_files_need = []
    feat_cir_need = []
    info = pd.read_excel(r"/home/wangtao/project/Cirros_HCC/data/table/PFS_ZYJ_20240319_PD_CP_text2.xlsx")
    ids = info['序号']
    hospitals = info['病例来源']
    radio_info = np.load(r'/home/wangtao/project/Survival/radio_filter.npy')

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']

    continuous_clinicl = info.iloc[:, [5, 10]]
    discrete_clinicl = info.iloc[:, [6, 7, 8, 9, 11, 12, 23]]
    cpx_clinical = info.iloc[:, [13, 14, 15, 16]]
    # discrete_clinicl = info.iloc[:, [7, 8, 9]]
    continuous_clinicl = np.asarray(continuous_clinicl)
    cpx_clinical = np.asarray(cpx_clinical)
    discrete_clinicl = np.asarray(discrete_clinicl)
    discrete_clinicl[np.where(np.isnan(discrete_clinicl) == True)] = 1
    # OHencoder = OneHotEncoder(categories='auto', sparse_output=False)
    # discrete_clinicl = OHencoder.fit_transform(discrete_clinicl)

    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    cpx_clinical_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    radio_sum = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        continuous_clinicl_hospital = []
        discrete_clinicl_hospital = []
        cpx_clinical_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        image_cir_files_need_hospital = []
        feat_cir_need_hospital = []
        radio_hospital = []
        for image_file in image_files_hospital:
            ID = image_file.split('/')[-1].split('.')[0]
            center_now = image_file.split('/')[-2]
            pos = np.where(ids.astype(str) == ID)
            pos_match = np.where(match_ids_sum[:, -1] == center_now + '_' + ID)
            if len(pos[0]) != 0:
                cp_now = cp_labels[pos][0]
                if cp_now != 3 and np.isnan(cp_now) == False:
                    pos_radio = np.where(radio_info[:, 0].astype(int).astype(str) == ID)
                    if len(pos_radio[0]) != 0:
                        feat_radio = radio_info[pos_radio[0], 1:-2]
                    else:
                        feat_radio = np.zeros([1, 20])
                    id_cir_now = match_ids_sum[pos_match[0], 0][0]
                    feat_cir_now = match_features_sum[pos_match[0], 0][0][None, :]
                    image_cir_file = os.path.join(r'/home/wangtao/project/Survival/data_cir/preprocess/Liver/data',
                                                  id_cir_now.split('_')[0], id_cir_now + '.nii.gz')
                    surv_label = surv_labels[pos][0]
                    if_censored = if_censoreds[pos][0]
                    text_now = gpt_text[pos][0]

                    cpx_clinicl_now = cpx_clinical[pos][0]
                    continuous_clinicl_now = continuous_clinicl[pos][0]
                    discrete_clinicl_now = discrete_clinicl[pos][0]
                    cpx_clinical_hospital.append(cpx_clinicl_now)
                    image_files_need_hospital.append(image_file)
                    ids_hospital.append(ID)
                    text_hospital.append(text_now)
                    labels_hospital.append(surv_label)
                    censored_hospital.append(if_censored)
                    image_cir_files_need_hospital.append(image_cir_file)
                    feat_cir_need_hospital.append(feat_cir_now)
                    continuous_clinicl_hospital.append(continuous_clinicl_now)
                    discrete_clinicl_hospital.append(discrete_clinicl_now)
                    radio_hospital.append(feat_radio)

        cpx_clinical_hospital = np.array(cpx_clinical_hospital)
        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        image_cir_files_need_hospital = np.array(image_cir_files_need_hospital)
        feat_cir_need_hospital = np.concatenate(feat_cir_need_hospital)
        continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        radio_hospital = np.array(radio_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        image_cir_files_need.append(image_cir_files_need_hospital)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)
        feat_cir_need.append(feat_cir_need_hospital)
        continuous_clinicl_sum.append(continuous_clinicl_hospital)
        discrete_clinicl_sum.append(discrete_clinicl_hospital)
        radio_sum.append(radio_hospital)
        cpx_clinical_sum.append(cpx_clinical_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1998)
    # age
    ori_text1 = ('The patient is {age} years of age; advanced age tends to be associated with a poorer prognosis, '
                 'and is correlated with liver function parameters.')
    # gender
    ori_text2 = ('The patient is {gender}, and gender may be highly correlated with hepatitis types.')
    # hepatitis
    ori_text3_1 = ('The patient has been diagnosed with {medical_history}, which may progress to cirrhosis '
                   'if inadequately managed.')
    ori_text3_2 = ('The patient demonstrates no history of hepatitis-related pathologies')
    # cirrhosis
    ori_text4 = ('Patients with established cirrhosis exhibit a significantly elevated propensity for progression '
                 'to hepatocellular carcinoma, representing a well-documented oncogenic trajectory.')
    # hepatic function
    ori_text5 = ('Deteriorating hepatic function serves as a critical prognostic indicator, with more severe impairment'
                 ' correlating strongly with adverse clinical outcomes. Hepatic function dictates initial treatment.')
    # Child-Pugh classification
    ori_text6 = ('The Child-Pugh classification of patient is {CP_stage}. The Child-Pugh classification is utilized '
                 'to assess hepatic function. A higher Child-Pugh classification score typically portends a '
                 'less favorable prognosis.')
    # AFP level
    ori_text7 = ('The AFP level of patient is {AFP_grade}. Elevated AFP levels typically indicate heightened tumor '
                 'activity and may be implicated in the presence of an increased tumor lesion as well as potential '
                 'extrahepatic metastatic dissemination.')
    # tumor size
    ori_text8 = ('The maximum tumor diameter in the patient measures {tumor_size} mm. Larger maximal tumor diameter '
                 'exhibits higher Barcelona Clinic Liver Cancer stage, higher risk of extrahepatic metastatic '
                 'dissemination, and higher risk of vascular invasion. Larger tumor dimensions consistently confer '
                 'statistically inferior prognoses.')
    # extrahepatic metastasis
    ori_text9 = ('The occurrence of extrahepatic metastasis invariably portends an unfavorable prognostic trajectory.')
    # multifocal lesions
    ori_text10_1 = ('The patient has {tumor_number} tumor lesions. The presence of multifocal lesions portends elevated'
                    ' risks of vascular invasion, advanced Barcelona Clinic Liver Cancer staging, and poor'
                    ' prognosis.')
    ori_text10_2 = ('The patient only has one tumor lesion.')
    # vascular invasion
    ori_text11 = ('Vascular invasion portends a grim prognosis.')
    # BCLC
    ori_text12 = ('Staging according to the Barcelona Clinic Liver Cancer (BCLC) system is stage {BCLC_stage}. '
                  'The BCLC staging not only dictates the selection of initial treatment but is also closely '
                  'correlated with hepatic function. Higher BCLC stages typically portend poorer prognosis.')
    # initial treatment
    ori_text13 = ('The initial treatment strategy employed is {treatment_strategy}. There exists a strong correlation '
                  'between the selection of initial therapy and prognosis; patients undergoing hepatic resection '
                  'potentially experience more favourable outcomes.')
    # INR
    ori_text14_1 = ('The prothrombin time of patient is {INR} s. A prolonged prothrombin time correlates with a higher '
                  'Child-Pugh classification.')
    ori_text14_2 = ('The prothrombin time of patient is unavailable.')
    # ALB
    ori_text15_1 = ('The albumin level of patient is {ALB} g/L. A lower albumin level is associated with a higher '
                  'Child-Pugh classification.')
    ori_text15_2 = ('The albumin level of patient is unavailable.')
    # bilirubin concentration
    ori_text16_1 = ('The total bilirubin concentration of patient is {TBIL} μmol/L. '
                  'Elevated total bilirubin levels correlate with a higher Child-Pugh classification.')
    ori_text16_2 = ('The total bilirubin concentration of patient is unavailable.')
    # ascites
    ori_text17_1 = ('The alanine aminotransferase level of patient stands at {ALT} U/L. Elevated alanine aminotransferase '
                  'denotes liver injury and may also suggest a less favorable prognosis.')
    ori_text17_2 = ('The alanine aminotransferase level of patient is unavailable.')

    train_val_image_files = []
    train_val_image_cir_files = []
    train_val_feat_cir = []
    train_val_labels = []
    train_val_con_clinics = []
    train_val_dis_clinics = []
    train_val_cpx_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []
    train_val_radio = []

    train_image_files = []
    train_image_cir_files = []
    train_feat_cir = []
    train_labels = []
    train_con_clinics = []
    train_dis_clinics = []
    train_cpx_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []
    train_radio = []

    valid_image_files = []
    valid_image_cir_files = []
    valid_feat_cir = []
    valid_labels = []
    valid_con_clinics = []
    valid_dis_clinics = []
    valid_cpx_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []
    valid_radio = []

    test_image_files = []
    test_image_cir_files = []
    test_feat_cir = []
    test_labels = []
    test_con_clinics = []
    test_dis_clinics = []
    test_cpx_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []
    test_radio = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        image_cir_files_split = np.array(image_cir_files_need[hospital])
        feat_cir_split = feat_cir_need[hospital]
        labels_split = labels[hospital]
        # continuous_clinics_split = continuous_clinicl_sum[hospital]
        # discrete_clinics_split = discrete_clinicl_sum[hospital]
        # clinics_split = np.concatenate([continuous_clinics_split, discrete_clinics_split], axis=-1)
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        continuous_clinicl_split = continuous_clinicl_sum[hospital]
        radio_split = radio_sum[hospital]
        discrete_clinicl_split = discrete_clinicl_sum[hospital]
        cpx_clinical_split = cpx_clinical_sum[hospital]
        # if hospital_name_split in ['Center2', 'Center3', 'Center4', 'Center8', 'Center9', 'Center5', 'Center11']:  # 0420_1
        if hospital_name_split not in ['Center5', 'Center9']:
            train_val_image_files.append(image_files_split)
            train_val_image_cir_files.append(image_cir_files_split)
            train_val_labels.append(labels_split)
            # train_val_clinics.append(clinics_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
            train_val_feat_cir.append(feat_cir_split)
            train_val_dis_clinics.append(discrete_clinicl_split)
            train_val_con_clinics.append(continuous_clinicl_split)
            train_val_radio.append(radio_split)
            train_val_cpx_clinics.append(cpx_clinical_split)
        else:
            test_image_files.append(image_files_split)
            test_image_cir_files.append(image_cir_files_split)
            test_labels.append(labels_split)
            # test_clinics.append(clinics_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)
            test_feat_cir.append(feat_cir_split)
            test_dis_clinics.append(discrete_clinicl_split)
            test_con_clinics.append(continuous_clinicl_split)
            test_radio.append(radio_split)
            test_cpx_clinics.append(cpx_clinical_split)

    for hospital in range(len(train_val_image_files)):
        image_files_split = np.array(train_val_image_files[hospital])
        image_cir_files_split = np.array(train_val_image_cir_files[hospital])
        labels_split = train_val_labels[hospital]
        feat_cir_split = train_val_feat_cir[hospital]
        # clinics_split = train_val_clinics[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        censoreds_split = train_val_censoreds[hospital]
        continuous_clinicl_split = train_val_con_clinics[hospital]
        discrete_clinicl_split = train_val_dis_clinics[hospital]
        cpx_clinicl_split = train_val_cpx_clinics[hospital]
        radio_split = train_val_radio[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_image_cir_files.append(image_cir_files_split)
            train_labels.append(labels_split)
            train_dis_clinics.append(discrete_clinicl_split)
            train_con_clinics.append(continuous_clinicl_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
            train_feat_cir.append(feat_cir_split)
            train_radio.append(radio_split)
            train_cpx_clinics.append(cpx_clinicl_split)
        else:
            # for train_index, valid_index in kf.split(image_files_split, censoreds_split):
            #     _fold = _fold + 1
            #     if _fold == fold:
            train_index = np.loadtxt(
                r'/home/wangtao/project/Survival/split_ids/train_index{}_fold{}_nt59.csv'.format(hospital, fold),
                delimiter=',', dtype=str)
            train_index = train_index.astype(int)
            valid_index = np.loadtxt(
                r'/home/wangtao/project/Survival/split_ids/valid_index{}_fold{}_nt59.csv'.format(hospital, fold),
                delimiter=',', dtype=str)
            valid_index = valid_index.astype(int)
            train_image_files_split = image_files_split[train_index]
            train_image_cir_files_split = image_cir_files_split[train_index]
            train_image_files.append(train_image_files_split)
            train_image_cir_files.append(train_image_cir_files_split)
            train_feat_cir_split = feat_cir_split[train_index]
            train_feat_cir.append(train_feat_cir_split)
            train_labels_split = labels_split[train_index]
            train_labels.append(train_labels_split)
            train_radio_split = radio_split[train_index]
            train_radio.append(train_radio_split)

            train_ids_split = ids_split[train_index]
            train_ids.append(train_ids_split)
            train_censoreds_split = censoreds_split[train_index]
            train_censoreds.append(train_censoreds_split)
            train_text_split = text_split[train_index]
            train_text.append(train_text_split)

            train_discrete_clinicl_split = discrete_clinicl_split[train_index]
            train_dis_clinics.append(train_discrete_clinicl_split)

            train_continuous_clinicl_split = continuous_clinicl_split[train_index]
            train_con_clinics.append(train_continuous_clinicl_split)

            train_cpx_clinicl_split = cpx_clinicl_split[train_index]
            train_cpx_clinics.append(train_cpx_clinicl_split)

            valid_image_files_split = image_files_split[valid_index]
            valid_image_cir_files_split = image_cir_files_split[valid_index]
            valid_image_files.append(valid_image_files_split)
            valid_image_cir_files.append(valid_image_cir_files_split)
            valid_feat_cir_split = feat_cir_split[valid_index]
            valid_feat_cir.append(valid_feat_cir_split)
            valid_labels_split = labels_split[valid_index]
            valid_labels.append(valid_labels_split)
            # valid_clinics_split = clinics_split[valid_index]
            # valid_clinics.append(valid_clinics_split)
            valid_ids_split = ids_split[valid_index]
            valid_ids.append(valid_ids_split)
            valid_censoreds_split = censoreds_split[valid_index]
            valid_censoreds.append(valid_censoreds_split)
            valid_text_split = text_split[valid_index]
            valid_text.append(valid_text_split)
            valid_radio_split = radio_split[valid_index]
            valid_radio.append(valid_radio_split)

            valid_discrete_clinicl_split = discrete_clinicl_split[valid_index]
            valid_dis_clinics.append(valid_discrete_clinicl_split)

            valid_continuous_clinicl_split = continuous_clinicl_split[valid_index]
            valid_con_clinics.append(valid_continuous_clinicl_split)

            valid_cpx_clinicl_split = cpx_clinicl_split[valid_index]
            valid_cpx_clinics.append(valid_cpx_clinicl_split)

    # Scaler = StandardScaler()
    # OHencoder = OneHotEncoder(categories='auto', sparse_output=False)
    train_text = []
    train_image_files = np.concatenate(train_image_files, axis=0)
    train_image_cir_files = np.concatenate(train_image_cir_files, axis=0)
    train_feat_cir = np.concatenate(train_feat_cir, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_con_clinics = np.concatenate(train_con_clinics, axis=0)
    train_dis_clinics = np.concatenate(train_dis_clinics, axis=0)
    train_cpx_clinics = np.concatenate(train_cpx_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    # train_text = np.concatenate(train_text, axis=0)
    train_radio = np.concatenate(train_radio, axis=0)[:, 0, :]
    # train_con_clinics_ts = Scaler.fit_transform(train_con_clinics)
    # train_clinics = np.concatenate([train_con_clinics_ts, train_dis_clinics], axis=-1)
    train_clinics = np.concatenate([train_con_clinics, train_dis_clinics], axis=-1)
    for i in range(len(train_clinics)):
        age = int(train_clinics[i][0])
        tumor_size = train_clinics[i][1]
        sex = train_clinics[i][2]
        if sex == 1:
            sex = 'male'
        else:
            sex = 'female'

        treatment_strategy = train_clinics[i][3]
        if treatment_strategy == 1:
            treatment_strategy = 'hepatectomy'
        elif treatment_strategy == 2:
            treatment_strategy = 'transcatheter arterial chemoembolization'
        elif treatment_strategy == 3:
            treatment_strategy = 'transcatheter arterial chemoembolization'
        elif treatment_strategy == 4:
            treatment_strategy = 'radiofrequency ablation'

        AFP_grade = train_clinics[i][4]
        if AFP_grade == 1:
            AFP_grade = 'are less than 20 ng/mL'
        elif AFP_grade == 2:
            AFP_grade = 'range between 20-400 ng/mL'
        elif AFP_grade == 3:
            AFP_grade = 'are greater than 400 ng/mL'

        medical_history = train_clinics[i][5]
        if medical_history == 0:
            medical_history = 'non-hepatitis'
        elif medical_history == 1:
            medical_history = 'hepatitis A'
        elif medical_history == 2:
            medical_history = 'hepatitis B'
        elif medical_history == 3:
            medical_history = 'hepatitis C'

        tumor_number = train_clinics[i][6]
        if tumor_number == 4:
            tumor_number = 'more than 3'
        else:
            tumor_number = int(tumor_number)

        BCLC_stage = train_clinics[i][7]
        if BCLC_stage == 1:
            BCLC_stage = 0
        elif BCLC_stage == 2:
            BCLC_stage = 'A'
        elif BCLC_stage == 3:
            BCLC_stage = 'B'

        CP_stage = train_clinics[i][-1]
        if CP_stage == 1:
            CP_stage = 'A'
        elif CP_stage == 2:
            CP_stage = 'B'
        elif CP_stage == 3:
            CP_stage = 'C'

        TBIL = train_cpx_clinics[i][0]
        if TBIL == -1:
            TBIL = 'None'

        ALB = train_cpx_clinics[i][1]
        if ALB == -1:
            ALB = 'None'

        ALT = train_cpx_clinics[i][2]
        if ALT == -1:
            ALT = 'None'

        INR = train_cpx_clinics[i][3]
        if INR == -1:
            INR = 'None'

        replacement_data = {
            'age': age,
            'gender': sex,
            'medical_history': medical_history,
            'tumor_number': tumor_number,
            'tumor_size': tumor_size,
            'AFP_grade': AFP_grade,
            'BCLC_stage': BCLC_stage,
            'treatment_strategy': treatment_strategy,
            'CP_stage': CP_stage
        }

        replacement_data2 = {
            'TBIL': TBIL,
            'ALB': ALB,
            'INR': INR,
            'ALT': ALT
        }
        formatted_text1 = ori_text1.format(**replacement_data)
        formatted_text2 = ori_text2.format(**replacement_data)
        if medical_history == 'non-hepatitis':
            formatted_text3 = ori_text3_2
        else:
            formatted_text3 = ori_text3_1.format(**replacement_data)
        formatted_text4 = ori_text4
        formatted_text5 = ori_text5
        formatted_text6 = ori_text6.format(**replacement_data)
        formatted_text7 = ori_text7.format(**replacement_data)
        formatted_text8 = ori_text8.format(**replacement_data)
        formatted_text9 = ori_text9
        if tumor_number == 1:
            formatted_text10 = ori_text10_2
        else:
            formatted_text10 = ori_text10_1.format(**replacement_data)
        formatted_text11 = ori_text11
        formatted_text12 = ori_text12.format(**replacement_data)
        formatted_text13 = ori_text13.format(**replacement_data)
        if INR == 'None':
            formatted_text14 = ori_text14_2
        else:
            formatted_text14 = ori_text14_1.format(**replacement_data2)
        if ALB == 'None':
            formatted_text15 = ori_text15_2
        else:
            formatted_text15 = ori_text15_1.format(**replacement_data2)
        if TBIL == 'None':
            formatted_text16 = ori_text16_2
        else:
            formatted_text16 = ori_text16_1.format(**replacement_data2)
        if ALT == 'None':
            formatted_text17 = ori_text17_2
        else:
            formatted_text17 = ori_text17_1.format(**replacement_data2)
        formatted_text = [formatted_text1, formatted_text2, formatted_text3, formatted_text4, formatted_text5,
                          formatted_text6, formatted_text7, formatted_text8, formatted_text9, formatted_text10,
                          formatted_text11, formatted_text12, formatted_text13, formatted_text14, formatted_text15,
                          formatted_text16, formatted_text17]
        train_text.append(formatted_text)
    # train_indices = list(range(len(train_image_files)))
    # train_selected_indices = random.sample(train_indices, len(train_image_files) - 40)
    # train_image_files = train_image_files[train_selected_indices]
    # train_image_cir_files = train_image_cir_files[train_selected_indices]
    # train_feat_cir = train_feat_cir[train_selected_indices]
    # train_labels = train_labels[train_selected_indices]
    # train_ids = train_ids[train_selected_indices]
    # train_censoreds = train_censoreds[train_selected_indices]
    # train_text = train_text[train_selected_indices]
    # train_clinics = train_clinics[train_selected_indices]
    # train_radio = train_radio[train_selected_indices]
    
    valid_text = []
    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_image_cir_files = np.concatenate(valid_image_cir_files, axis=0)
    valid_feat_cir = np.concatenate(valid_feat_cir, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    valid_con_clinics = np.concatenate(valid_con_clinics, axis=0)
    valid_dis_clinics = np.concatenate(valid_dis_clinics, axis=0)
    valid_cpx_clinics = np.concatenate(valid_cpx_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    # valid_text = np.concatenate(valid_text, axis=0)
    # valid_con_clinics_ts = Scaler.transform(valid_con_clinics)
    # valid_clinics = np.concatenate([valid_con_clinics_ts, valid_dis_clinics], axis=-1)
    valid_clinics = np.concatenate([valid_con_clinics, valid_dis_clinics], axis=-1)
    valid_radio = np.concatenate(valid_radio, axis=0)[:, 0, :]
    for i in range(len(valid_clinics)):
        age = int(valid_clinics[i][0])
        tumor_size = valid_clinics[i][1]
        sex = valid_clinics[i][2]
        if sex == 1:
            sex = 'male'
        else:
            sex = 'female'

        treatment_strategy = valid_clinics[i][3]
        if treatment_strategy == 1:
            treatment_strategy = 'hepatectomy'
        elif treatment_strategy == 2:
            treatment_strategy = 'transcatheter arterial chemoembolization'
        elif treatment_strategy == 3:
            treatment_strategy = 'transcatheter arterial chemoembolization'
        elif treatment_strategy == 4:
            treatment_strategy = 'radiofrequency ablation'

        AFP_grade = valid_clinics[i][4]
        if AFP_grade == 1:
            AFP_grade = 'are less than 20 ng/mL'
        elif AFP_grade == 2:
            AFP_grade = 'range between 20-400 ng/mL'
        elif AFP_grade == 3:
            AFP_grade = 'are greater than 400 ng/mL'

        medical_history = valid_clinics[i][5]
        if medical_history == 0:
            medical_history = 'non-hepatitis'
        elif medical_history == 1:
            medical_history = 'hepatitis A'
        elif medical_history == 2:
            medical_history = 'hepatitis B'
        elif medical_history == 3:
            medical_history = 'hepatitis C'

        tumor_number = valid_clinics[i][6]
        if tumor_number == 4:
            tumor_number = 'more than 3'
        else:
            tumor_number = int(tumor_number)

        BCLC_stage = valid_clinics[i][7]
        if BCLC_stage == 1:
            BCLC_stage = 0
        elif BCLC_stage == 2:
            BCLC_stage = 'A'
        elif BCLC_stage == 3:
            BCLC_stage = 'B'

        CP_stage = valid_clinics[i][-1]
        if CP_stage == 1:
            CP_stage = 'A'
        elif CP_stage == 2:
            CP_stage = 'B'
        elif CP_stage == 3:
            CP_stage = 'C'

        TBIL = valid_cpx_clinics[i][0]
        if TBIL == -1:
            TBIL = 'None'

        ALB = valid_cpx_clinics[i][1]
        if ALB == -1:
            ALB = 'None'

        ALT = valid_cpx_clinics[i][2]
        if ALT == -1:
            ALT = 'None'

        INR = valid_cpx_clinics[i][3]
        if INR == -1:
            INR = 'None'

        replacement_data = {
            'age': age,
            'gender': sex,
            'medical_history': medical_history,
            'tumor_number': tumor_number,
            'tumor_size': tumor_size,
            'AFP_grade': AFP_grade,
            'BCLC_stage': BCLC_stage,
            'treatment_strategy': treatment_strategy,
            'CP_stage': CP_stage
        }

        replacement_data2 = {
            'TBIL': TBIL,
            'ALB': ALB,
            'INR': INR,
            'ALT': ALT
        }
        formatted_text1 = ori_text1.format(**replacement_data)
        formatted_text2 = ori_text2.format(**replacement_data)
        if medical_history == 'non-hepatitis':
            formatted_text3 = ori_text3_2
        else:
            formatted_text3 = ori_text3_1.format(**replacement_data)
        formatted_text4 = ori_text4
        formatted_text5 = ori_text5
        formatted_text6 = ori_text6.format(**replacement_data)
        formatted_text7 = ori_text7.format(**replacement_data)
        formatted_text8 = ori_text8.format(**replacement_data)
        formatted_text9 = ori_text9
        if tumor_number == 1:
            formatted_text10 = ori_text10_2
        else:
            formatted_text10 = ori_text10_1.format(**replacement_data)
        formatted_text11 = ori_text11
        formatted_text12 = ori_text12.format(**replacement_data)
        formatted_text13 = ori_text13.format(**replacement_data)
        if INR == 'None':
            formatted_text14 = ori_text14_2
        else:
            formatted_text14 = ori_text14_1.format(**replacement_data2)
        if ALB == 'None':
            formatted_text15 = ori_text15_2
        else:
            formatted_text15 = ori_text15_1.format(**replacement_data2)
        if TBIL == 'None':
            formatted_text16 = ori_text16_2
        else:
            formatted_text16 = ori_text16_1.format(**replacement_data2)
        if ALT == 'None':
            formatted_text17 = ori_text17_2
        else:
            formatted_text17 = ori_text17_1.format(**replacement_data2)
        formatted_text = [formatted_text1, formatted_text2, formatted_text3, formatted_text4, formatted_text5,
                          formatted_text6, formatted_text7, formatted_text8, formatted_text9, formatted_text10,
                          formatted_text11, formatted_text12, formatted_text13, formatted_text14, formatted_text15,
                          formatted_text16, formatted_text17]
        valid_text.append(formatted_text)
    # valid_indices = list(range(len(valid_image_files)))
    # valid_selected_indices = random.sample(valid_indices, len(valid_image_files) - 20)
    # valid_image_files = valid_image_files[valid_selected_indices]
    # valid_image_cir_files = valid_image_cir_files[valid_selected_indices]
    # valid_feat_cir = valid_feat_cir[valid_selected_indices]
    # valid_labels = valid_labels[valid_selected_indices]
    # valid_ids = valid_ids[valid_selected_indices]
    # valid_censoreds = valid_censoreds[valid_selected_indices]
    # valid_text = valid_text[valid_selected_indices]
    # valid_clinics = valid_clinics[valid_selected_indices]
    # valid_radio = valid_radio[valid_selected_indices]

    test_text = []
    test_image_files = np.concatenate(test_image_files, axis=0)
    test_image_cir_files = np.concatenate(test_image_cir_files, axis=0)
    test_feat_cir = np.concatenate(test_feat_cir, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_con_clinics = np.concatenate(test_con_clinics, axis=0)
    test_dis_clinics = np.concatenate(test_dis_clinics, axis=0)
    test_cpx_clinics = np.concatenate(test_cpx_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    # test_text = np.concatenate(test_text, axis=0)
    # test_con_clinics_ts = Scaler.transform(test_con_clinics)
    # test_clinics = np.concatenate([test_con_clinics_ts, test_dis_clinics], axis=-1)
    test_clinics = np.concatenate([test_con_clinics, test_dis_clinics], axis=-1)
    test_radio = np.concatenate(test_radio, axis=0)[:, 0, :]
    for i in range(len(test_clinics)):
        age = int(test_clinics[i][0])
        tumor_size = test_clinics[i][1]
        sex = test_clinics[i][2]
        if sex == 1:
            sex = 'male'
        else:
            sex = 'female'

        treatment_strategy = test_clinics[i][3]
        if treatment_strategy == 1:
            treatment_strategy = 'hepatectomy'
        elif treatment_strategy == 2:
            treatment_strategy = 'transcatheter arterial chemoembolization'
        elif treatment_strategy == 3:
            treatment_strategy = 'transcatheter arterial chemoembolization'
        elif treatment_strategy == 4:
            treatment_strategy = 'radiofrequency ablation'

        AFP_grade = test_clinics[i][4]
        if AFP_grade == 1:
            AFP_grade = 'are less than 20 ng/mL'
        elif AFP_grade == 2:
            AFP_grade = 'range between 20-400 ng/mL'
        elif AFP_grade == 3:
            AFP_grade = 'are greater than 400 ng/mL'

        medical_history = test_clinics[i][5]
        if medical_history == 0:
            medical_history = 'non-hepatitis'
        elif medical_history == 1:
            medical_history = 'hepatitis A'
        elif medical_history == 2:
            medical_history = 'hepatitis B'
        elif medical_history == 3:
            medical_history = 'hepatitis C'

        tumor_number = test_clinics[i][6]
        if tumor_number == 4:
            tumor_number = 'more than 3'
        else:
            tumor_number = int(tumor_number)

        BCLC_stage = test_clinics[i][7]
        if BCLC_stage == 1:
            BCLC_stage = 0
        elif BCLC_stage == 2:
            BCLC_stage = 'A'
        elif BCLC_stage == 3:
            BCLC_stage = 'B'

        CP_stage = test_clinics[i][-1]
        if CP_stage == 1:
            CP_stage = 'A'
        elif CP_stage == 2:
            CP_stage = 'B'
        elif CP_stage == 3:
            CP_stage = 'C'

        TBIL = test_cpx_clinics[i][0]
        if TBIL == -1:
            TBIL = 'None'

        ALB = test_cpx_clinics[i][1]
        if ALB == -1:
            ALB = 'None'

        ALT = test_cpx_clinics[i][2]
        if ALT == -1:
            ALT = 'None'

        INR = test_cpx_clinics[i][3]
        if INR == -1:
            INR = 'None'

        replacement_data = {
            'age': age,
            'gender': sex,
            'medical_history': medical_history,
            'tumor_number': tumor_number,
            'tumor_size': tumor_size,
            'AFP_grade': AFP_grade,
            'BCLC_stage': BCLC_stage,
            'treatment_strategy': treatment_strategy,
            'CP_stage': CP_stage
        }

        replacement_data2 = {
            'TBIL': TBIL,
            'ALB': ALB,
            'INR': INR,
            'ALT': ALT
        }
        formatted_text1 = ori_text1.format(**replacement_data)
        formatted_text2 = ori_text2.format(**replacement_data)
        if medical_history == 'non-hepatitis':
            formatted_text3 = ori_text3_2
        else:
            formatted_text3 = ori_text3_1.format(**replacement_data)
        formatted_text4 = ori_text4
        formatted_text5 = ori_text5
        formatted_text6 = ori_text6.format(**replacement_data)
        formatted_text7 = ori_text7.format(**replacement_data)
        formatted_text8 = ori_text8.format(**replacement_data)
        formatted_text9 = ori_text9
        if tumor_number == 1:
            formatted_text10 = ori_text10_2
        else:
            formatted_text10 = ori_text10_1.format(**replacement_data)
        formatted_text11 = ori_text11
        formatted_text12 = ori_text12.format(**replacement_data)
        formatted_text13 = ori_text13.format(**replacement_data)
        if INR == 'None':
            formatted_text14 = ori_text14_2
        else:
            formatted_text14 = ori_text14_1.format(**replacement_data2)
        if ALB == 'None':
            formatted_text15 = ori_text15_2
        else:
            formatted_text15 = ori_text15_1.format(**replacement_data2)
        if TBIL == 'None':
            formatted_text16 = ori_text16_2
        else:
            formatted_text16 = ori_text16_1.format(**replacement_data2)
        if ALT == 'None':
            formatted_text17 = ori_text17_2
        else:
            formatted_text17 = ori_text17_1.format(**replacement_data2)
        formatted_text = [formatted_text1, formatted_text2, formatted_text3, formatted_text4, formatted_text5,
                          formatted_text6, formatted_text7, formatted_text8, formatted_text9, formatted_text10,
                          formatted_text11, formatted_text12, formatted_text13, formatted_text14, formatted_text15,
                          formatted_text16, formatted_text17]
        test_text.append(formatted_text)


    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)

    # train_text = train_text.tolist()
    # valid_text = valid_text.tolist()
    # test_text = test_text.tolist()

    train_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"], ),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        RandRotated(keys=["image", "liver", "tumor", "cir_liver"], range_x=15, prob=0.5, keep_size=True),
        RandFlipd(keys=["image", "liver", "tumor", "cir_liver"], spatial_axis=0, prob=0.2),
        RandZoomd(keys=["image", "liver", "tumor", "cir_liver"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    train_dataset = LiverDataset_disentangle_cir_feat_cli_radio(train_image_files, train_image_cir_files, train_clinics,
                                                          train_feat_cir, train_labels, train_censoreds,
                                                          train_durations_discrete, train_events, train_ids, train_text,
                                                          train_radio, train_transforms)
    valid_dataset = LiverDataset_disentangle_cir_feat_cli_radio(valid_image_files, valid_image_cir_files, valid_clinics,
                                                          valid_feat_cir,
                                                          valid_labels, valid_censoreds, valid_durations_discrete,
                                                          valid_events, valid_ids, valid_text, valid_radio,
                                                                val_transforms,)
    test_dataset = LiverDataset_disentangle_cir_feat_cli_radio(test_image_files, test_image_cir_files, test_clinics,
                                                         test_feat_cir,
                                                         test_labels, test_censoreds, test_durations_discrete,
                                                         test_events, test_ids, test_text, test_radio,
                                                               val_transforms)

    # train_infos = np.concatenate([train_ids[:, None], train_image_files[:, None], train_labels[:, None],
    #                          train_censoreds[:, None], train_clinics, np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], valid_image_files[:, None], valid_labels[:, None],
    #                               valid_censoreds[:, None], valid_clinics, np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], test_image_files[:, None], test_labels[:, None],
    #                               test_censoreds[:, None], test_clinics, np.array(test_text)[:, None]], axis=1)
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_infos.csv', train_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_infos.csv', valid_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_infos.csv', test_infos, fmt='%s',
    #            delimiter=',')
    # train_infos = np.concatenate([train_ids[:, None], np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], np.array(test_text)[:, None]], axis=1)
    # train_infos = pd.DataFrame(train_infos, columns=['ids', 'text'])
    # valid_infos = pd.DataFrame(valid_infos, columns=['ids', 'text'])
    # test_infos = pd.DataFrame(test_infos, columns=['ids', 'text'])
    # train_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_text.xlsx', index=False)
    # valid_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_text.xlsx', index=False)
    # test_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_text.xlsx', index=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=2)
    x_image_files, x_clinics, x_cir_feat, x_labels, x_censoreds, x_durations_discrete, \
        x_events, x_ids, x_text, x_radio = next(iter(train_loader))
    # for i, batch_data in enumerate(train_loader):
    #     s = 0
    # train_sample = next(iter(train_loader))
    return train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader

def create_dataset_ext_week_persudo_pair_addcirf_fixid2r_radio2(matched_feature_file1, matched_feature_file2,
                                                         matched_ids_file1, matched_ids_file2,
                                                         matched_censor_file1, matched_censor_file2, matched_surv_file1,
                                                         matched_surv_file2,
                                                         root_dir, fold, num_durations=6, batch_size=6,
                                                         val_batch_size=4, seed=1998):
    set_seeds(seed)
    features_cp0 = np.load(matched_feature_file1)
    features_cp1 = np.load(matched_feature_file2)[:, 0, ...]
    ids_cp0 = np.load(matched_ids_file1)
    ids_cp1 = np.load(matched_ids_file2)[:, 0, ...]
    censors_cp0 = np.load(matched_censor_file1)
    censors_cp1 = np.load(matched_censor_file2)
    survs_cp0 = np.load(matched_surv_file1)
    survs_cp1 = np.load(matched_surv_file2)

    match_ids_sum = np.concatenate([ids_cp0, ids_cp1])
    match_features_sum = np.concatenate([features_cp0, features_cp1])
    match_censors_sum = np.concatenate([censors_cp0, censors_cp1])
    match_survs_sum = np.concatenate([survs_cp0, survs_cp1])
    # print_config()
    data_dir1 = root_dir
    hospital_names = (
        sorted([x for x in os.listdir(data_dir1) if os.path.isdir(os.path.join(data_dir1, x))]))
    image_files = [[os.path.join(data_dir1, class_name, x)
                    for x in os.listdir(os.path.join(data_dir1, class_name))]
                   for class_name in hospital_names]

    image_files_need = []
    image_cir_files_need = []
    feat_cir_need = []
    info = pd.read_excel(r"/home/wangtao/project/Cirros_HCC/data/table/PFS_ZYJ_20240319_PD_CP_text.xlsx")
    ids = info['序号']
    hospitals = info['病例来源']
    radio_info = np.load(r'/home/wangtao/project/Survival/radio_filter.npy')

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']

    list1 = ['AFP3', 'hepatitis3', 'BCLC3', 'max_lession', 'AFP1', 'BCLC1', 'multi4']
    list2 = ['age', 'max_lession', 'sex1', 'sex2', 'treat1', 'treat2', 'treat3', 'treat4', 'AFP1', 'AFP2',
             'AFP3', 'hepatitis1', 'hepatitis2', 'hepatitis3', 'hepatitis4', 'multi1', 'multi2', 'multi3',
             'multi4', 'BCLC1', 'BCLC2', 'BCLC3', 'cp1', 'cp2', 'cp3']

    # 找出 list1 中每个元素在 list2 中的索引位置
    index_positions = [list2.index(item) if item in list2 else None for item in list1]
    index_positions = np.array(index_positions)
    index_positions = np.sort(index_positions)

    continuous_clinicl = info.iloc[:, [5, 10]]
    discrete_clinicl = info.iloc[:, [6, 7, 8, 9, 11, 12, 23]]
    # discrete_clinicl = info.iloc[:, [7, 8, 9]]
    continuous_clinicl = np.asarray(continuous_clinicl)
    discrete_clinicl = np.asarray(discrete_clinicl)
    discrete_clinicl[np.where(np.isnan(discrete_clinicl) == True)] = 1
    OHencoder = OneHotEncoder(categories='auto', sparse_output=False)
    discrete_clinicl = OHencoder.fit_transform(discrete_clinicl)

    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    radio_sum = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        continuous_clinicl_hospital = []
        discrete_clinicl_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        image_cir_files_need_hospital = []
        feat_cir_need_hospital = []
        radio_hospital = []
        for image_file in image_files_hospital:
            ID = image_file.split('/')[-1].split('.')[0]
            center_now = image_file.split('/')[-2]
            pos = np.where(ids.astype(str) == ID)
            pos_match = np.where(match_ids_sum[:, -1] == center_now + '_' + ID)
            if len(pos[0]) != 0:
                cp_now = cp_labels[pos][0]
                if cp_now != 3 and np.isnan(cp_now) == False:
                    pos_radio = np.where(radio_info[:, 0].astype(int).astype(str) == ID)
                    if len(pos_radio[0]) != 0:
                        feat_radio = radio_info[pos_radio[0], 1:]
                    else:
                        feat_radio = np.zeros([1, 22])
                    id_cir_now = match_ids_sum[pos_match[0], 0][0]
                    feat_cir_now = match_features_sum[pos_match[0], 0][0][None, :]
                    image_cir_file = os.path.join(r'/home/wangtao/project/Survival/data_cir/preprocess/Liver/data',
                                                  id_cir_now.split('_')[0], id_cir_now + '.nii.gz')
                    surv_label = surv_labels[pos][0]
                    if_censored = if_censoreds[pos][0]
                    text_now = gpt_text[pos][0]

                    continuous_clinicl_now = continuous_clinicl[pos][0]
                    discrete_clinicl_now = discrete_clinicl[pos][0]
                    image_files_need_hospital.append(image_file)
                    ids_hospital.append(ID)
                    text_hospital.append(text_now)
                    labels_hospital.append(surv_label)
                    censored_hospital.append(if_censored)
                    image_cir_files_need_hospital.append(image_cir_file)
                    feat_cir_need_hospital.append(feat_cir_now)
                    continuous_clinicl_hospital.append(continuous_clinicl_now)
                    discrete_clinicl_hospital.append(discrete_clinicl_now)
                    radio_hospital.append(feat_radio)

        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        image_cir_files_need_hospital = np.array(image_cir_files_need_hospital)
        feat_cir_need_hospital = np.concatenate(feat_cir_need_hospital)
        continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        radio_hospital = np.array(radio_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        image_cir_files_need.append(image_cir_files_need_hospital)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)
        feat_cir_need.append(feat_cir_need_hospital)
        continuous_clinicl_sum.append(continuous_clinicl_hospital)
        discrete_clinicl_sum.append(discrete_clinicl_hospital)
        radio_sum.append(radio_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1998)

    train_val_image_files = []
    train_val_image_cir_files = []
    train_val_feat_cir = []
    train_val_labels = []
    train_val_con_clinics = []
    train_val_dis_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []
    train_val_radio = []

    train_image_files = []
    train_image_cir_files = []
    train_feat_cir = []
    train_labels = []
    train_con_clinics = []
    train_dis_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []
    train_radio = []

    valid_image_files = []
    valid_image_cir_files = []
    valid_feat_cir = []
    valid_labels = []
    valid_con_clinics = []
    valid_dis_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []
    valid_radio = []

    test_image_files = []
    test_image_cir_files = []
    test_feat_cir = []
    test_labels = []
    test_con_clinics = []
    test_dis_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []
    test_radio = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        image_cir_files_split = np.array(image_cir_files_need[hospital])
        feat_cir_split = feat_cir_need[hospital]
        labels_split = labels[hospital]
        # continuous_clinics_split = continuous_clinicl_sum[hospital]
        # discrete_clinics_split = discrete_clinicl_sum[hospital]
        # clinics_split = np.concatenate([continuous_clinics_split, discrete_clinics_split], axis=-1)
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        continuous_clinicl_split = continuous_clinicl_sum[hospital]
        radio_split = radio_sum[hospital]
        discrete_clinicl_split = discrete_clinicl_sum[hospital]
        # if hospital_name_split in ['Center2', 'Center3', 'Center4', 'Center8', 'Center9', 'Center5', 'Center11']:  # 0420_1
        if hospital_name_split not in ['Center5', 'Center9']:
            train_val_image_files.append(image_files_split)
            train_val_image_cir_files.append(image_cir_files_split)
            train_val_labels.append(labels_split)
            # train_val_clinics.append(clinics_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
            train_val_feat_cir.append(feat_cir_split)
            train_val_dis_clinics.append(discrete_clinicl_split)
            train_val_con_clinics.append(continuous_clinicl_split)
            train_val_radio.append(radio_split)
        else:
            test_image_files.append(image_files_split)
            test_image_cir_files.append(image_cir_files_split)
            test_labels.append(labels_split)
            # test_clinics.append(clinics_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)
            test_feat_cir.append(feat_cir_split)
            test_dis_clinics.append(discrete_clinicl_split)
            test_con_clinics.append(continuous_clinicl_split)
            test_radio.append(radio_split)

    for hospital in range(len(train_val_image_files)):
        image_files_split = np.array(train_val_image_files[hospital])
        image_cir_files_split = np.array(train_val_image_cir_files[hospital])
        labels_split = train_val_labels[hospital]
        feat_cir_split = train_val_feat_cir[hospital]
        # clinics_split = train_val_clinics[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        censoreds_split = train_val_censoreds[hospital]
        continuous_clinicl_split = train_val_con_clinics[hospital]
        discrete_clinicl_split = train_val_dis_clinics[hospital]
        radio_split = train_val_radio[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_image_cir_files.append(image_cir_files_split)
            train_labels.append(labels_split)
            train_dis_clinics.append(discrete_clinicl_split)
            train_con_clinics.append(continuous_clinicl_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
            train_feat_cir.append(feat_cir_split)
            train_radio.append(radio_split)
        else:
            # for train_index, valid_index in kf.split(image_files_split, censoreds_split):
            #     _fold = _fold + 1
            #     if _fold == fold:
            train_index = np.loadtxt(
                r'/home/wangtao/project/Survival/split_ids/train_index{}_fold{}_nt59.csv'.format(hospital, fold),
                delimiter=',', dtype=str)
            train_index = train_index.astype(int)
            valid_index = np.loadtxt(
                r'/home/wangtao/project/Survival/split_ids/valid_index{}_fold{}_nt59.csv'.format(hospital, fold),
                delimiter=',', dtype=str)
            valid_index = valid_index.astype(int)
            train_image_files_split = image_files_split[train_index]
            train_image_cir_files_split = image_cir_files_split[train_index]
            train_image_files.append(train_image_files_split)
            train_image_cir_files.append(train_image_cir_files_split)
            train_feat_cir_split = feat_cir_split[train_index]
            train_feat_cir.append(train_feat_cir_split)
            train_labels_split = labels_split[train_index]
            train_labels.append(train_labels_split)
            train_radio_split = radio_split[train_index]
            train_radio.append(train_radio_split)

            train_ids_split = ids_split[train_index]
            train_ids.append(train_ids_split)
            train_censoreds_split = censoreds_split[train_index]
            train_censoreds.append(train_censoreds_split)
            train_text_split = text_split[train_index]
            train_text.append(train_text_split)

            train_discrete_clinicl_split = discrete_clinicl_split[train_index]
            train_dis_clinics.append(train_discrete_clinicl_split)

            train_continuous_clinicl_split = continuous_clinicl_split[train_index]
            train_con_clinics.append(train_continuous_clinicl_split)

            valid_image_files_split = image_files_split[valid_index]
            valid_image_cir_files_split = image_cir_files_split[valid_index]
            valid_image_files.append(valid_image_files_split)
            valid_image_cir_files.append(valid_image_cir_files_split)
            valid_feat_cir_split = feat_cir_split[valid_index]
            valid_feat_cir.append(valid_feat_cir_split)
            valid_labels_split = labels_split[valid_index]
            valid_labels.append(valid_labels_split)
            # valid_clinics_split = clinics_split[valid_index]
            # valid_clinics.append(valid_clinics_split)
            valid_ids_split = ids_split[valid_index]
            valid_ids.append(valid_ids_split)
            valid_censoreds_split = censoreds_split[valid_index]
            valid_censoreds.append(valid_censoreds_split)
            valid_text_split = text_split[valid_index]
            valid_text.append(valid_text_split)
            valid_radio_split = radio_split[valid_index]
            valid_radio.append(valid_radio_split)

            valid_discrete_clinicl_split = discrete_clinicl_split[valid_index]
            valid_dis_clinics.append(valid_discrete_clinicl_split)

            valid_continuous_clinicl_split = continuous_clinicl_split[valid_index]
            valid_con_clinics.append(valid_continuous_clinicl_split)

    Scaler = StandardScaler()
    # OHencoder = OneHotEncoder(categories='auto', sparse_output=False)

    train_image_files = np.concatenate(train_image_files, axis=0)
    train_image_cir_files = np.concatenate(train_image_cir_files, axis=0)
    train_feat_cir = np.concatenate(train_feat_cir, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_con_clinics = np.concatenate(train_con_clinics, axis=0)
    train_dis_clinics = np.concatenate(train_dis_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    train_text = np.concatenate(train_text, axis=0)
    train_radio = np.concatenate(train_radio, axis=0)[:, 0, :]
    train_con_clinics_ts = Scaler.fit_transform(train_con_clinics)
    # train_dis_clinics_ts = OHencoder.fit_transform(train_dis_clinics)
    train_clinics = np.concatenate([train_con_clinics_ts, train_dis_clinics], axis=-1)


    # ### train_image_files, train_image_cir_files, train_clinics, train_feat_cir,
    #                                                   train_labels, train_censoreds, train_durations_discrete,
    #                                                   train_events, train_ids, train_text, train_transforms
    train_indices = list(range(len(train_image_files)))
    train_selected_indices = random.sample(train_indices, len(train_image_files) - 40)
    train_image_files = train_image_files[train_selected_indices]
    train_image_cir_files = train_image_cir_files[train_selected_indices]
    train_feat_cir = train_feat_cir[train_selected_indices]
    train_labels = train_labels[train_selected_indices]
    train_ids = train_ids[train_selected_indices]
    train_censoreds = train_censoreds[train_selected_indices]
    train_text = train_text[train_selected_indices]
    train_clinics = train_clinics[train_selected_indices]
    train_radio = train_radio[train_selected_indices]

    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_image_cir_files = np.concatenate(valid_image_cir_files, axis=0)
    valid_feat_cir = np.concatenate(valid_feat_cir, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    valid_con_clinics = np.concatenate(valid_con_clinics, axis=0)
    valid_dis_clinics = np.concatenate(valid_dis_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    valid_text = np.concatenate(valid_text, axis=0)
    valid_con_clinics_ts = Scaler.transform(valid_con_clinics)
    # valid_dis_clinics_ts = OHencoder.fit_transform(valid_dis_clinics)
    valid_clinics = np.concatenate([valid_con_clinics_ts, valid_dis_clinics], axis=-1)
    valid_radio = np.concatenate(valid_radio, axis=0)[:, 0, :]

    valid_indices = list(range(len(valid_image_files)))
    valid_selected_indices = random.sample(valid_indices, len(valid_image_files) - 20)
    valid_image_files = valid_image_files[valid_selected_indices]
    valid_image_cir_files = valid_image_cir_files[valid_selected_indices]
    valid_feat_cir = valid_feat_cir[valid_selected_indices]
    valid_labels = valid_labels[valid_selected_indices]
    valid_ids = valid_ids[valid_selected_indices]
    valid_censoreds = valid_censoreds[valid_selected_indices]
    valid_text = valid_text[valid_selected_indices]
    valid_clinics = valid_clinics[valid_selected_indices]
    valid_radio = valid_radio[valid_selected_indices]

    test_image_files = np.concatenate(test_image_files, axis=0)
    test_image_cir_files = np.concatenate(test_image_cir_files, axis=0)
    test_feat_cir = np.concatenate(test_feat_cir, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_con_clinics = np.concatenate(test_con_clinics, axis=0)
    test_dis_clinics = np.concatenate(test_dis_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    test_text = np.concatenate(test_text, axis=0)
    test_con_clinics_ts = Scaler.transform(test_con_clinics)
    # test_dis_clinics_ts = OHencoder.fit_transform(test_dis_clinics)
    test_clinics = np.concatenate([test_con_clinics_ts, test_dis_clinics], axis=-1)
    test_radio = np.concatenate(test_radio, axis=0)[:, 0, :]

    train_clinics = train_clinics[:, index_positions]
    valid_clinics = valid_clinics[:, index_positions]
    test_clinics = test_clinics[:, index_positions]


    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)

    train_text = train_text.tolist()
    valid_text = valid_text.tolist()
    test_text = test_text.tolist()

    train_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"], ),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        RandRotated(keys=["image", "liver", "tumor", "cir_liver"], range_x=15, prob=0.5, keep_size=True),
        RandFlipd(keys=["image", "liver", "tumor", "cir_liver"], spatial_axis=0, prob=0.2),
        RandZoomd(keys=["image", "liver", "tumor", "cir_liver"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    train_dataset = LiverDataset_disentangle_cir_feat_cli_radio(train_image_files, train_image_cir_files, train_clinics,
                                                          train_feat_cir, train_labels, train_censoreds,
                                                          train_durations_discrete, train_events, train_ids, train_text,
                                                          train_radio, train_transforms)
    valid_dataset = LiverDataset_disentangle_cir_feat_cli_radio(valid_image_files, valid_image_cir_files, valid_clinics,
                                                          valid_feat_cir,
                                                          valid_labels, valid_censoreds, valid_durations_discrete,
                                                          valid_events, valid_ids, valid_text, valid_radio,
                                                                val_transforms,)
    test_dataset = LiverDataset_disentangle_cir_feat_cli_radio(test_image_files, test_image_cir_files, test_clinics,
                                                         test_feat_cir,
                                                         test_labels, test_censoreds, test_durations_discrete,
                                                         test_events, test_ids, test_text, test_radio,
                                                               val_transforms)

    # train_infos = np.concatenate([train_ids[:, None], train_image_files[:, None], train_labels[:, None],
    #                          train_censoreds[:, None], train_clinics, np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], valid_image_files[:, None], valid_labels[:, None],
    #                               valid_censoreds[:, None], valid_clinics, np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], test_image_files[:, None], test_labels[:, None],
    #                               test_censoreds[:, None], test_clinics, np.array(test_text)[:, None]], axis=1)
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_infos.csv', train_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_infos.csv', valid_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_infos.csv', test_infos, fmt='%s',
    #            delimiter=',')
    # train_infos = np.concatenate([train_ids[:, None], np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], np.array(test_text)[:, None]], axis=1)
    # train_infos = pd.DataFrame(train_infos, columns=['ids', 'text'])
    # valid_infos = pd.DataFrame(valid_infos, columns=['ids', 'text'])
    # test_infos = pd.DataFrame(test_infos, columns=['ids', 'text'])
    # train_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_text.xlsx', index=False)
    # valid_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_text.xlsx', index=False)
    # test_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_text.xlsx', index=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=2)
    x_image_files, x_clinics, x_cir_feat, x_labels, x_censoreds, x_durations_discrete, \
        x_events, x_ids, x_text, x_radio = next(iter(train_loader))
    # for i, batch_data in enumerate(train_loader):
    #     s = 0
    # train_sample = next(iter(train_loader))
    return train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader

def create_dataset_ext_week_persudo_pair_radiomics(matched_feature_file1, matched_feature_file2,
                                                         matched_ids_file1, matched_ids_file2,
                                                         matched_censor_file1, matched_censor_file2, matched_surv_file1,
                                                         matched_surv_file2,
                                                         root_dir, fold, num_durations=6, batch_size=6,
                                                         val_batch_size=4, seed=1998):
    set_seeds(seed)
    features_cp0 = np.load(matched_feature_file1)
    features_cp1 = np.load(matched_feature_file2)[:, 0, ...]
    ids_cp0 = np.load(matched_ids_file1)
    ids_cp1 = np.load(matched_ids_file2)[:, 0, ...]
    censors_cp0 = np.load(matched_censor_file1)
    censors_cp1 = np.load(matched_censor_file2)
    survs_cp0 = np.load(matched_surv_file1)
    survs_cp1 = np.load(matched_surv_file2)

    match_ids_sum = np.concatenate([ids_cp0, ids_cp1])
    match_features_sum = np.concatenate([features_cp0, features_cp1])
    match_censors_sum = np.concatenate([censors_cp0, censors_cp1])
    match_survs_sum = np.concatenate([survs_cp0, survs_cp1])
    # print_config()
    data_dir1 = root_dir
    hospital_names = (
        sorted([x for x in os.listdir(data_dir1) if os.path.isdir(os.path.join(data_dir1, x))]))
    image_files = [[os.path.join(data_dir1, class_name, x)
                    for x in os.listdir(os.path.join(data_dir1, class_name))]
                   for class_name in hospital_names]

    image_files_need = []
    image_cir_files_need = []
    feat_cir_need = []
    info = pd.read_excel(r"/home/wangtao/project/Cirros_HCC/data/table/PFS_ZYJ_20240319_PD_CP_text.xlsx")
    radiomics_hcc = np.loadtxt(r'/home/wangtao/project/Survival/radiomics_feat/hcc_feat.csv',
                               dtype=str, delimiter=',')[1:]
    radiomics_title = np.loadtxt(r'/home/wangtao/project/Survival/radiomics_feat/hcc_feat.csv',
                               dtype=str, delimiter=',')[:1]
    radiomics_cir = np.loadtxt(r'/home/wangtao/project/Survival/radiomics_feat/cir_feat.csv',
                               dtype=str, delimiter=',')[1:]

    ids_radiomics_hcc = radiomics_hcc[:, 0]
    ids_radiomics_cir = radiomics_cir[:, 0]

    ids = info['序号']
    hospitals = info['病例来源']

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']
    clinical_title = ['age', 'max_lession', 'sex1', 'sex2', 'treat1', 'treat2', 'treat3', 'treat4', 'AFP1', 'AFP2',
                      'AFP3', 'hepatitis1', 'hepatitis2', 'hepatitis3', 'hepatitis4', 'multi1', 'multi2', 'multi3',
                      'multi4', 'BCLC1', 'BCLC2', 'BCLC3', 'cp1', 'cp2', 'cp3']
    continuous_clinicl = info.iloc[:, [5, 10]]
    discrete_clinicl = info.iloc[:, [6, 7, 8, 9, 11, 12, 23]]
    # discrete_clinicl = info.iloc[:, [7, 8, 9]]
    continuous_clinicl = np.asarray(continuous_clinicl)
    discrete_clinicl = np.asarray(discrete_clinicl)
    discrete_clinicl[np.where(np.isnan(discrete_clinicl) == True)] = 1
    OHencoder = OneHotEncoder(categories='auto', sparse_output=False)
    discrete_clinicl = OHencoder.fit_transform(discrete_clinicl)

    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    radiomics_sum = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        continuous_clinicl_hospital = []
        discrete_clinicl_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        radiomics_hospital = []
        image_cir_files_need_hospital = []
        feat_cir_need_hospital = []
        for image_file in image_files_hospital:
            ID = image_file.split('/')[-1].split('.')[0]
            center_now = image_file.split('/')[-2]
            pos = np.where(ids.astype(str) == ID)
            pos_radiomics_hcc = np.where(ids_radiomics_hcc.astype(str) == ID + '.nii.gz')
            pos_match = np.where(match_ids_sum[:, -1] == center_now + '_' + ID)
            if len(pos[0]) != 0:
                cp_now = cp_labels[pos][0]
                if cp_now != 3 and np.isnan(cp_now) == False:
                    id_cir_now = match_ids_sum[pos_match[0], 0][0]
                    pos_radiomics_cir = np.where(ids_radiomics_cir.astype(str) == id_cir_now + '.nii.gz')
                    if len(pos_radiomics_cir[0]) != 0 and len(pos_radiomics_hcc[0]) != 0:
                        radiomics_feat_hcc_now = radiomics_hcc[pos_radiomics_hcc][:, 1:]
                        radiomics_feat_cir_now = radiomics_cir[pos_radiomics_cir][:, 1:]
                        radiomics_feat_now = np.concatenate([radiomics_feat_cir_now, radiomics_feat_hcc_now], axis=-1)
                        radiomics_hospital.append(radiomics_feat_now)


                        feat_cir_now = match_features_sum[pos_match[0], 0][0][None, :]
                        image_cir_file = os.path.join(r'/home/wangtao/project/Survival/data_cir/preprocess/Liver/data',
                                                      id_cir_now.split('_')[0], id_cir_now + '.nii.gz')
                        surv_label = surv_labels[pos][0]
                        if_censored = if_censoreds[pos][0]
                        text_now = gpt_text[pos][0]

                        continuous_clinicl_now = continuous_clinicl[pos][0]
                        discrete_clinicl_now = discrete_clinicl[pos][0]
                        image_files_need_hospital.append(image_file)
                        ids_hospital.append(ID)
                        text_hospital.append(text_now)
                        labels_hospital.append(surv_label)
                        censored_hospital.append(if_censored)
                        image_cir_files_need_hospital.append(image_cir_file)
                        feat_cir_need_hospital.append(feat_cir_now)
                        continuous_clinicl_hospital.append(continuous_clinicl_now)
                        discrete_clinicl_hospital.append(discrete_clinicl_now)

        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        image_cir_files_need_hospital = np.array(image_cir_files_need_hospital)
        feat_cir_need_hospital = np.concatenate(feat_cir_need_hospital)
        continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        radiomics_hospital = np.array(radiomics_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        image_cir_files_need.append(image_cir_files_need_hospital)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)
        feat_cir_need.append(feat_cir_need_hospital)
        continuous_clinicl_sum.append(continuous_clinicl_hospital)
        discrete_clinicl_sum.append(discrete_clinicl_hospital)
        radiomics_sum.append(radiomics_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1998)

    train_val_image_files = []
    train_val_image_cir_files = []
    train_val_feat_cir = []
    train_val_labels = []
    train_val_con_clinics = []
    train_val_dis_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []
    train_val_radiomics = []

    train_image_files = []
    train_image_cir_files = []
    train_feat_cir = []
    train_labels = []
    train_con_clinics = []
    train_dis_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []
    train_radiomics = []

    valid_image_files = []
    valid_image_cir_files = []
    valid_feat_cir = []
    valid_labels = []
    valid_con_clinics = []
    valid_dis_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []
    valid_radiomics = []

    test_image_files = []
    test_image_cir_files = []
    test_feat_cir = []
    test_labels = []
    test_con_clinics = []
    test_dis_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []
    test_radiomics = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        image_cir_files_split = np.array(image_cir_files_need[hospital])
        feat_cir_split = feat_cir_need[hospital]
        labels_split = labels[hospital]
        radiomics_split = radiomics_sum[hospital]
        # continuous_clinics_split = continuous_clinicl_sum[hospital]
        # discrete_clinics_split = discrete_clinicl_sum[hospital]
        # clinics_split = np.concatenate([continuous_clinics_split, discrete_clinics_split], axis=-1)
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        continuous_clinicl_split = continuous_clinicl_sum[hospital]
        discrete_clinicl_split = discrete_clinicl_sum[hospital]
        # if hospital_name_split in ['Center2', 'Center3', 'Center4', 'Center8', 'Center9', 'Center5', 'Center11']:  # 0420_1
        if hospital_name_split not in ['Center5', 'Center9']:
            train_val_image_files.append(image_files_split)
            train_val_image_cir_files.append(image_cir_files_split)
            train_val_labels.append(labels_split)
            train_val_radiomics.append(radiomics_split)
            # train_val_clinics.append(clinics_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
            train_val_feat_cir.append(feat_cir_split)
            train_val_dis_clinics.append(discrete_clinicl_split)
            train_val_con_clinics.append(continuous_clinicl_split)
        else:
            test_image_files.append(image_files_split)
            test_image_cir_files.append(image_cir_files_split)
            test_labels.append(labels_split)
            test_radiomics.append(radiomics_split)
            # test_clinics.append(clinics_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)
            test_feat_cir.append(feat_cir_split)
            test_dis_clinics.append(discrete_clinicl_split)
            test_con_clinics.append(continuous_clinicl_split)

    for hospital in range(len(train_val_image_files)):
        image_files_split = np.array(train_val_image_files[hospital])
        image_cir_files_split = np.array(train_val_image_cir_files[hospital])
        labels_split = train_val_labels[hospital]
        feat_cir_split = train_val_feat_cir[hospital]
        # clinics_split = train_val_clinics[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        radiomics_split = train_val_radiomics[hospital]
        censoreds_split = train_val_censoreds[hospital]
        continuous_clinicl_split = train_val_con_clinics[hospital]
        discrete_clinicl_split = train_val_dis_clinics[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_image_cir_files.append(image_cir_files_split)
            train_labels.append(labels_split)
            train_dis_clinics.append(discrete_clinicl_split)
            train_con_clinics.append(continuous_clinicl_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
            train_feat_cir.append(feat_cir_split)
            train_radiomics.append(radiomics_split)
        else:
            for train_index, valid_index in kf.split(image_files_split, censoreds_split):
                _fold = _fold + 1
                if _fold == fold:
                    # train_index = np.loadtxt(
                    #     r'/home/wangtao/project/Survival/split_ids/train_index{}_fold{}_nt59.csv'.format(hospital, fold),
                    #     delimiter=',', dtype=str)
                    # train_index = train_index.astype(int)
                    # valid_index = np.loadtxt(
                    #     r'/home/wangtao/project/Survival/split_ids/valid_index{}_fold{}_nt59.csv'.format(hospital, fold),
                    #     delimiter=',', dtype=str)
                    # valid_index = valid_index.astype(int)
                    train_image_files_split = image_files_split[train_index]
                    train_image_cir_files_split = image_cir_files_split[train_index]
                    train_image_files.append(train_image_files_split)
                    train_image_cir_files.append(train_image_cir_files_split)
                    train_feat_cir_split = feat_cir_split[train_index]
                    train_feat_cir.append(train_feat_cir_split)
                    train_labels_split = labels_split[train_index]
                    train_labels.append(train_labels_split)

                    train_ids_split = ids_split[train_index]
                    train_ids.append(train_ids_split)
                    train_censoreds_split = censoreds_split[train_index]
                    train_censoreds.append(train_censoreds_split)
                    train_text_split = text_split[train_index]
                    train_text.append(train_text_split)

                    train_discrete_clinicl_split = discrete_clinicl_split[train_index]
                    train_dis_clinics.append(train_discrete_clinicl_split)

                    train_continuous_clinicl_split = continuous_clinicl_split[train_index]
                    train_con_clinics.append(train_continuous_clinicl_split)

                    train_radiomics_split = radiomics_split[train_index]
                    train_radiomics.append(train_radiomics_split)

                    valid_image_files_split = image_files_split[valid_index]
                    valid_image_cir_files_split = image_cir_files_split[valid_index]
                    valid_image_files.append(valid_image_files_split)
                    valid_image_cir_files.append(valid_image_cir_files_split)
                    valid_feat_cir_split = feat_cir_split[valid_index]
                    valid_feat_cir.append(valid_feat_cir_split)
                    valid_labels_split = labels_split[valid_index]
                    valid_labels.append(valid_labels_split)
                    # valid_clinics_split = clinics_split[valid_index]
                    # valid_clinics.append(valid_clinics_split)
                    valid_ids_split = ids_split[valid_index]
                    valid_ids.append(valid_ids_split)
                    valid_censoreds_split = censoreds_split[valid_index]
                    valid_censoreds.append(valid_censoreds_split)
                    valid_text_split = text_split[valid_index]
                    valid_text.append(valid_text_split)

                    valid_discrete_clinicl_split = discrete_clinicl_split[valid_index]
                    valid_dis_clinics.append(valid_discrete_clinicl_split)

                    valid_continuous_clinicl_split = continuous_clinicl_split[valid_index]
                    valid_con_clinics.append(valid_continuous_clinicl_split)

                    valid_radiomics_split = radiomics_split[valid_index]
                    valid_radiomics.append(valid_radiomics_split)
            _fold = 0

    Scaler = StandardScaler()
    Scaler2 = StandardScaler()
    # OHencoder = OneHotEncoder(categories='auto', sparse_output=False)

    train_image_files = np.concatenate(train_image_files, axis=0)
    train_image_cir_files = np.concatenate(train_image_cir_files, axis=0)
    train_feat_cir = np.concatenate(train_feat_cir, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_con_clinics = np.concatenate(train_con_clinics, axis=0)
    train_dis_clinics = np.concatenate(train_dis_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    train_text = np.concatenate(train_text, axis=0)
    train_con_clinics_ts = Scaler.fit_transform(train_con_clinics)

    # train_dis_clinics_ts = OHencoder.fit_transform(train_dis_clinics)
    train_clinics = np.concatenate([train_con_clinics_ts, train_dis_clinics], axis=-1)
    train_radiomics = np.concatenate(train_radiomics, axis=0)[:, 0 ,:]
    train_radiomics = Scaler2.fit_transform(train_radiomics)
    # ### train_image_files, train_image_cir_files, train_clinics, train_feat_cir,
    #                                                   train_labels, train_censoreds, train_durations_discrete,
    #                                                   train_events, train_ids, train_text, train_transforms


    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_image_cir_files = np.concatenate(valid_image_cir_files, axis=0)
    valid_feat_cir = np.concatenate(valid_feat_cir, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    valid_con_clinics = np.concatenate(valid_con_clinics, axis=0)
    valid_dis_clinics = np.concatenate(valid_dis_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    valid_text = np.concatenate(valid_text, axis=0)
    valid_con_clinics_ts = Scaler.transform(valid_con_clinics)
    # valid_dis_clinics_ts = OHencoder.fit_transform(valid_dis_clinics)
    valid_clinics = np.concatenate([valid_con_clinics_ts, valid_dis_clinics], axis=-1)
    valid_radiomics = np.concatenate(valid_radiomics, axis=0)[:, 0 ,:]
    valid_radiomics = Scaler2.transform(valid_radiomics)

    test_image_files = np.concatenate(test_image_files, axis=0)
    test_image_cir_files = np.concatenate(test_image_cir_files, axis=0)
    test_feat_cir = np.concatenate(test_feat_cir, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_con_clinics = np.concatenate(test_con_clinics, axis=0)
    test_dis_clinics = np.concatenate(test_dis_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    test_text = np.concatenate(test_text, axis=0)
    test_con_clinics_ts = Scaler.transform(test_con_clinics)
    # test_dis_clinics_ts = OHencoder.fit_transform(test_dis_clinics)
    test_clinics = np.concatenate([test_con_clinics_ts, test_dis_clinics], axis=-1)
    test_radiomics = np.concatenate(test_radiomics, axis=0)[:, 0 ,:]
    test_radiomics = Scaler2.transform(test_radiomics)

    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)

    train_text = train_text.tolist()
    valid_text = valid_text.tolist()
    test_text = test_text.tolist()


    return (train_ids, train_radiomics, train_clinics, train_censoreds, train_labels,
            valid_ids, valid_radiomics, valid_clinics, valid_censoreds, valid_labels,
            test_ids, test_radiomics, test_clinics, test_censoreds, test_labels, radiomics_title, clinical_title)

def create_dataset_ext_week_persudo_pair_addcirf_fixid3(matched_feature_file1, matched_feature_file2, matched_ids_file1, matched_ids_file2,
                                matched_censor_file1, matched_censor_file2, matched_surv_file1, matched_surv_file2,
                                         root_dir, fold, num_durations=6, batch_size=6, val_batch_size=4, seed=1998):
    features_cp0 = np.load(matched_feature_file1)
    features_cp1 = np.load(matched_feature_file2)[:, 0, ...]
    ids_cp0 = np.load(matched_ids_file1)
    ids_cp1 = np.load(matched_ids_file2)[:, 0, ...]
    censors_cp0 = np.load(matched_censor_file1)
    censors_cp1 = np.load(matched_censor_file2)
    survs_cp0 = np.load(matched_surv_file1)
    survs_cp1 = np.load(matched_surv_file2)

    match_ids_sum = np.concatenate([ids_cp0, ids_cp1])
    match_features_sum = np.concatenate([features_cp0, features_cp1])
    match_censors_sum = np.concatenate([censors_cp0, censors_cp1])
    match_survs_sum = np.concatenate([survs_cp0, survs_cp1])
    # print_config()
    data_dir1 = root_dir
    hospital_names = (
        sorted([x for x in os.listdir(data_dir1) if os.path.isdir(os.path.join(data_dir1, x))]))
    image_files = [[os.path.join(data_dir1, class_name, x)
                     for x in os.listdir(os.path.join(data_dir1, class_name))]
                    for class_name in hospital_names]
    train_ids_txt_features = np.load(r'/home/wangtao/project/Survival/saved_txt_features/train_ids.npy')
    valid_ids_txt_features = np.load(r'/home/wangtao/project/Survival/saved_txt_features/valid_ids.npy')
    test_ids_txt_features = np.load(r'/home/wangtao/project/Survival/saved_txt_features/test_ids.npy')
    ids_txt_features = np.concatenate([train_ids_txt_features, valid_ids_txt_features, test_ids_txt_features])

    train_txt_features1 = np.load(r'/home/wangtao/project/Survival/saved_txt_features/train_features.npy')
    valid_txt_features1 = np.load(r'/home/wangtao/project/Survival/saved_txt_features/valid_features.npy')
    test_txt_features1 = np.load(r'/home/wangtao/project/Survival/saved_txt_features/test_features.npy')
    txt_features1 = np.concatenate([train_txt_features1, valid_txt_features1, test_txt_features1])

    train_txt_features2 = np.load(r'/home/wangtao/project/Survival/saved_txt_features/train_featuresx.npy')
    valid_txt_features2 = np.load(r'/home/wangtao/project/Survival/saved_txt_features/valid_featuresx.npy')
    test_txt_features2 = np.load(r'/home/wangtao/project/Survival/saved_txt_features/test_featuresx.npy')
    txt_features2 = np.concatenate([train_txt_features2, valid_txt_features2, test_txt_features2])


    image_files_need = []
    image_cir_files_need = []
    feat_cir_need = []
    info = pd.read_excel(r"/home/wangtao/project/Cirros_HCC/data/table/PFS_ZYJ_20240319_PD_CP_text.xlsx")
    ids = info['序号']
    hospitals = info['病例来源']

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    gpt_text = info['text']

    continuous_clinicl = info.iloc[:, [5, 10]]
    discrete_clinicl = info.iloc[:, [6, 7, 8, 9, 11, 12, 23]]
    # discrete_clinicl = info.iloc[:, [7, 8, 9]]
    continuous_clinicl = np.asarray(continuous_clinicl)
    discrete_clinicl = np.asarray(discrete_clinicl)
    discrete_clinicl[np.where(np.isnan(discrete_clinicl)==True)] = 1
    OHencoder = OneHotEncoder(categories='auto', sparse_output=False)
    discrete_clinicl = OHencoder.fit_transform(discrete_clinicl)

    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    # surv_labels += 1
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    txt_feat_sum = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        continuous_clinicl_hospital = []
        discrete_clinicl_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        image_cir_files_need_hospital = []
        feat_cir_need_hospital = []
        txt_feat_hospital = []
        for image_file in image_files_hospital:
            ID = image_file.split('/')[-1].split('.')[0]
            pos_txt_feature = np.where(ids_txt_features.astype(str) == ID)
            # txt_feat_now = txt_features2[pos_txt_feature[0]].mean(axis=1)
            txt_feat_now = txt_features1[pos_txt_feature[0]]
            center_now = image_file.split('/')[-2]
            pos = np.where(ids.astype(str) == ID)
            pos_match = np.where(match_ids_sum[:, -1] == center_now + '_' + ID)
            if len(pos[0]) != 0:
                cp_now = cp_labels[pos][0]
                if cp_now != 3 and np.isnan(cp_now) == False:
                    id_cir_now = match_ids_sum[pos_match[0], 0][0]
                    feat_cir_now = match_features_sum[pos_match[0], 0][0][None, :]
                    image_cir_file = os.path.join(r'/home/wangtao/project/Survival/data_cir/preprocess/Liver/data',
                                                  id_cir_now.split('_')[0], id_cir_now + '.nii.gz')
                    surv_label = surv_labels[pos][0]
                    if_censored = if_censoreds[pos][0]
                    text_now = gpt_text[pos][0]

                    continuous_clinicl_now = continuous_clinicl[pos][0]
                    discrete_clinicl_now = discrete_clinicl[pos][0]
                    image_files_need_hospital.append(image_file)
                    ids_hospital.append(ID)
                    text_hospital.append(text_now)
                    labels_hospital.append(surv_label)
                    censored_hospital.append(if_censored)
                    image_cir_files_need_hospital.append(image_cir_file)
                    feat_cir_need_hospital.append(feat_cir_now)
                    continuous_clinicl_hospital.append(continuous_clinicl_now)
                    discrete_clinicl_hospital.append(discrete_clinicl_now)
                    txt_feat_hospital.append(txt_feat_now)

        txt_feat_hospital = np.concatenate(txt_feat_hospital)
        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        image_cir_files_need_hospital = np.array(image_cir_files_need_hospital)
        feat_cir_need_hospital = np.concatenate(feat_cir_need_hospital)
        continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        image_cir_files_need.append(image_cir_files_need_hospital)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)
        feat_cir_need.append(feat_cir_need_hospital)
        continuous_clinicl_sum.append(continuous_clinicl_hospital)
        discrete_clinicl_sum.append(discrete_clinicl_hospital)
        txt_feat_sum.append(txt_feat_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    train_val_image_files = []
    train_val_image_cir_files = []
    train_val_feat_cir = []
    train_val_labels = []
    train_val_con_clinics = []
    train_val_dis_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []
    train_val_txt_feats = []

    train_image_files = []
    train_image_cir_files = []
    train_feat_cir = []
    train_labels = []
    train_con_clinics = []
    train_dis_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []
    train_txt_feats = []

    valid_image_files = []
    valid_image_cir_files = []
    valid_feat_cir = []
    valid_labels = []
    valid_con_clinics = []
    valid_dis_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []
    valid_txt_feats = []

    test_image_files = []
    test_image_cir_files = []
    test_feat_cir = []
    test_labels = []
    test_con_clinics = []
    test_dis_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []
    test_txt_feats = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        image_cir_files_split = np.array(image_cir_files_need[hospital])
        feat_cir_split = feat_cir_need[hospital]
        labels_split = labels[hospital]
        # continuous_clinics_split = continuous_clinicl_sum[hospital]
        # discrete_clinics_split = discrete_clinicl_sum[hospital]
        # clinics_split = np.concatenate([continuous_clinics_split, discrete_clinics_split], axis=-1)
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        continuous_clinicl_split = continuous_clinicl_sum[hospital]
        discrete_clinicl_split = discrete_clinicl_sum[hospital]
        txt_feat_split = txt_feat_sum[hospital]
        # if hospital_name_split in ['Center2', 'Center3', 'Center4', 'Center8', 'Center9', 'Center5', 'Center11']:  # 0420_1
        if hospital_name_split not in ['Center5', 'Center9']:
            train_val_image_files.append(image_files_split)
            train_val_image_cir_files.append(image_cir_files_split)
            train_val_labels.append(labels_split)
            # train_val_clinics.append(clinics_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
            train_val_feat_cir.append(feat_cir_split)
            train_val_dis_clinics.append(discrete_clinicl_split)
            train_val_con_clinics.append(continuous_clinicl_split)
            train_val_txt_feats.append(txt_feat_split)
        else:
            test_image_files.append(image_files_split)
            test_image_cir_files.append(image_cir_files_split)
            test_labels.append(labels_split)
            # test_clinics.append(clinics_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)
            test_feat_cir.append(feat_cir_split)
            test_dis_clinics.append(discrete_clinicl_split)
            test_con_clinics.append(continuous_clinicl_split)
            test_txt_feats.append(txt_feat_split)

    for hospital in range(len(train_val_image_files)):
        image_files_split = np.array(train_val_image_files[hospital])
        image_cir_files_split = np.array(train_val_image_cir_files[hospital])
        labels_split = train_val_labels[hospital]
        feat_cir_split = train_val_feat_cir[hospital]
        # clinics_split = train_val_clinics[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        censoreds_split = train_val_censoreds[hospital]
        continuous_clinicl_split = train_val_con_clinics[hospital]
        discrete_clinicl_split = train_val_dis_clinics[hospital]
        txt_feat_split = train_val_txt_feats[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_image_cir_files.append(image_cir_files_split)
            train_labels.append(labels_split)
            train_dis_clinics.append(discrete_clinicl_split)
            train_con_clinics.append(continuous_clinicl_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
            train_feat_cir.append(feat_cir_split)
            train_txt_feats.append(txt_feat_split)
        else:
            # for train_index, valid_index in kf.split(image_files_split, censoreds_split):
            #     _fold = _fold + 1
            #     if _fold == fold:
            train_index = np.loadtxt(r'/home/wangtao/project/Survival/split_ids/train_index{}_fold{}_nt59.csv'.format(hospital, fold),
                                     delimiter=',', dtype=str)
            train_index = train_index.astype(int)
            valid_index = np.loadtxt(r'/home/wangtao/project/Survival/split_ids/valid_index{}_fold{}_nt59.csv'.format(hospital, fold),
                                     delimiter=',', dtype=str)
            valid_index = valid_index.astype(int)
            train_image_files_split = image_files_split[train_index]
            train_image_cir_files_split = image_cir_files_split[train_index]
            train_image_files.append(train_image_files_split)
            train_image_cir_files.append(train_image_cir_files_split)
            train_feat_cir_split = feat_cir_split[train_index]
            train_feat_cir.append(train_feat_cir_split)
            train_labels_split = labels_split[train_index]
            train_labels.append(train_labels_split)
            train_txt_feats_split = txt_feat_split[train_index]
            train_txt_feats.append(train_txt_feats_split)

            train_ids_split = ids_split[train_index]
            train_ids.append(train_ids_split)
            train_censoreds_split = censoreds_split[train_index]
            train_censoreds.append(train_censoreds_split)
            train_text_split = text_split[train_index]
            train_text.append(train_text_split)

            train_discrete_clinicl_split = discrete_clinicl_split[train_index]
            train_dis_clinics.append(train_discrete_clinicl_split)

            train_continuous_clinicl_split = continuous_clinicl_split[train_index]
            train_con_clinics.append(train_continuous_clinicl_split)

            valid_image_files_split = image_files_split[valid_index]
            valid_image_cir_files_split = image_cir_files_split[valid_index]
            valid_image_files.append(valid_image_files_split)
            valid_image_cir_files.append(valid_image_cir_files_split)
            valid_feat_cir_split = feat_cir_split[valid_index]
            valid_feat_cir.append(valid_feat_cir_split)
            valid_labels_split = labels_split[valid_index]
            valid_labels.append(valid_labels_split)

            valid_txt_feats_split = txt_feat_split[valid_index]
            valid_txt_feats.append(valid_txt_feats_split)
            # valid_clinics_split = clinics_split[valid_index]
            # valid_clinics.append(valid_clinics_split)
            valid_ids_split = ids_split[valid_index]
            valid_ids.append(valid_ids_split)
            valid_censoreds_split = censoreds_split[valid_index]
            valid_censoreds.append(valid_censoreds_split)
            valid_text_split = text_split[valid_index]
            valid_text.append(valid_text_split)

            valid_discrete_clinicl_split = discrete_clinicl_split[valid_index]
            valid_dis_clinics.append(valid_discrete_clinicl_split)

            valid_continuous_clinicl_split = continuous_clinicl_split[valid_index]
            valid_con_clinics.append(valid_continuous_clinicl_split)

    Scaler = StandardScaler()
    # OHencoder = OneHotEncoder(categories='auto', sparse_output=False)


    train_image_files = np.concatenate(train_image_files, axis=0)
    train_image_cir_files = np.concatenate(train_image_cir_files, axis=0)
    train_feat_cir = np.concatenate(train_feat_cir, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_con_clinics = np.concatenate(train_con_clinics, axis=0)
    train_dis_clinics = np.concatenate(train_dis_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    train_text = np.concatenate(train_text, axis=0)
    train_con_clinics_ts = Scaler.fit_transform(train_con_clinics)
    # train_dis_clinics_ts = OHencoder.fit_transform(train_dis_clinics)
    train_clinics = np.concatenate([train_con_clinics_ts, train_dis_clinics], axis=-1)
    train_txt_feats = np.concatenate(train_txt_feats, axis=0)

    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_image_cir_files = np.concatenate(valid_image_cir_files, axis=0)
    valid_feat_cir = np.concatenate(valid_feat_cir, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    valid_con_clinics = np.concatenate(valid_con_clinics, axis=0)
    valid_dis_clinics = np.concatenate(valid_dis_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    valid_text = np.concatenate(valid_text, axis=0)
    valid_con_clinics_ts = Scaler.transform(valid_con_clinics)
    # valid_dis_clinics_ts = OHencoder.fit_transform(valid_dis_clinics)
    valid_clinics = np.concatenate([valid_con_clinics_ts, valid_dis_clinics], axis=-1)
    valid_txt_feats = np.concatenate(valid_txt_feats, axis=0)

    test_image_files = np.concatenate(test_image_files, axis=0)
    test_image_cir_files = np.concatenate(test_image_cir_files, axis=0)
    test_feat_cir = np.concatenate(test_feat_cir, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_con_clinics = np.concatenate(test_con_clinics, axis=0)
    test_dis_clinics = np.concatenate(test_dis_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    test_text = np.concatenate(test_text, axis=0)
    test_con_clinics_ts = Scaler.transform(test_con_clinics)
    # test_dis_clinics_ts = OHencoder.fit_transform(test_dis_clinics)
    test_clinics = np.concatenate([test_con_clinics_ts, test_dis_clinics], axis=-1)
    test_txt_feats = np.concatenate(test_txt_feats, axis=0)

    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(test_labels, test_censoreds)

    train_text = train_text.tolist()
    valid_text = valid_text.tolist()
    test_text = test_text.tolist()

    train_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"], ),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        RandRotated(keys=["image", "liver", "tumor", "cir_liver"], range_x=15, prob=0.5, keep_size=True),
        RandFlipd(keys=["image", "liver", "tumor", "cir_liver"], spatial_axis=0, prob=0.2),
        RandZoomd(keys=["image", "liver", "tumor", "cir_liver"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambda(lambda x: x.permute(0, 3, 2, 1)),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver"], ),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver"]),
        # Lambdad(keys=["image", "liver", "tumor"], func=lambda x: x.permute(0, 3, 2, 1)),
    ])

    train_dataset = LiverDataset_disentangle_cir_feat_cli_txt(train_image_files, train_image_cir_files, train_clinics,
                                                              train_txt_feats, train_feat_cir,
                                                              train_labels, train_censoreds, train_durations_discrete,
                                                              train_events, train_ids, train_text, train_transforms)
    valid_dataset = LiverDataset_disentangle_cir_feat_cli_txt(valid_image_files, valid_image_cir_files, valid_clinics,
                                                              valid_txt_feats, valid_feat_cir,
                                                              valid_labels, valid_censoreds, valid_durations_discrete,
                                                              valid_events, valid_ids, valid_text,  val_transforms)
    test_dataset = LiverDataset_disentangle_cir_feat_cli_txt(test_image_files, test_image_cir_files, test_clinics,
                                                             test_txt_feats, test_feat_cir,
                                                             test_labels, test_censoreds, test_durations_discrete,
                                                             test_events, test_ids, test_text, val_transforms)

    # train_infos = np.concatenate([train_ids[:, None], train_image_files[:, None], train_labels[:, None],
    #                          train_censoreds[:, None], train_clinics, np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], valid_image_files[:, None], valid_labels[:, None],
    #                               valid_censoreds[:, None], valid_clinics, np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], test_image_files[:, None], test_labels[:, None],
    #                               test_censoreds[:, None], test_clinics, np.array(test_text)[:, None]], axis=1)
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_infos.csv', train_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_infos.csv', valid_infos, fmt='%s',
    #            delimiter=',')
    # np.savetxt(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_infos.csv', test_infos, fmt='%s',
    #            delimiter=',')
    # train_infos = np.concatenate([train_ids[:, None], np.array(train_text)[:, None]], axis=1)
    # valid_infos = np.concatenate([valid_ids[:, None], np.array(valid_text)[:, None]], axis=1)
    # test_infos = np.concatenate([test_ids[:, None], np.array(test_text)[:, None]], axis=1)
    # train_infos = pd.DataFrame(train_infos, columns=['ids', 'text'])
    # valid_infos = pd.DataFrame(valid_infos, columns=['ids', 'text'])
    # test_infos = pd.DataFrame(test_infos, columns=['ids', 'text'])
    # train_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/train_text.xlsx', index=False)
    # valid_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/valid_text.xlsx', index=False)
    # test_infos.to_excel(r'/home/wangtao/project/FundationModel/MyFM/Mymodel3/test_text.xlsx', index=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=2)
    x_image_files, x_clinics, x_txt_features, x_cir_feat, x_labels, x_censoreds, x_durations_discrete, \
        x_events, x_ids, x_text = next(iter(train_loader))
    # for i, batch_data in enumerate(train_loader):
    #     s = 0
    # train_sample = next(iter(train_loader))
    return train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader

def set_seeds(seed=1998):
    """Set Python random seeding and PyTorch seeds.
    Parameters
    ----------
    seed: int, default: 42
        Random number generator seeds for PyTorch and python
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seeds(1998)
    # train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader = (
    #     create_dataset_ext(r"/home/wangtao/project/Liver_project/Liver_surv/data2/preprocessed/Liver/data/",
    #                        r"/home/wangtao/project/Liver_project/Liver_surv/data2/preprocessed_first/"
    #                        r"Liver/data/", 1))
    # aa = next(iter(train_loader))

    # train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader = (
    #     create_dataset_ext_week_disentangle("/home/wangtao/project/Survival/data/preprocess/tumor2/data",
    #                               1, batch_size=4))
    # print(len(train_dataset), len(valid_dataset), len(test_dataset))
    # train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader = (
    #     create_dataset_ext_week_disentangle_nocp3("/home/wangtao/project/Survival/data/preprocess/tumor2/data",
    #                                         1, batch_size=4))
    # print(len(train_dataset), len(valid_dataset), len(test_dataset))

    # train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader = \
    #     create_dataset_ext_week_persudo_pair(
    #         "/home/wangtao/project/Survival/saved_feaures/matched_feaures_cp0.npy",
    #         "/home/wangtao/project/Survival/saved_feaures/matched_feaures_cp1.npy",
    #         "/home/wangtao/project/Survival/saved_feaures/matched_ids_cp0.npy",
    #         "/home/wangtao/project/Survival/saved_feaures/matched_ids_cp1.npy",
    #         "/home/wangtao/project/Survival/saved_feaures/matched_censors_cp0.npy",
    #         "/home/wangtao/project/Survival/saved_feaures/matched_censors_cp1.npy",
    #         "/home/wangtao/project/Survival/saved_feaures/matched_survs_cp0.npy",
    #         "/home/wangtao/project/Survival/saved_feaures/matched_survs_cp1.npy",
    #         "/home/wangtao/project/Survival/data/preprocess/tumor2/data", 1, batch_size=4)

    train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader = \
        create_dataset_ext_week_persudo_pair_addcirf_fixid3(
            "/home/wangtao/project/Survival/saved_feaures/matched_feaures_cp0.npy",
            "/home/wangtao/project/Survival/saved_feaures/matched_feaures_cp1.npy",
            "/home/wangtao/project/Survival/saved_feaures/matched_ids_cp0.npy",
            "/home/wangtao/project/Survival/saved_feaures/matched_ids_cp1.npy",
            "/home/wangtao/project/Survival/saved_feaures/matched_censors_cp0.npy",
            "/home/wangtao/project/Survival/saved_feaures/matched_censors_cp1.npy",
            "/home/wangtao/project/Survival/saved_feaures/matched_survs_cp0.npy",
            "/home/wangtao/project/Survival/saved_feaures/matched_survs_cp1.npy",
            "/home/wangtao/project/Survival/data/preprocess/liver/data", 1, batch_size=4)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    s = 0
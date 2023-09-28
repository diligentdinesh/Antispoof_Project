import numpy as np
import tensorflow as tf
import pickle
import os


class ASVspoof2019(tf.keras.utils.Sequence):
    def __init__(
        self,
        access_type,
        path_to_features,
        path_to_protocol,
        part="train",
        feature="LFCC",
        genuine_only=False,
        feat_len=750,
        padding="repeat",
        batch_size=32
    ):
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.genuine_only = genuine_only
        self.feat_len = feat_len
        self.feature = feature
        self.path_to_protocol = path_to_protocol
        self.padding = padding
        self.batch_size = batch_size
        protocol = os.path.join(
            self.path_to_protocol, f"ASVspoof2019.{access_type}.cm.{self.part}.trl.txt"
        )
        if self.access_type == "LA":
            self.tag = {
                "-": 0,
                "A01": 1,
                "A02": 2,
                "A03": 3,
                "A04": 4,
                "A05": 5,
                "A06": 6,
                "A07": 7,
                "A08": 8,
                "A09": 9,
                "A10": 10,
                "A11": 11,
                "A12": 12,
                "A13": 13,
                "A14": 14,
                "A15": 15,
                "A16": 16,
                "A17": 17,
                "A18": 18,
                "A19": 19,
            }
        else:
            self.tag = {
                "-": 0,
                "AA": 1,
                "AB": 2,
                "AC": 3,
                "BA": 4,
                "BB": 5,
                "BC": 6,
                "CA": 7,
                "CB": 8,
                "CC": 9,
            }
        self.label = {"spoof": 1, "bonafide": 0}

        with open(protocol, "r") as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            if genuine_only:
                assert self.part in ["train", "dev"]
                if self.access_type == "LA":
                    num_bonafide = {"train": 2580, "dev": 2548}
                    self.all_info = audio_info[: num_bonafide[self.part]]
                else:
                    self.all_info = audio_info[:5400]
            else:
                self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        feat_mats = []
        filenames = []
        tags = []
        labels = []
        for speaker, filename, _, tag, label in self.all_info[idx * self.batch_size:(idx + 1) * self.batch_size]:
            # speaker, filename, _, tag, label = self.all_info[idx]
            try:
                with open(
                    os.path.join(self.ptf + "/" + filename + f"{self.feature}.pkl"), "rb"
                ) as feature_handle:
                    feat_mat = pickle.load(feature_handle)
            except:

                def the_other(train_or_dev):
                    assert train_or_dev in ["train", "dev"]
                    res = "dev" if train_or_dev == "train" else "train"
                    return res

                with open(
                    os.path.join(
                        self.path_to_features,
                        the_other(self.part) + "/" + filename + f"{self.feature}.pkl",
                    ),
                    "rb",
                ) as feature_handle:
                    feat_mat = pickle.load(feature_handle)

            feat_mat = tf.convert_to_tensor(feat_mat, dtype=tf.float32)
            this_feat_len = feat_mat.shape[1]
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                feat_mat = feat_mat[:, startp : startp + self.feat_len]
            if this_feat_len < self.feat_len:
                if self.padding == "zero":
                    feat_mat = self.padding_func(feat_mat, self.feat_len)
                elif self.padding == "repeat":
                    feat_mat = self.repeat_padding_func(feat_mat, self.feat_len)
                else:
                    raise ValueError("Padding should be zero or repeat!")
            feat_mats.append(feat_mat)
            filenames.append(filename)
            tags.append(self.tag[tag])
            labels.append(self.label[label])
            # return feat_mat, filename, self.tag[tag], self.label[label]
        return tf.expand_dims(tf.stack(feat_mats), axis=3), tf.stack(filenames), tf.stack(tags), tf.stack(labels)

    def on_epoch_end(self):
        pass

    def padding_func(self, spec, ref_len):
        width, cur_len = spec.shape
        assert ref_len > cur_len
        padd_len = ref_len - cur_len
        return tf.concat([spec, tf.zeros([width, padd_len], dtype=tf.float32)], axis=1)

    def repeat_padding_func(self, spec, ref_len):
        mul = tf.math.ceil(ref_len / tf.cast(spec.shape[1], dtype=tf.float32))
        spec = tf.tile(spec, [1, tf.cast(mul, dtype=tf.int32)])[:, :ref_len]
        return spec


if __name__ == "__main__":
    path_to_features = "/dataNVME/neil/ASVspoof2019Features/"  # if run on GPU
    path_to_protocol = "/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/"
    training_set = ASVspoof2019(
        'LA',
        '/home/user/Antispoof_Project/LA/Features_py/',
        '/home/user/Antispoof_Project/LA/ASVspoof2019_LA_cm_protocols/',
        "train",
        "LFCC",
        feat_len=750,
        padding='repeat',
    )
    feat, _, _, labels = training_set[29]
    print("Feature shape", feat.shape)
    print("label shape: ", labels.shape)
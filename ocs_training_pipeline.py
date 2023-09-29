import argparse
import os
import json
import shutil
# from resnet import setup_seed, ResNet
# from loss import *
# from dataset import ASVspoof2019
from collections import defaultdict
from tqdm import tqdm
import eval_metrics as em
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import tensorflow as tf
from models.loss import ocsoftmaxmodel


from models.ResNet import ResNet18
from models.DataGeneratorPKL import ASVspoof2019

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    # Data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/home/user/Antispoof_Project/LA/Features_py/')
    parser.add_argument("-p", "--path_to_protocol", type=str, help="protocol path",
                        default='/home/user/Antispoof_Project/LA/ASVspoof2019_LA_cm_protocols/')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=True, default='./models/try/')

    # Dataset prepare
    parser.add_argument("--feat_len", type=int, help="features length", default=750)
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat'],
                        help="how to pad short utterance")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")
    parser.add_argument('--seed', type=int, help="random number seed", default=598)

    parser.add_argument('--add_loss', type=str, default="ocsoftmax",
                        choices=["softmax", 'amsoftmax', 'ocsoftmax'], help="loss for one-class training")
    parser.add_argument('--weight_loss', type=float, default=1, help="weight for other loss")
    parser.add_argument('--r_real', type=float, default=0.9, help="r_real for ocsoftmax")
    parser.add_argument('--r_fake', type=float, default=0.2, help="r_fake for ocsoftmax")
    parser.add_argument('--alpha', type=float, default=20, help="scale factor for ocsoftmax")

    parser.add_argument('--continue_training', action='store_true', help="continue training with previously trained model")

    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    # setup_seed(args.seed)

    if args.continue_training:
        assert os.path.exists(args.out_fold)
    else:
        # Path for output data
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        # Path for input data
        assert os.path.exists(args.path_to_features)

        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")

    # assign device
    args.cuda = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

    print('Cuda device available: ', args.cuda)
    args.device = tf.device("cuda" if args.cuda else "cpu")
    print(args.device)

    return args

# import tensorflow as tf

# # Define a learning rate schedule
# learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.001,
#     decay_steps=10000,
#     decay_rate=0.96
# )

# # Create an optimizer with the learning rate schedule
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
# def adjust_learning_rate(args, optimizer, epoch_num):
#     lr = args.lr * (args.lr_decay ** (epoch_num // args.interval))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def train_lfcc(lfcc_model, lfcc, labels, lfcc_optimizer):
    with tf.GradientTape() as tape: 
        feats, lfcc_outputs = lfcc_model(lfcc, training=True)
        lfcc_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, lfcc_outputs)
    lfcc_gradients = tape.gradient(lfcc_loss, lfcc_model.trainable_variables)
    lfcc_optimizer.apply_gradients(zip(lfcc_gradients, lfcc_model.trainable_variables))
    return feats

@tf.function
def train_ocsoftmax(ocsoftmax, feats, labels, ocsoftmax_optimizer):
    with tf.GradientTape() as tape:
        ocsoftmaxloss, _ = ocsoftmax([feats, labels], training=True)
        ocsoftmaxloss = ocsoftmaxloss * args.weight_loss
    gradients = tape.gradient(ocsoftmaxloss, ocsoftmax.trainable_variables)
    ocsoftmax_optimizer.apply_gradients(zip(gradients, ocsoftmax.trainable_variables))
    return ocsoftmaxloss

@tf.function
def eval_lfcc(lfcc_model, lfcc):
    feats, lfcc_outputs = lfcc_model(lfcc)
    score = lfcc_outputs[:, 0]
    return feats, score

@tf.function
def eval_ocsoftmax(ocsoftmax, feats, labels):
    ocsoftmaxloss, _ = ocsoftmax([feats, labels])
    return ocsoftmax

def train(args):
    print("[INFO] Loading Resnet Model")
    lfcc_model = ResNet18(input_shape=(60, 750, 1), classes=2)
    print("[INFO] Loaded Resnet Model")

    # if args.continue_training:
    #     lfcc_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt')).to(args.device)

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.lr,
        decay_steps=args.interval,
        decay_rate=args.lr_decay
    )

    lfcc_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule, beta_1=args.beta_1, beta_2=args.beta_2)

    print("[INFO] Loading Training Data")
    trainDataLoader = ASVspoof2019(args.access_type, args.path_to_features, args.path_to_protocol, 'train',
                                'LFCC', feat_len=args.feat_len, padding=args.padding)
    print("[INFO] Loading Validation Data")
    valDataLoader = ASVspoof2019(args.access_type, args.path_to_features, args.path_to_protocol, 'dev',
                                  'LFCC', feat_len=args.feat_len, padding=args.padding)

    if args.add_loss == 'ocsoftmax':
        print("[INFO] Loading OCSoftmax")
        
        ocsoftmax = ocsoftmaxmodel(args.enc_dim, r_real=args.r_real, r_fake=args.r_fake, alpha=args.alpha)
        ocsoftmax_optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate_schedule)
    

    early_stop_cnt = 0
    prev_eer = 1e8

    monitor_loss = args.add_loss

    for epoch_num in tqdm(range(args.num_epochs)):
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        print('\nEpoch: %d ' % (epoch_num + 1))
        for i , (lfcc, audio_fn, tags, labels) in enumerate(tqdm(trainDataLoader)):
            feats = train_lfcc(lfcc_model, lfcc, labels, lfcc_optimizer)
            # with tf.GradientTape() as tape: 
            #     feats, lfcc_outputs = lfcc_model(lfcc, training=True)
            #     lfcc_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, lfcc_outputs)
            # lfcc_gradients = tape.gradient(lfcc_loss, lfcc_model.trainable_variables)
            # lfcc_optimizer.apply_gradients(zip(lfcc_gradients, lfcc_model.trainable_variables))


            
            if args.add_loss =="ocsoftmax":
                ocsoftmaxloss = train_ocsoftmax(ocsoftmax, feats, labels, ocsoftmax_optimizer)
                # with tf.GradientTape() as tape:
                #     ocsoftmaxloss, _ = ocsoftmax([feats, labels], training=True)
                #     ocsoftmaxloss = ocsoftmaxloss * args.weight_loss
                # gradients = tape.gradient(ocsoftmaxloss, ocsoftmax.trainable_variables)
                # ocsoftmax_optimizer.apply_gradients(zip(gradients, ocsoftmax.trainable_variables))
                trainlossDict[args.add_loss].append(tf.keras.backend.eval(ocsoftmaxloss))

                with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                            str(np.nanmean(trainlossDict[monitor_loss])) + "\n")
            
        idx_loader, score_loader = [], []
        for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(valDataLoader)):
            feats, score = eval_lfcc(lfcc_model, lfcc)
            # feats, lfcc_outputs = lfcc_model(lfcc)
            # score = lfcc_outputs[:, 0]
            if args.add_loss == 'ocsoftmax':
                ocsoftmaxloss = eval_ocsoftmax(ocsoftmax, feats, labels)
                # ocsoftmaxloss, _ = ocsoftmax([feats, labels])
                devlossDict[args.add_loss].append(tf.keras.backend.eval(ocsoftmaxloss))
            idx_loader.append(labels)
            score_loader.append(score)

        scores = tf.concat(score_loader, 0).data.cpu().numpy()
        labels = tf.concat(idx_loader, 0).data.cpu().numpy()
        val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
        with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
            log.write(str(epoch_num) + "\t" + str(np.nanmean(devlossDict[monitor_loss])) + "\t" + str(val_eer) +"\n")
        print("Val EER: {}".format(val_eer))

        if val_eer < prev_eer:
            # Save the model checkpoint
            lfcc_model.save(os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt'))
            if args.add_loss == "ocsoftmax":
                ocsoftmax.save(os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))
            else:
                loss_model = None
            prev_eer = val_eer
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 100:
            with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 19))
            break
    return lfcc_model, ocsoftmax
   
    
if __name__ == "__main__":
    args = initParams()
    train(args)
    # model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_lfcc_model.pt'))
    # if args.add_loss == "softmax":
    #     loss_model = None
    # else:
    #     loss_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))

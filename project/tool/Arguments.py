import random


class Arguments():
    def __init__(self):
        self.clients = 4
        self.rounds = 100
        self.multiplier=1
        self.sparse=False
        self.epochs = 1
        self.death='random'
        self.growth='magnitude'
        self.redistribution=None
        self.update_frequency=35
        self.density=0.05
        self.fix=True
        self.sparse_init="ERK"
        self.local_batches = 8
        self.C = 0.9
        self.drop_rate = 0.1
        self.torch_seed = 0
        self.log_interval = 10
        self.use_cuda = True
        self.ngpu=1
        self.save_model = False
        self.size =256
        self.workers = 0
        self.train_batch_size =4
        self.val_batch_size=1
        self.lambda_c = 0.1
        self.lambda_a = 0.1
        self.lambda_d = 0.0001
        self.lr = 0.0002
        self.candi_num = 1
        self.beta1 = 0.5 # beta1 for adam
        self.beta2 = 0.999
        self.decay_rate = 0.5  # Learning rate decay
        self.seed = 666  # random seed to use
        self.outpath = './outputs'  # folder to output images and model checkpoint
        self.alpha = 0.1  # weight given to dice loss while generator training
        self.decay_epoch=70
        self.decay_start_epoch=100
        self.save_start_epoch=100
        self.snapshots=30
        self.save_folder='./'
        self.pretrained_model=''
        self.save_type='solo'
        self.pretrained=False
        self.model_type1='Local'
        self.model_type2 = 'Global'
        self.model_path= './models'
        self.result_path= './result'
        self.mode = 'fedbn'
        self.mode1 = 'fed'
        self.mode2 = 'fed'
        self.type='mixFedGAN'
        self.log_step=2
        self.val_step=2
        self.t=3
        self.img_ch=1
        self.out_ch=1
        self.drop_rate=0.2
        self.ema_consistency=1
        self.start_epoch=0
        self.global_step=0
        self.consistency=0.1
        self.consistency_rampup=200
        self.num_layers_keep=1
        self.frac=0.6
        self.mu=1
        self.temp=3.0
        self.temperature=0.5
        self.weight_l1=1e-3
        self.num_classes=2
        self.ema_decay=0.99
        self.model_p='./outputs/checkpoint_epoch_80.pth'
        self.labeled_bs=4
        self.ict_alpha=0.2
        self. ema_const = 0.95
        self.threshold=0.5
        self.lambda_u=0.5
        self.privacy=False
        self.dp_sigma=1.3
        self.lambda_adv = 0.1
        self.model_buffer_size=1
        self.load_pool_file=None
        self.load_first_net=True
        self.node_num=3
        self.multiplier=1
        self.self_training=True
        self.pool_scale=0.5
        self.adv_loss_type='wgan-gp'
        self.lambda_pi=10
        self.lambda_pa=1
        self.lambda_adv=0.1
        self.nusers=4
        self.epsilon=1.2
        self.ord=2
        self.dp=0.001
        self.gamma= 0.5
        self.agg="avg"
        self.prune = 'prun'
        self.rho=0
        self.quant=True
        self.percent=0.5
        self.compress_mode='l2'
        self.low_threshold = 0.25
        self.high_threshold = 0.75
        self.quant_scheme='mse'
        self.s=0.0001
        self.sr=True

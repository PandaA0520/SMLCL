from main import multiRun

###########################
## ag_news
n_labeled_per_class = 100
bs = 8
ul_ratio = 10
lr = 2e-5
weight_u_loss = 1
psl_threshold_h = 0.98  # 0.98, 0.99
adaptive_threshold = True
num_nets = 2
cross_labeling  = True
weight_disagreement = True
disagree_weight = 0.9
ema_mode = False
ema_momentum = 0.9
val_interval = 25
early_stop_tolerance = 10
max_step = 100000


device_idx = 0
experiment_home = './experiment/ag_news'
dataset = 'ag_news'   # 'ag_news', 'yahoo', 'imdb'

# ATCM
num_runs = 5
num_nets = 2
cross_labeling = True
adaptive_threshold = True
weight_disagreement = 'True'

multiRun(device_idx=device_idx, experiment_home=experiment_home, dataset=dataset, num_runs=num_runs,
        n_labeled_per_class=n_labeled_per_class, bs=bs, ul_ratio=ul_ratio, lr=lr,
        weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, adaptive_threshold=adaptive_threshold,
        num_nets=num_nets, cross_labeling=cross_labeling, 
        weight_disagreement=weight_disagreement, disagree_weight=disagree_weight,
        ema_mode=ema_mode, ema_momentum=ema_momentum,
        val_interval=val_interval, early_stop_tolerance=early_stop_tolerance, max_step=max_step)













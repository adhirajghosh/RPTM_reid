from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.PRETRAIN_CHOICE= 'imagenet'
_C.MODEL.PRETRAIN_PATH= ''
_C.MODEL.ARCH= 'SE_net'
_C.MODEL.DROPRATE= 0
_C.MODEL.STRIDE= 1
_C.MODEL.POOL= 'avg'
_C.MODEL.GPU_ID= ('0')
_C.MODEL.RPTM_SELECT= 'mean'

# ---------------------------------------------------------------------------- #
# Input options
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.HEIGHT= 128
_C.INPUT.WIDTH= 128
_C.INPUT.PROB = 0.5
_C.INPUT.RANDOM_ERASE = True
_C.INPUT.JITTER= True
_C.INPUT.AUG= True

# ---------------------------------------------------------------------------- #
# Dataset options
# ---------------------------------------------------------------------------- #

_C.DATASET = CN()
_C.DATASET.SOURCE_NAME= ['veri']
_C.DATASET.TARGET_NAME= ['veri']
_C.DATASET.ROOT_DIR= ''
_C.DATASET.TRAIN_DIR= ''
_C.DATASET.SPLIT_DIR= ''

# ---------------------------------------------------------------------------- #
# Dataloader options
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.SAMPLER= 'RandomSampler'
_C.DATALOADER.NUM_INSTANCE= 6
_C.DATALOADER.NUM_WORKERS= 16

# ---------------------------------------------------------------------------- #
# Solver options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME= 'SGD'
_C.SOLVER.MAX_EPOCHS= 80
_C.SOLVER.BASE_LR= 0.005
_C.SOLVER.LR_SCHEDULER= 'multi-step'
_C.SOLVER.STEPSIZE= [20, 40, 60]
_C.SOLVER.GAMMA= 0.1
_C.SOLVER.WEIGHT_DECAY= 5e-4
_C.SOLVER.MOMENTUM= 0.9
_C.SOLVER.SGD_DAMP= 0.0
_C.SOLVER.NESTEROV= True
_C.SOLVER.WARMUP_FACTOR= 0.01
_C.SOLVER.WARMUP_EPOCHS= 10
_C.SOLVER.WARMUP_METHOD= 'linear'
_C.SOLVER.LARGE_FC_LR= False
_C.SOLVER.TRAIN_BATCH_SIZE= 20
_C.SOLVER.USE_AMP= True
_C.SOLVER.CHECKPOINT_PERIOD= 10
_C.SOLVER.LOG_PERIOD= 50
_C.SOLVER.EVAL_PERIOD= 1

# ---------------------------------------------------------------------------- #
# Loss options
# ---------------------------------------------------------------------------- #
_C.LOSS = CN()
_C.LOSS.MARGIN= 1.0
_C.LOSS.LAMBDA_HTRI= 1.0
_C.LOSS.LAMBDA_XENT= 1.0

# ---------------------------------------------------------------------------- #
# Test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EVAL= True
_C.TEST.TEST_BATCH_SIZE= 100
_C.TEST.RE_RANKING= True
_C.TEST.VIS_RANK= True
_C.TEST.WEIGHT= ''
_C.TEST.NECK_FEAT= 'after'
_C.TEST.FEAT_NORM= 'yes'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.MISC = CN()
_C.MISC.SAVE_DIR= './logs/veri/'
_C.MISC.GMS_PATH= './gms/veri/'
_C.MISC.INDEX_PATH= './pkl/veri/index_vp.pkl'
_C.MISC.USE_GPU= True
_C.MISC.PRINT_FREQ= 100
_C.MISC.FP16= True
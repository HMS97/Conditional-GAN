from easydict import EasyDict as edict
import socket
__C     = edict()
cfg     = __C

# Common
# __C                               = edict()
__C.NUM_WORKER                    = 4                     # number of data workers
__C.EPOCH                    = 25
__C.LR                       = 1e-2
__C.BATCH_SIZE               =  32
__C.MOMENTUM                          = 0.5
__C.SHOW_INTERVAL                       = 1000
__C.NUM_LABELS                = 10
__C.NZ                      = 100
__C.LR_POLICY = 'step' 
__C.STEP_DEC = 50
__C.UPLOAD = True
__C.SAVE_IMGS = True
__C.ISTRAIN  = True
__C.GPU_IDS = [0]
__C.SAVE_DIR = './saved_models'
__C.SAVE_SAMPLE = './sample'

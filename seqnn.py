import pdb
import sys
import time
from natsort import natsort
import numpy as np
import torch
import torch.nn as nn
from modules import StochasticReverseComplement,StochasticShift

class SeqNN():
    def __init__(self,params):
        super(SeqNN, self).__init__()
        self.set_defaults()
        for key,value in params.items():
            self.__setattr__(key, value)
        self.build_model()
        self.ensemble = None
    def set_defaults(self):
        self.augment_rc = False
        self.augment_shift = [0]
        self.strand_pair = []
        self.verbose = True
    def build_block(self,current,block_params):
        """Construct a SeqNN block.
            Args:
            Returns:
              current
            """
        block_args = {}

        # extract name
        block_name = block_params['name']
    def build_model(self,save_reprs = True):
        ###########################################
        # input
        ###########################################
        sequence = 1
        current = sequence
        # augmentation
        if self.augment_rc:
            current, reverse_bool = StochasticReverseComplement()(current)
        if self.augment_shift != [0]:
            current = StochasticShift(self.augment_shift)(current)
        self.preds_triu = False

        ###################################################
        # build convolution blocks
        ###################################################

        self.reprs = []
        for bi,block_params



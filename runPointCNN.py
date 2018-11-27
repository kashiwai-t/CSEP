# -*- coding: utf-8 -*-
import os
import numpy as np
import pdb
import matplotlib.pylab as plt
import pickle
import pandas as pd
import CSEP

visualPath = 'visualization'

############## MAIN #####################
# pointCNN�@

if __name__ == "__main__":
	cellSize = 0.05				# �Z���̑傫���i���j
	mL = 2.5					# �ŏ��}�O�j�`���[�h
	Ss = [10, 30,50,100]		# �������p�����[�^�i�O���b�h���S����̋���[km]�j
	sTrainDay = '1950-01-01'	# �w�K�̊J�n��
	eTrainDay = '2016-12-31'	# �w�K�̏I����
	sTestDay = '2017-01-01'		# �]���̊J�n��
	eTestDay = '2017-12-31'		# �]���̏I����

	# CSEP�̃f�[�^�N���X
	myCSEP = CSEP.Data(sTrainDay, eTrainDay, sTestDay, eTestDay)
	
	# �f�[�^���X���C�f�B���O�E�B���h�E�ŕ���
	myCSEP.splitData2Slice(winIn=120, winOut=3, stride=1)
	
	# �f�[�^�̍쐬
	myCSEP.makePointCNNData(trainRatio=0.8)

#########################################

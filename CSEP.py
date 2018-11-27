# -*- coding: utf-8 -*-
import os
import numpy as np
import pdb
import matplotlib.pylab as plt
import pickle
import pandas as pd
import pandas.tseries.offsets as offsets
import math
import datetime
from sklearn.mixture import GaussianMixture as GMM
import test

#########################################


class Data:

	#--------------------------
	# データの読み込み
	def __init__(self, sTrain, eTrain, sTest, eTest, mL, dataPath='data'):
		self.sTrain = sTrain
		self.eTrain = eTrain
		self.sTest = sTest
		self.eTest = eTest
		self.dataPath = dataPath
		self.mL = mL
        
		fullPath = os.path.join(self.dataPath,'atr.dat')
		self.data = pd.read_csv(fullPath,sep='\t',index_col='date', parse_dates=['date'])
		
		# 学習データ
		self.dataTrain = self.data[sTrain:eTrain]
		
		# テストデータ
		self.dataTest = self.data[sTest:eTest]
		
	#--------------------------
	#--------------------------
	def term_cul (self,sTday,eTday):
		eDay = datetime.date(int(eTday[0:4]),int(eTday[5:7]),int(eTday[8:10]))
		sDay = datetime.date(int(sTday[0:4]),int(sTday[5:7]),int(sTday[8:10]))

		e_s_days = (eDay-sDay).days

		return e_s_days


	#--------------------------

	#--------------------------
	#GMM（ガウシアンミクスチャーモデリング）
	def gaussian(self,data):
		#pdb.set_trace()
		num = data.shape[0]

		Data = np.array(data['latitude'])
		Data =np.append(Data, np.array(data['longitude'])).reshape((2,num))
		col = Data.shape[1]

		mu = np.mean(Data,axis=0)
		sigma = np.std(Data,axis=0)
		for i in range(col):
			Data[:,i] = (Data[:,i] - mu[i]) / sigma[i]
		return Data

	def exp(self,data):
		pdb.set_trace()
		Data = data
		X_train = self.gaussian(Data)
		N = len(X_train)
		n_components = 2
		pdb.set_trace()
		gmm = GMM(n_components, covariance_type='full')
		gmm.fit(X_train)

		# 結果を表示
		print("*** weights")
		print(gmm.weights_)

		print("*** means")
		print(gmm.means_)

		'''
		print("*** covars")
		print(gmm.covars_)
		'''

		# 訓練データを描画
		plt.plot(X_train[:, 0], X_train[:, 1], 'gx')

		# 推定したガウス分布を描画
		x = np.linspace(-2.5, 2.5, 1000)
		y = np.linspace(-2.5, 2.5, 1000)
		X, Y = np.meshgrid(x, y)

		# 各ガウス分布について
		for k in range(n_components):
			# 平均を描画
			plt.plot(gmm.means_[k][0], gmm.means_[k][1], 'ro')
			# ガウス分布の等高線を描画
			Z = mlab.bivariate_normal(X, Y,
			np.sqrt(gmm.covars_[k][0][0]), np.sqrt(gmm.covars_[k][1][1]),
			gmm.means_[k][0], gmm.means_[k][1],
			gmm.covars_[k][0][1])
			plt.contour(X, Y, Z)
		pdb.set_trace()
		plt.show()

		return 0
	#-------------------


	#--------------------------
	# Train_Num:学習データの一日の平均地震回数（回数/日）
	# obs      :テストデータの観測回数

	def Poisson(self, model, obs, Train_Num, TestTerm):
		# 尤度計算用、ポアソン
		ave = model*(TestTerm*Train_Num)

		den = 1/(math.factorial(int(obs)))

		# ポアソン分布の計算
		poi = (pow(ave,obs)*pow(math.e,(-ave)))*den


		return -math.log(poi)

	#--------------------------

	#-------------------------------------------
	# 尤度関数
	# ポアソン分布を用いて尤度の決定
	# 確率密度の積

	def likelifood(self, model, obs, lats, lons, Mg, Train_Num, TestTerm):
		LL=1.0
		flag = False

		tmpLL = np.zeros([len(lats), len(lons), len(Mg)])
		tmplogLL = np.zeros([len(lats), len(lons), len(Mg)])

		for n, Num in enumerate(Mg):
			for i, lat in enumerate(lats):
				#print("latitude:{}...".format(lat))
				for j, lon in enumerate(lons):
					#tmpLL[i][j][n] = math.log(self.Poisson(model[i][j][n],obs[i][j][n], Train_Num, TestTerm)) # 対数尤度
					tmpLL[i][j][n] = self.Poisson(model[i][j][n],obs[i][j][n], Train_Num, TestTerm) # 尤度
					
		
		pdb.set_trace()
		#LL = np.prod(tmpLL) # 尤度
		LL = np.sum(tmpLL) # 対数尤度
	
		return LL
		#return -math.log(LL)

	'''
	def likelifood(self, model, obs, lats, lons, Train_Num, TestTerm):
		LL=1.0
		flag = False

		tmpLL = np.zeros([len(lats), len(lons)])
		tmplogLL = np.zeros([len(lats), len(lons)])


		for i, lat in enumerate(lats):
			#print("latitude:{}...".format(lat))
			for j, lon in enumerate(lons):
				tmpLL[i][j] == self.poisson(model[i][j],obs[i][j], Train_Num, TestTerm)
				if model[i][j] == 0:
					if obs[i][j] == 0:
						tmpLL[i][j] = 1
						tmplogLL[i][j] = 1.0
					if obs[i][j] > 0:
						LL = 0
						flag = True
						break
				else:
					#tmpLL尤度の計算
					tmpLL[i][j] = math.pow(model[i][j], obs[i][j])*math.pow(math.e, (-1)*model[i][j])/math.factorial(int(model[i][j]))
					#tmplogLL対数尤度の計算
					tmplogLL[i][j] = -1*model[i][j] + obs[i][j] *  math.log(model[i][j]) - math.log(math.factorial(int(model[i][j])))
			if flag:
				break
				
		#LL = np.prod(tmpLL) #尤度の積は小さくなりすぎる??
		#LL = np.sum(tmplogLL) #対数尤度の和

		LL = np.prod(tmpLL)

		return LL
	'''


	#-----------------------------------------------
	#-----------------------------------------------
	# データタイプを選択し、そのデータを返す
	# all  :気象庁データの全て（1923〜2016）
	# train:学習データ
	# test :テストデータ

	def Data_sel(self, dataType='train'):
		if dataType=='all':
			data = self.data
		elif dataType=='train':
			data = self.dataTrain
		elif dataType=='test':
			data = self.dataTest
		Data = data[data['magnitude'].values > (self.mL-0.05)]	
		return Data

	#------------------------------------------------


	def splitTestDataInGrid(self, sLat, sLon, eLat, eLon, term, data):
		date = datetime.datetime.strptime(self.sTest, '%Y-%m-%d') + datetime.timedelta(term)
		datestr = date.strftime("%Y-%m-%d")
		pdb.set_trace()	
		tmpData = data[(data['date'].values >= self.sTest) & (data['date'].values <= datestr)]
		if (len(tmpData) == 0 ):
			tmpData = np.arange(1)
		return tmpData


	#--------------------------
	# グリッド内のデータ取り出し
	# sLat: 開始緯度
	# sLon: 開始経度
	# eLat: 終了緯度
	# eLon: 終了経度
	# dataType: all, train, test
	def getDataInGrid(self, sLat, sLon, eLat, eLon, Data):
				
		# Indexを数字で割り振る
		d = np.arange(len(Data))
		Data['No'] = d

		# magnitudeの強さだけの配列
		D_mg = np.array(Data['magnitude'])

		# D_mgは何番目のビンかを決める
		for i ,D in enumerate(D_mg):
			D_mg[i] = round((D-self.mL)*10,1) + 1

		'''
		tmpData = Data[(Data['latitude'] >= sLat) & (Data['latitude'] < eLat) &
		 (Data['longitude'] >= sLon)  & (Data['longitude'] < eLon)]
		''' 
		
		x_data = np.array(Data[(Data['latitude'] >= sLat) & (Data['latitude'] < eLat) & (Data['longitude'] >= sLon)  & (Data['longitude'] < eLon)]['No'])
		# 範囲内に地震がなければ
		if (len(x_data) == 0 ):
			tmpData = np.arange(1)
		
		# 範囲内に地震があれば
		else:
			tmpData = D_mg[x_data]

		return tmpData
	#--------------------------
	
	#--------------------------
	# sliding windowでデータを分割
	# winIn: 入力用のウィンドウ幅（単位：月）
	# winOut: 出力用のウィンドウ幅（単位：月）	
	# stride: ずらし幅（単位：月）
	def splitData2Slice(self, winIn=120, winOut=3, stride=1):
	
		# ウィンドウ幅と、ずらし幅のoffset
		winInOffset = offsets.DateOffset(months=winIn, days=-1)
		winOutOffset = offsets.DateOffset(months=winOut, days=-1)
		strideOffset = offsets.DateOffset(months=stride)
		
		# 学習データの開始・終了のdatetime
		sTrainDT = pd.to_datetime(self.sTrain)
		eTrainDT = pd.to_datetime(self.eTrain)
		
		#---------------
		# 各ウィンドウのdataframeを取得
		self.dfX = []
		self.dfY = []
		
		# 現在の日時
		currentDT = sTrainDT
		endDTList = [] # Saito temporarily added (7/9)
		while currentDT + winInOffset + winOutOffset <= eTrainDT:
			endDTList.append(currentDT+winInOffset) # Saito temporarily added (7/9)
		
			# 現在の日時からwinInOffset分を抽出
			self.dfX.append(self.dataTrain[currentDT:currentDT+winInOffset])

			# 現在の日時からwinInOffset分を抽出
			self.dfY.append(self.dataTrain[currentDT+winInOffset:currentDT+winInOffset+winOutOffset])
			
			# 現在の日時をstrideOffset分ずらす
			currentDT = currentDT + strideOffset
		#---------------
        
		return self.dfX, self.dfY, endDTList, # Saito temporarily added (7/9)
	#--------------------------

	#--------------------------
	# pointCNN用のデータ作成
	def makePointCNNData(self, trainRatio=0.8):
		# 学習データとテストデータ数
		self.nData = len(self.dfX)
		self.nTrain = np.floor(self.nData * trainRatio).astype(int)
		self.nTest = self.nData - self.nTrain
		
		# ランダムにインデックスをシャッフル
		self.randInd = np.random.permutation(self.nData)
		
		
		# 学習データ
		self.xTrain = self.dfX[self.randInd[0:self.nTrain]]
		self.yTrain = self.dfY[self.randInd[0:self.nTrain]]

		# 評価データ
		self.xTest = self.dfX[self.randInd[self.nTrain:]]
		self.yTest = self.dfY[self.randInd[self.nTrain:]]
		
		
		# ミニバッチの初期化
		self.batchCnt = 0
		self.batchRandInd = np.random.permutation(self.nTrain)
		#--------------------		
	#--------------------------
	
	#------------------------------------
	# pointCNN用のミニバッチの取り出し
	def nextPointCNNBatch(self,batchSize):

		sInd = batchSize * self.batchCnt
		eInd = sInd + batchSize
		'''
		batchX = []
		batchY = []
		'''
		batchX = self.xTrain[self.batchRandInd[sInd:eInd]]
		batchY = self.yTrain[self.batchRandInd[sInd:eInd]]
		
		
		if eInd+batchSize > self.nTrain:
			self.batchCnt = 0
		else:
			self.batchCnt += 1

		return batchX, batchY
	#------------------------------------
		
	#--------------------------
	# ヒュベニの公式を用いた緯度・経度座標系の2点間の距離(km)
	# https://qiita.com/chiyoyo/items/b10bd3864f3ce5c56291
	# を参考にして作成
	# lat1: 1点目の緯度
	# lon1: 1点目の経度
	# lat2: 2点目の緯度
	# lon2: 2点目の経度	
	# mode: 測地系の切り替え
	def deg2dis(self, lat1, lon1, lat2, lon2, mode=True):
		#lat2 = data['latitude'].values
		#lon2 = data['longitude'].values
		
		# 緯度経度をラジアンに変換
		radLat1 = lat1/180*np.pi # 緯度１
		radLon1 = lon1/180*np.pi # 経度１
		radLat2 = lat2/180*np.pi # 緯度２
		radLon2 = lon2/180*np.pi # 経度２
		
		# 緯度差
		radLatDiff = radLat1 - radLat2

		# 経度差算
		radLonDiff = radLon1 - radLon2;

		# 平均緯度
		radLatAve = (radLat1 + radLat2) / 2.0

		# 測地系による値の違い
		a = [6378137.0 if mode else 6377397.155][0]						# 赤道半径
		b = [6356752.314140356 if mode else 6356078.963][0]				# 極半径
		e2 = [0.00669438002301188 if mode else 0.00667436061028297][0]	# 第一離心率^2
		a1e2 = [6335439.32708317 if mode else 6334832.10663254][0]		# 赤道上の子午線曲率半径

		sinLat = np.sin(radLatAve)
		W2 = 1.0 - e2 * (sinLat**2)
		M = a1e2 / (np.sqrt(W2)*W2)		# 子午線曲率半径M
		N = a / np.sqrt(W2)				# 卯酉線曲率半径

		t1 = M * radLatDiff;
		t2 = N * np.cos(radLatAve) * radLonDiff
		dist = np.sqrt((t1 * t1) + (t2 * t2))

		return dist/1000
	#--------------------------



#########################################

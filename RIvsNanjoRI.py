# -*- coding: utf-8 -*-
import os
import numpy as np
import pdb
import matplotlib.pylab as plt
import pickle
import pandas as pd
import CSEP
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import time
import datetime
from sklearn.mixture import GaussianMixture as GMM

visualPath = 'visualization'

############## MAIN #####################
# RI法：グリッドごとに地震回数をカウント
# NanjoRI法：グリッドごとの地震回数を半径Sの円に入るグリッド数で平滑化

if __name__ == "__main__":
	cellSize = 0.05				# セルの大きさ（°）
	mL = 5.0					# 最小マグニチュード
	#Ss = [10,30,50,100]		# 平滑化パラメータ（グリッド中心からの距離[km]）
	Ss = [50]		# 平滑化パラメータ（グリッド中心からの距離[km]）

	sTrainDay = '1980-01-01'	# 学習の開始日
	eTrainDay = '2016-12-31'	# 学習の終了日

	#sTrainDay = '2016-01-01'	# 学習の開始日 # データ量を減らして
	#eTrainDay = '2016-12-31'	# 学習の終了日 # データ量を減らして

	sTestDay = '2017-01-01'		# 評価の開始日
	eTestDay = '2017-12-31'		# 評価の終了日


	  

	# CSEPのデータクラス
	myCSEP = CSEP.Data(sTrainDay, eTrainDay, sTestDay, eTestDay, mL)
	
	# CSEP関東領域グリッド（lon:138.475-141.525, lat:34.475-37.025）
	lats = np.arange(34.475, 37.025, cellSize)
	lons = np.arange(138.475, 141.525, cellSize)
	
	# CSEPマグニチュード別ビン
	Mg = np.arange(mL,9.1,0.1)

	# CSEPの学習期間とテスト期間
	TrainTerm = myCSEP.term_cul(sTrainDay,eTrainDay)

	# マグニチュードの発生回数
	numsRI = np.zeros([len(lats), len(lons),len(Mg)])			# RI法
	numsNanjoRI = np.zeros([len(lats), len(lons),len(Mg)])		# NanjoRI法

	# セル中心
	cellCsFlat = np.array([[lat + cellSize/2, lon + cellSize/2] for lat in lats for lon in lons])
	cellCs = np.reshape(cellCsFlat, [len(lats), len(lons), 2])
	
	trainData = myCSEP.Data_sel(dataType='train') # trainDataの選別

	pdb.set_trace()
	#x=myCSEP.exp(trainData)

	for S in Ss:
		a=0
		print("S:{}km".format(S))
		for i, lat in enumerate(lats):
			print("latitude:{}...".format(lat))
			for j, lon in enumerate(lons):
				tmpData = myCSEP.getDataInGrid(lat, lon, lat+cellSize, lon+cellSize, trainData)
				#-------------
				# RI法：各セルのマグニチュードmL以上の地震をマグニチュード毎カウント
				'''
				if(tmpData['magnitude'].values > mL):
					numsRI[i,j,k]
				'''
				if(tmpData[0]!=0):
					tmpData = tmpData - 1
					for Num in tmpData:
						a=a+1
						print(a)
						numsRI[i,j,int(Num)] += 1
				
				#numsRI[i,j] = np.sum(tmpData['magnitude'].values > mL)
				#-------------
			
				#-------------
				# Nanjo RI法：各セルのマグニチュードmL以上の地震をカウントし、
				# 距離S以内のセル全てに(Nsb+1)^-1をカウント
				
				# セル中心からS以内にセル中心があるセルのインデックスとセル数Nsbを取得
						dists = myCSEP.deg2dis(cellCs[i,j,0],cellCs[i,j,1], cellCsFlat[:,0],cellCsFlat[:,1])
						latInds,lonInds = np.where(np.reshape(dists,[len(lats),len(lons)])<S)	# インデックス
						Nsb = len(latInds)	# セル数
			
				# (Nsb + 1)^-1を割り当てる
						for k, l in zip(latInds, lonInds):
							numsNanjoRI[k,l,int(Num)] += (numsRI[i,j,int(Num)]*(1/(Nsb + 1)))
				#-------------
			
		# カウントされた地震の保存
		numsRI_o = numsRI
					  
		EQ_num = a
		#-------------------
		# 最も良い尤度の時のSパラメータの選択





		#---------------------

		#---------------------	
		# 正規化
		numsRI = numsRI / np.sum(numsRI)
		numsNanjoRI = numsNanjoRI / np.sum(numsNanjoRI)
		#---------------------	

		


	
		#---------------------
		# プロット
		'''
		plt.close()
		fig, figInds = plt.subplots(ncols=2)
		figInds[0].imshow(numsRI,cmap="bwr")
		figInds[0].set_title('Relative Intensity')

		figInds[1].imshow(numsNanjoRI,cmap="bwr")
		figInds[1].set_title('Nanjo Relative Intensity')

		fullPath = os.path.join(visualPath,'RIvsNanjoRI_{}km.png'.format(S))
		plt.savefig(fullPath)
		'''
		

	'''
	for n , Num in enumerate(Mg): 
		plt.close()
		flag=0
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		for i,lat in enumerate(lats):
			for j,lon in enumerate(lons):
				if(numsRI[i,j,n] ==0):
					c = 'b'
					m = 'o'
				else:
					c = 'r'
					m = '^'
					flag=1
					print('numsRI({},{},{}) = {}'.format(i,j,n,numsRI[i,j,n]))
				ax.scatter(lat,lon,numsRI[i,j,n], c = c, marker=m)
				ax.set_xlabel('latitude')
				ax.set_ylabel('lontitude')
				ax.set_zlabel('Magnitude = {}'.format(Num))
		if(flag!=0):
			plt.show()

	for n , Num in enumerate(Mg): 
		plt.close()
		flag=0
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		for i,lat in enumerate(lats):
			for j,lon in enumerate(lons):
				if(numsNanjoRI[i,j,n] ==0):
					c = 'b'
					m = 'o'
				else:
					c = 'r'
					m = '^'
					flag=1
					print('numsNanjoRI({},{},{}) = {}'.format(i,j,n,numsNanjoRI[i,j,n]))
				ax.scatter(lat,lon,numsNanjoRI[i,j,n], c = c, marker=m)
				ax.set_xlabel('latitude')
				ax.set_ylabel('lontitude')
				ax.set_zlabel('Magnitude = {}'.format(Num))
		if(flag!=0):
			plt.show()
	'''


		#---------------------
	# add masuyama

	Days = [1, 10,30,365]		# 評価期間（テストデータのスタートからの日数[day]）

	# マグニチュードの発生回数(sTrain~評価期間終了までの)
	obsRI = np.zeros([len(lats), len(lons),len(Mg)])

	testData = myCSEP.Data_sel(dataType='test') # testDataの選別
	for k, term in enumerate(Days):
		obsRI = np.zeros([len(lats), len(lons),len(Mg)])
		print("term: {}days".format(term))
		for i, lat in enumerate(lats):
			#print("latitude:{}...".format(lat))
			for j, lon in enumerate(lons):
				tmpDATA = myCSEP.getDataInGrid(lat, lon, lat+cellSize, lon+cellSize, testData)
				if (tmpDATA[0] != 0):
					tmpData = myCSEP.splitTestDataInGrid(lat, lon, lat+cellSize, lon+cellSize, term, tmpDATA)
				#-------------
				# 各セルのマグニチュードmL以上の地震をカウント

				if(tmpDATA == 0):
					break
				else:
					for Num in tmpData:
						obsRI[i,j,int(Num)] += 1



						
				#obsRI[i, j] = np.sum(tmpData['magnitude'].values > mL)
				#-------------
	
		#obsRI=観測データ（トレーニングとテストの地震のカウント数を合計）
		obsRI_o = obsRI

		obsRI = numsRI_o + obsRI
		obsRI = obsRI / np.sum(obsRI) # 正規化


		#-------------------------------
		# plot
		'''
		for n , Num in enumerate(Mg): 
			plt.close()
			flag=0
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			for i,lat in enumerate(lats):
				for j,lon in enumerate(lons):
					if(obsRI_o[i,j,n] ==0):
						c = 'b'
						m = 'o'
					else:
						c = 'r'
						m = '^'
						flag=1
						print('obsRI({},{},{}) = {}'.format(i,j,n,obsRI_o[i,j,n]))
					ax.scatter(lat,lon,obsRI_o[i,j,n], c = c, marker=m)
					ax.set_xlabel('latitude')
					ax.set_ylabel('lontitude')
					ax.set_zlabel('Magnitude = {}'.format(Num))
			if(flag!=0):
				plt.show()
		'''
		#--------------------------------


		
		#モデルの尤度を計算
		#'''
		Train_Num = EQ_num / TrainTerm
		LL = myCSEP.likelifood(numsRI, obsRI_o, lats, lons, Mg, Train_Num, term)
		print('LL numsRI : {}'.format(float(LL)))
		
		LL = myCSEP.likelifood(numsNanjoRI, obsRI_o, lats, lons, Mg, Train_Num, term)
		print('LL numsNanjoRI : {}'.format(float(LL)))
		#'''
		#---------------------	

#########################################
#########################################

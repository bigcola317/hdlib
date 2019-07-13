#!/usr/bin/env python3

''' 
=================
HD encoding class
=================
'''
import torch as t


__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "17.5.2019"



class hd_encode():
	def __init__(self,D,encoding,device,nitem=1,ngramm = 3, sparsity: int=90, resolution: int=100):
		'''	
		Encoding 
		Parameters
		----------
		encoding: string 
			Encoding architecture {"sumNgramm"}
		nitem: int
			number of items in itemmemory 
		ngramm: int
			number of ngramms
		sparsity: int
			number of 0s as a fraction of resolution
		'''
		self._D = D
		self._device = device
		# encoding scheme 
		if encoding =="sumNgramm":
			self.encode = self._compute_sumNgramm
			self._nitem = nitem
			self._ngramm = ngramm
			# malloc for Ngramm block, ngramm result, and sum vector  
			self._block = t.Tensor(self._ngramm,self._D).zero_().to(self._device)
			self._Y = t.Tensor(self._D).to(self._device)
			self._SumVec= t.Tensor(self._D).zero_().to(self._device)

		else: 
			raise ValueError("No valid encoding! got "+ code)

		# item memory initialization 
		self._generateItemMemory(sparsity, resolution)
		# print("Item memories:")
		# print(self._itemMemory)

		return



	def _generateItemMemory(self, sparsity: int, resolution: int):
		'''
		Generates sparse random vectors for all items. Sparsity level is
		defined by a given integer percentage
		'''
		self._itemMemory = t.randint(1, resolution+1, (self._nitem, self._D)).to(self._device)
		for i in range(self._nitem):
			for j in range(self._D):
				condition = self._itemMemory[i][j] > sparsity
				if condition:
					self._itemMemory[i][j] = 1
				else:
					self._itemMemory[i][j] = 0


	def _charToKey(self, char):
		return ord(char)-ord('a')

	def _lookupItemMemory(self,key):
		'''	
		Encoding 
		Parameters
		----------
		key: int 
			key to itemmemory
		Return
		------
		out: Torch tensor, size=[D,]
		'''
		return self._itemMemory[key]


	def _compute_sumNgramm(self,X,clip=False):
		'''	
		compute sum of ngramms 
		Parameters
		----------
		X: torch tensor, size = [n_samples,n_feat] 
			feature vectors
		Return
		------

		'''
		# reset block to zero
		self._block.zero_().to(self._device)
		self._SumVec.zero_()


		n_samlpes,n_feat = X.shape
		add_cnt = 0
		
		for feat_idx in range(n_feat):
			# X[0] is text, and feat_idx is current letter in text
			ngramm = self._ngrammencoding(X[0],feat_idx)
			if feat_idx >= self._ngramm-1:
				self._SumVec.add_(ngramm)
				add_cnt +=1

		if clip: 
			self._SumVec = self._threshold(self._SumVec,add_cnt)
			add_cnt = 1
			
		# put here clipping option 
		return self._SumVec, add_cnt

	def _ngrammencoding(self,X,start):
		'''	
		Load next ngramm

		Parameters
		----------
		X: Torch tensor, size = [n_samples, D]
			Training samples 

		Results
		-------
		Y: Torch tensor, size = [D,]
		'''

		# rotate shift current block 
		for i in range(self._ngramm-1,0,-1): 
			self._block[i] = self._circshift(self._block[i-1],1)
		# write new first entry in shift register, X[start] is current letter in text
		self._block[0] = self._lookupItemMemory(X[start])

		# calculate ngramm of _block (_Y)
		self._Y = self._block[0]
		
		for i in range(1,self._ngramm):
			self._Y = self._bind(self._Y,self._block[i])

		return self._Y

	def _wordGrammEncoding(self, word):
		'''
		Compute ngramm encoding for a given word
		:param word: word to encode
		:return: return word ngramm
		'''

		n = len(word)
		# alloc shift register
		shift_reg = t.ShortTensor(n, self._D).zero_()
		# fill shift register with initial item memories
		for i, letter in enumerate(word):
			key = self._charToKey(letter)
			if key >= self._nitem or key < 0:
				print("Error! Key not valid: char = {}".format(letter))
			else:
				shift_reg[n-i-1] = self._lookupItemMemory(key)
		# print("Letter encodings")
		# print(shift_reg)
		# shift item memories
		for i in range(1, n):
			shift_reg[i] = self._circshift(shift_reg[i], i)
		# print("Shifted letter encodings")
		# print(shift_reg)

		# calculate ngramm of _block (_Y)
		wordgramm = shift_reg[0]
		# print("XOR 0: {}".format(wordgramm))

		for i in range(1, n):
			wordgramm = self._bind(wordgramm, shift_reg[i])
			# print("XOR {}: {}".format(i, wordgramm))

		return wordgramm


	def encodeText(self, text):

		words = text.split()
		# print(words)
		res = t.ShortTensor(self._D).zero_()

		for word in words:
			ngramm = self._wordGrammEncoding(word)
			res = res | ngramm
			# print("Ngramm: {}\nRes: {}\n".format(ngramm, res))

		return res


	def _circshift(self,X,n):
		'''	
		Load next ngramm

		Parameters
		----------
		X: Torch tensor, size = [D,]
			

		Results
		-------
		Y: Torch tensor, size = [n_samples-n]
		'''
		return t.cat((X[-n:], X[:-n]))

	def _bind(self,X1,X2): 
		'''	
		Bind two vectors with XOR 

		Parameters
		----------
		X1: Torch tensor, size = [D,]
			input vector 1 
		X2: Torch tensor, size = [D,]
			input vector 2 
			
		Results
		-------
		Y: Torch tensor, size = [D,]
			bound vector
		'''
		# X1!= X2
		return ((t.mul((-2*X1+1), (2*X2-1))+1)/2)

	def _threshold(self,X,cnt):
		'''	
		Threshold a vector to binary 
		Parameters
		----------
		X : Torch tensor, size = [D,]
			input vector to be thresholded
		cnt: int 
			number of added binary vectors, used for determininig threshold 
			
		Results
		-------
		Y: Torch tensor, size = [D,]
			thresholded vector
		'''
		# even 
		if cnt % 2 == 0: 
			X.add_(t.randint(0,2,(self._D,)).type(t.FloatTensor).to(self._device)) # add random vector 
			cnt += 1
		
		return (X > (cnt/2)).type(t.cuda.FloatTensor)


	




		

		

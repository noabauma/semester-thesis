#!/usr/bin/env python
from mirheo_scripts import *

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def model(s):
	A_eps 	= s["Parameters"][0]
	B		= s["Parameters"][1]
	sigma	= s["Parameters"][2]	#SW2 sigma
	rc		= s["Parameters"][3]
	sig 	= s["Parameters"][4]	#[sigma]
	
	wca_grs 	 = droplet_grs(A_eps * 6.947695457055374e-21, B, sigma, rc)
	wca_cnt 	 = droplet_cnt(A_eps * 6.947695457055374e-21, B, sigma, rc)
	eta, lambda_ = water_in_cnt(A_eps * 6.947695457055374e-21, B, sigma, rc)


	result = [wca_grs, wca_cnt, eta, lambda_]
	sdev   = [sig*abs(wca_grs), sig*abs(wca_cnt), sig*abs(eta), sig*abs(lambda_)]
	
	s["Reference Evaluations"] = result
	s["Standard Deviation"]    = sdev


#This numbers are arbitrary
def getReferenceData():
  y=[]
  y.append(86.0)		#WCA on graphene		[°Degree]		(Werder et al. 2003)
  y.append(106.925)		#WCA in CNT	(96,0)		[°Degree]		(Werder et al. 2001)
  y.append(0.775)		#Viscosity 				[mPa*s]			(Thomas et al. 2010)
  y.append(12000.0)		#Friction coefficient	[kg/(m*s)]		(Falk et al. 2010)
  return y

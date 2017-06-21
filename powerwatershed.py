import time
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import random_walker
from skimage.data import *
from skimage.exposure import rescale_intensity
import skimage
def grey_weights_PW(rs,cs,image_name,edges,seeds,size_seeds,weights) : 
	M = rs*(cs-1)+(rs-1)*cs #number of edges
	numvoisins = 4
	for i in range(0,M) :
		normal_weights[i]=  255-checkabs(img[edges[0][i]]-img[edges[1][i]])
	for j in range(0,size_seeds):
		for k in range(1,numvoisin+1):
			n = neighbor_node_edge(seeds[j], k, rs, cs)
			if n != -1 :
				seeds_function[n]= normal_weights[n]
	gageodilate_union_find(seeds_function, normal_weights,edges,rs,cs,255)


def checkabs(x):
	if x >= 0:
		return x
	else :
		return -x
		
def neighbor_node_edge(i ,k ,rs,cs) : #return the index of the k_th edge neighbor of the node "i" 
	rs_cs = rs*cs;                       #return -1 if the neighbor is outside the image
	zp = i % (rs_cs);
	z = i / (rs_cs);
	V = (cs-1)*rs;
	H = (rs-1)*cs;
	if k == 1 :
		if (zp % rs >= rs-1) :
			return -1
		else :
			return (zp+V)-(zp/rs)+z*(V+H+rs_cs)
	elif k ==2 :
		if (zp / rs >= cs-1) :
			return -1
		else :
			return zp+z*(V+H+rs_cs)
	elif k==3 :
		if zp % rs == 0 :
			return -1
		else  :
			return (zp+V)-(zp/rs)-1+z*(V+H+rs_cs)
	elif k == 4:
		if zp / rs == 0 :
			return -1;
		else :
			return zp-rs+z*(V+H+rs_cs)
	elif k == -1:
		return i+rs_cs
	elif k ==  0:
		return i + rs_cs*2
	
	
def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""
    assert pgmf.readline() == 'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster
	
def rowsize(pgmf):
	"""Return a raster of integers from a PGM as a list of lists."""
	assert pgmf.readline() == 'P5\n'
	(width, height) = [int(i) for i in pgmf.readline().split()]
	return height
	
def colsize(pgmf):
	"""Return a raster of integers from a PGM as a list of lists."""
	assert pgmf.readline() == 'P5\n'
	(width, height) = [int(i) for i in pgmf.readline().split()]
	return width
	
def compute_edges(edges,rs,cs):
	M = 0
	for i in range(cs):
		for j in range(rs):
			if i < (cs - 1) :
				edges[0][M] = j+i*rs
				edges[1][M]=j+(i+1)*rs
				M += 1
	for i in range(cs):
		for j in range(rs):
			if j < (rs - 1) :
				edges[0][M]=j+i*rs
				edges[1][M]=j+1+i*rs
				M += 1
	return edges
	
result = []

def gageodilate_union_find(seed,imageWeight,edge,rs,cs,max_weight) :
	global result
	M = rs*(cs-1)+(rs-1)*cs  
	for k in range(0,k) :
		Fth[k]=k
		result[k]=seed[k]
		seed[k]=imageWeight[k]
		Es[k] = k #E sorted by decreasing weights
		
	Max = max_weight
	seed.sort()
	for k in range(M-1,-1,-1) :
		p = Es[k]
		for i in range(1,7) :
			n = neighbor_edge(p,i,rs,cs)
			if n != -1 :
				if Mrk[n] == 0 :
					element_link_geod_dilate(n, p, Fth, imageWeight, result,Max)
			Mrk[p] = True
	for k in range(0,M) :
		p = Es[k]
		if Fth[p] == p :
			if result[p] == MAX :
				result[p] = imageWeight[p]
		else :
			result[p] = result[Fth[p]]

def element_link_geod_dilate(n,p,Fth,imageWeight,result,Max) :
	r = element_find(n, Fth)
	if r != p :
		if (imageWeight[r] == imageWeight[p]) or (imageWeight[p] >= result[r]) :
			Fth[r] = p
			result[p] = mcmax(result[r], result[p])
		else :
			result[p] = Max
	

def mcmax(a,b) :
	if a >= b :
		return a
	else :
		return b
		
def neighbor_edge(i,k,rs,cs) :
	V = (cs-1)*rs; 
	if (i >= V) :
		if k == 1 :
			if (i-V<rs-1) :
				return -1
			else :
				return ((i-V)/(rs-1)-1)*rs + ((i - V)%(rs-1))+1
		elif k ==2 :
			if (i-V) < rs-1 :
				return -1
			else :
				return ((i-V)/(rs-1)-1)*rs + ((i - V)%(rs-1))
		elif k==3 :
			if (i-V)%(rs-1) == 0 :
				return -1
			else  :
				return i-1
		elif k == 4:
			if i>(rs-1)*cs+ V -rs:
				return -1;
			else :
				return ((i-V)/(rs-1)-1)*rs + ((i - V)%(rs-1)) + rs
	else:
		if k == 1 :
			if i < rs :
				return -1
			else :
				return i-rs
		elif k ==2 :
			if i%rs == 0 :
				return -1
			else :
				return (i+V)-(i/rs)-1
		elif k==3 :
			if i%rs == 0 :
				return -1
			else  :
				return (i+V)-(i/rs)-1+rs-1
		elif k == 4:
			if i>=V-rs:
				return -1;
			else :
				return i+rs

def PowerWatershed_q2(edge,weights,normal_weights,max_weight,seed,lable,size_seeds,rs,cs,nb_labels,img_proba) :
	N = rs * cs
	M = rs*(cs-1)+(rs-1)*cs
	
	for i in range(0,nb_labels-1) :
		for j in range(0,N) :
			proba[i][j]=-1
	for i in range(0,size_seeds) :
		for j in range(nb_labels-1) :
			if lable[i] == j+1 :
				proba[j][seed[i]] = 1
			else :
				proba[j][seed[i]] = 0
	for k in range(0,N):
		Fth[k] = k
	for k in range(0,M):
		sorted_weights[k] = weights[k]
		Es[k] = k
	sorted_weights.sort()
	cpt_aretes = 0
	Ncpt_aretes = 0
	
	while (cpt_aretes < M) :
		while True :
			e_max = Es[cpt_aretes]
			cpt_aretes = cpt_aretes + 1
			if cpt_aretes == M:
				break
			if indic_E[e_max] != True :
				break
		if cpt_aretes == M :
			break


def watershedPower(input_file, output_file,seeds_file, algo , mult, geod):
	image_name = "figure_1.png"

	seeds_name = seeds_file
	sigma = 0.35
	
	fimage_read = open(image_name,'rb')
	fimage_seeds = open(seeds_name,'rb')
	
	image_read = read_pgm(fimage_read)
	seeds = read_pgm(fimage_seeds)
	
	
	rs = rowsize(fimage_read)
	cs = colsize(fimage_read)
	
	N = rs * cs 
	M = rs*(cs-1)+(rs-1)*cs #number of edges

	
	nblabels = 2
	j = 0
	for i in range(0,rs*cs):
		if seeds[i] > 155 :
			index_seeds[j]=i
			index_labels[j]=1
			j+=1
		if seeds[i] < 100 :
			index_seeds[j]=i
			index_labels[j]=2
			j+=1
	
	size_seeds = j
	markers = np.zeros(image.shape, dtype=np.uint)
	markers[image < -0.95] = 1
	markers[image > 0.95] = 2
	matrixedges = [[0 for col in range(2)] for row in range((cs-1)*rs+(rs-1)*cs) ]
	edges = compute_edges(matrixedges,rs,cs)
	weights = []
	normal_weights = grey_weights_PW(image_name,  edges,index_seeds, size_seeds, weights, quicksort)

	output = PowerWatershed_q2(edges, weights, normal_weights,max_weight,index_seeds, index_labels, size_seeds,rs, cs, ds, nblabels, quicksort, img_proba);
	labels = random_walker(image, markers, beta=10, mode='bf')
data = skimage.img_as_float(horse())
sigma = 0.35
data += np.random.normal(loc=0, scale=sigma, size=data.shape)
data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                         out_range=(-1, 1))

markers = np.zeros(data.shape, dtype=np.uint)
markers[data < -0.95] = 1
markers[data > 0.95] = 2
labels = random_walker(data, markers, beta=10, mode='bf')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                    sharex=True, sharey=True)
ax1.imshow(data, cmap='gray', interpolation='nearest')
ax1.axis('off')
ax1.set_adjustable('box-forced')
ax1.set_title('original data')
ax2.imshow(markers, cmap='magma', interpolation='nearest')
ax2.axis('off')
ax2.set_adjustable('box-forced')
ax2.set_title('Markers')
ax3.imshow(labels, cmap='gray', interpolation='nearest')
ax3.axis('off')
ax3.set_adjustable('box-forced')
ax3.set_title('Segmentation')

fig.tight_layout()
plt.show()
	


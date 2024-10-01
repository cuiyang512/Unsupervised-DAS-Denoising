import numpy as np


def yc_patch(A,l1,l2,o1,o2):

    n1,n2=np.shape(A);
    tmp=np.mod(n1-l1,o1)
    if tmp!=0:
        #print(np.shape(A), o1-tmp, n2)
        A=np.concatenate([A,np.zeros((o1-tmp,n2))],axis=0)

    tmp=np.mod(n2-l2,o2);
    if tmp!=0:
        A=np.concatenate([A,np.zeros((A.shape[0],o2-tmp))],axis=-1);


    N1,N2 = np.shape(A)
    X=[]
    for i1 in range (0,N1-l1+1, o1):
        for i2 in range (0,N2-l2+1,o2):
            tmp=np.reshape(A[i1:i1+l1,i2:i2+l2],(l1*l2,1));
            X.append(tmp);
    X = np.array(X)
    return X[:,:,0]

def yc_patch_inv(X1, n1, n2, l1, l2, o1, o2):
    tmp1 = np.mod(n1 - l1, o1)
    tmp2 = np.mod(n2 - l2, o2)
    if (tmp1 != 0) and (tmp2 != 0):
        A = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))
        mask = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))

    if (tmp1 != 0) and (tmp2 == 0):
        A = np.zeros((n1 + o1 - tmp1, n2))
        mask = np.zeros((n1 + o1 - tmp1, n2))

    if (tmp1 == 0) and (tmp2 != 0):
        A = np.zeros((n1, n2 + o2 - tmp2))
        mask = np.zeros((n1, n2 + o2 - tmp2))

    if (tmp1 == 0) and (tmp2 == 0):
        A = np.zeros((n1, n2))
        mask = np.zeros((n1, n2))

    N1, N2 = np.shape(A)
    ids = 0
    for i1 in range(0, N1 - l1 + 1, o1):
        for i2 in range(0, N2 - l2 + 1, o2):
            # print(i1,i2)
            #       [i1,i2,ids]
            A[i1:i1 + l1, i2:i2 + l2] = A[i1:i1 + l1, i2:i2 + l2] + np.reshape(X1[:, ids], (l1, l2))
            mask[i1:i1 + l1, i2:i2 + l2] = mask[i1:i1 + l1, i2:i2 + l2] + np.ones((l1, l2))
            ids = ids + 1

    A = A / mask;
    A = A[0:n1, 0:n2]
    return A

def snr(g,f,mode=1):
	"""
	SNR: calculate the signal-to-noise ratio (SNR)
	
	INPUT
	g: 		ground truth image
	f: 		noisy/restored image
	mode:	1->2D SNR, 2->3D SNR
	
	OUTPUT
	snr: 	SNR value
	
	The definition of SNR can be found in 
	Chen and Fomel, 2015, Random noise attenuation using local
	signal-and-noise orthogonalization, Geophysics.
	
	Author: Yangkang Chen, 2015
	"""
	
	import numpy as np

	if g.ndim==2:
		g=np.expand_dims(g, axis=2)

	if f.ndim==2:
		f=np.expand_dims(f, axis=2)
		
	g = np.double(g); #in case of data format is unit8,12,16
	f = np.double(f);

	if f.size != g.size:
		print('Dimesion of two images don''t match!');

	if mode ==1:
		s = g.shape[2];
		if s==1: #single channel	
			psnr = 20.*np.log10(np.linalg.norm(g[:,:,0],'fro')/np.linalg.norm(g[:,:,0]-f[:,:,0],'fro'));   
		else: #multi-channel
			psnr = np.zeros(s);
			for i in range(0,s):
				psnr[i] = 20.*np.log10(np.linalg.norm(g[:,:,i],'fro')/np.linalg.norm(g[:,:,i]-f[:,:,i],'fro'));

	else:
		[n1,n2,n3]=g.shape;
		psnr = 20.*np.log10(np.linalg.norm(g.reshape(n1,n2*n3,order='F'),'fro')/np.linalg.norm(g.reshape(n1,n2*n3,order='F')-f.reshape(n1,n2*n3,order='F'),'fro'));   

	return psnr

def scale(D,N=2,dscale=1.0):
	"""
	scale: Scale the data up to the Nth dimension = sfscale axis=N
	IN   D:   	intput data
	     N:      number of dimension for scaling
	             default: N=2
		dscale:  Scale by this factor
	    (does not include the rscale and pclip functions (not convenient actually))
 
	OUT   D1:  	output data
	
	Copyright (C) 2015 The University of Texas at Austin
	Copyright (C) 2015 Yangkang Chen
	Modified by Yangkang Chen on Jan, 2020
 
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published
	by the Free Software Foundation, either version 3 of the License, or
	any later version.
 
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details: http://www.gnu.org/licenses/
	"""
	import numpy as np

	if D.ndim==2:	#for 2D problems
		D=np.expand_dims(D, axis=2)
		
	if D.ndim==1:	#for 1D problems
		D=np.expand_dims(D, axis=1)
		D=np.expand_dims(D, axis=2)
		
	[n1,n2,n3]=D.shape;
	
	D1=D;
	
	if N==1:
		for i3 in range(0,n3):
			for i2 in range(0,n2):
				D1[:,i2,i3]=D1[:,i2,i3]/np.max(np.abs(D1[:,i2,i3]));
	elif N==2:
		for i3 in range(0,n3):
			D1[:,:,i3]=D1[:,:,i3]/np.max(np.abs(D1[:,:,i3]));
	elif N==3:
		D1=D1/np.max(np.abs(D1));
	elif N==0:
		D1=D1*dscale;
	else:
		print("Invalid argument value N.");

	
	
	D1=np.squeeze(D1);
	
	return D1

def data_normalization(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    normalized_data = (data - mean_val) / std_val
    return normalized_data

def adaptive_normalization(data):
    max_val = np.max(data)
    min_val = np.min(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


import numpy as np
def divne(num, den, Niter, rect, ndat, eps_dv, eps_cg, tol_cg,verb):

	n=num.size
	
	ifhasp0=0
	p=np.zeros(n)
	
	num=num.reshape(n,order='F')
	den=den.reshape(n,order='F')
	
	if eps_dv > 0.0:
		for i in range(0,n):
			norm = 1.0 / np.hypot(den[i], eps_dv);
			num[i] = num[i] * norm;
			den[i] = den[i] * norm;
	norm=np.sum(den*den);
	if norm == 0.0:
		rat=np.zeros(n);
		return rat
	norm = np.sqrt(n / norm);
	num=num*norm;
	den=den*norm;
		
	par_L={'nm':n,'nd':n,'w':den}
	par_S={'nm':n,'nd':n,'nbox':rect,'ndat':ndat,'ndim':3}
	
	
	rat = conjgrad(None, weight_lop, trianglen_lop, p, None, num, eps_cg, tol_cg, Niter,ifhasp0,[],par_L,par_S,verb);
	rat=rat.reshape(ndat[0],ndat[1],ndat[2],order='F')

	return rat


def weight_lop(din,par,adj,add):
	nm=par['nm'];
	nd=par['nd'];
	w=par['w'];

	if adj==1:
		d=din;
		if 'm' in par and add==1:
			m=par['m'];
		else:
			m=np.zeros(par['nm']);
	else:
		m=din;
		if 'd' in par and add==1:
			d=par['d'];
		else:
			d=np.zeros(par['nd']);
	m,d  = adjnull( adj,add,nm,nd,m,d );
	if adj==1:
		m=m+d*w; #dot product
	else: #forward
		d=d+m*w; #d becomes model, m becomes data


	if adj==1:
		dout=m;
	else:
		dout=d;

	return dout
	
def trianglen_lop(din,par,adj,add ):
	if adj==1:
		d=din;
		if 'm' in par and add==1:
			m=par['m'];
		else:
			m=np.zeros(par['nm']);
	else:
		m=din;
		if 'd' in par and add==1:
			d=par['d'];
		else:
			d=np.zeros(par['nd']);


	nm=par['nm'];	 #int
	nd=par['nd'];	 #int
	ndim=par['ndim']; #int
	nbox=par['nbox']; #vector[ndim]
	ndat=par['ndat']; #vector[ndim]

	[ m,d ] = adjnull( adj,add,nm,nd,m,d );

	tr = [];

	s =[1,ndat[0],ndat[0]*ndat[1]];

	for i in range(0,ndim):
		if (nbox[i] > 1):
			nnp = ndat[i] + 2*nbox[i];
			wt = 1.0 / (nbox[i]*nbox[i]);
			tr.append({'nx':ndat[i], 'nb':nbox[i], 'box':0, 'np':nnp, 'wt':wt, 'tmp':np.zeros(nnp)});
		else:
			tr.append('NULL');

	if adj==1:
		tmp=d;
	else:
		tmp=m;

	for i in range(0,ndim):
		if tr[i] != 'NULL':
			for j in range(0,int(nd/ndat[i])):
				i0=first_index(i,j,ndim,ndat,s);
				[tmp,tr[i]]=smooth2(tr[i],i0,s[i],0,tmp);
	
	if adj==1:
		m=m+tmp;
	else:
		d=d+tmp;
		
	if adj==1:
		dout=m;
	else:
		dout=d;

	return dout


def first_index( i, j, dim, n, s ):
	n123 = 1;
	i0 = 0;
	for k in range(0,dim):
		if (k == i):
			continue;
		ii = np.floor(np.mod((j/n123), n[k]));
		n123 = n123 * n[k];
		i0 = i0 + ii * s[k];

	return int(i0)


def smooth2( tr, o, d, der, x):
	tr['tmp'] = triple2(o, d, tr['nx'], tr['nb'], x, tr['tmp'], tr['box'], tr['wt']);
	tr['tmp'] = doubint2(tr['np'], tr['tmp'], (tr['box'] or der));
	x = fold2(o, d, tr['nx'], tr['nb'], tr['np'], x, tr['tmp']);

	return x,tr


def triple2( o, d, nx, nb, x, tmp, box, wt ):
	#BY Yangkang Chen, Nov, 04, 2019

	for i in range(0,nx+2*nb):
		tmp[i] = 0;

	if box:
		tmp[1:]	 = cblas_saxpy(nx,  +wt,x[o:],d,tmp[1:],   1); 	#y += a*x
		tmp[2*nb:]  = cblas_saxpy(nx,  -wt,x[o:],d,tmp[2*nb:],1);
	else:
		tmp		 = cblas_saxpy(nx,  -wt,x[o:],d,tmp,	   1); 	#y += a*x
		tmp[nb:]	= cblas_saxpy(nx,2.*wt,x[o:],d,tmp[nb:],  1);
		tmp[2*nb:]  = cblas_saxpy(nx,  -wt,x[o:],d,tmp[2*nb:],1);

	return tmp

def doubint2( nx, xx, der ):
	#Modified by Yangkang Chen, Nov, 04, 2019
	#integrate forward
	t = 0.0;
	for i in range(0,nx):
		t = t + xx[i];
		xx[i] = t;

	if der:
		return xx

	#integrate backward
	t = 0.0;
	for i in range(nx-1,-1,-1):
		t = t + xx[i];
		xx[i] = t

	return xx



def cblas_saxpy( n, a, x, sx, y, sy ):
	#y += a*x
	#Modified by Yangkang Chen, Nov, 04, 2019

	for i in range(0,n):
		ix = i * sx;
		iy = i * sy;
		y[iy] = y[iy] + a * x[ix];

	return y

def fold2(o, d, nx, nb, np, x, tmp):
	#Modified by Yangkang Chen, Nov, 04, 2019

	#copy middle
	for i in range(0,nx):
		x[o+i*d] = tmp[i+nb];

	#reflections from the right side
	for j in range(nb+nx,np+1,nx):
		if (nx <= np-j):
			for i in range(0,nx):
				x[o+(nx-1-i)*d] = x[o+(nx-1-i)*d] + tmp[j+i];
		else:
			for i in range(0,np-j):
				x[o+(nx-1-i)*d] = x[o+(nx-1-i)*d] + tmp[j+i];
		j = j + nx;
		if (nx <= np-j):
			for i in range(0,nx): 
				x[o+i*d] = x[o+i*d] + tmp[j+i];
		else:
			for i in range(0,np-j):
				x[o+i*d] = x[o+i*d] + tmp[j+i];

	#reflections from the left side
	for j in range(nb,-1,-nx):
		if (nx <= j):
			for i in range(0,nx):
				x[o+i*d] = x[o+i*d] + tmp[j-1-i];
		else:
			for i in range(0,j):
				x[o+i*d] = x[o+i*d] + tmp[j-1-i];
		j = j - nx;
		if (nx <= j):
			for i in range(0,nx):
				x[o+(nx-1-i)*d] = x[o+(nx-1-i)*d] + tmp[j-1-i];
		else:
			for i in range(0,j):
				x[o+(nx-1-i)*d] = x[o+(nx-1-i)*d] + tmp[j-1-i];
	return x

def adjnull( adj,add,nm,nd,m,d ):
	if add:
		return m,d

	if adj:
		m=np.zeros(nm);
		for i in range(0,nm):
			m[i] = 0.0;
	else:
		d=np.zeros(nd);
		for i in range(0,nd):
			d[i] = 0.0;

	return m,d


def conjgrad(opP,opL,opS, p, x, dat, eps_cg, tol_cg, N,ifhasp0,par_P,par_L,par_S,verb):

	nnp=p.size;
	nx=par_L['nm'];	#model size
	nd=par_L['nd'];	#data size

	if opP  is not None:
		d=-dat; #nd*1
		r=opP(d,par_P,0,0);
	else:
		r=-dat;  

	if ifhasp0:
		x=op_S(p,par_S,0,0);
		if opP  is not None:
			d=opL(x,par_L,0,0);
			par_P['d']=r;#initialize data
			r=opP(d,par_P,0,1);
		else:
			par_P['d']=r;#initialize data
			r=opL(x,par_L,0,1);

	else:
		p=np.zeros(nnp);#define np!
		x=np.zeros(nx);#define nx!

	dg=0;
	g0=0;
	gnp=0;
	r0=np.sum(r*r);   #nr*1

	for n in range(1,N+1):
		gp=eps_cg*p; #np*1
		gx=-eps_cg*x; #nx*1
		
		if opP is not None:
			d=opP(r,par_P,1,0);#adjoint
			par_L['m']=gx;#initialize model
			gx=opL(d,par_L,1,1);#adjoint,adding
		else:
			par_L['m']=gx;#initialize model
			gx=opL(r,par_L,1,1);#adjoint,adding

	
		par_S['m']=gp;#initialize model
		gp=opS(gx,par_S,1,1);#adjoint,adding
		gx=opS(gp.copy(),par_S,0,0);#forward,adding
		#The above gp.copy() instead of gp is the most striking bug that has been found because otherwise gp was modified during the shaping operation (opS) (Mar, 28, 2022)
		
		if opP is not None:
			d=opL(gx,par_P,0,0);#forward
			gr=opP(d,par_L,0,0);#forward
		else:
			gr=opL(gx,par_L,0,0);#forward

		gn = np.sum(gp*gp); #np*1

		if n==1:
			g0=gn;
			sp=gp; #np*1
			sx=gx; #nx*1
			sr=gr; #nr*1
		else:
			alpha=gn/gnp;
			dg=gn/g0;
		
			if alpha < tol_cg or dg < tol_cg:
				return x;
				break;
		
			gp=alpha*sp+gp;
			t=sp;sp=gp;gp=t;
		
			gx=alpha*sx+gx;
			t=sx;sx=gx;gx=t;
		
			gr=alpha*sr+gr;
			t=sr;sr=gr;gr=t;

	 
		beta=np.sum(sr*sr)+eps_cg*(np.sum(sp*sp)-np.sum(sx*sx));
		
		if verb:
			print('iteration: %d, res: %g !'%(n,np.sum(r* r) / r0));  

		alpha=-gn/beta;
	
		p=alpha*sp+p;
		x=alpha*sx+x;
		r=alpha*sr+r;
	
		gnp=gn;

	return x


def localsimi(d1,d2,rect,niter=50,eps=0.0,verb=1):

	import numpy as np
	
	if d1.ndim==2:
		d1=np.expand_dims(d1, axis=2)
	if d2.ndim==2:
		d2=np.expand_dims(d2, axis=2)
	[n1,n2,n3]=d1.shape

	nd=n1*n2*n3;
	ndat=[n1,n2,n3];
	eps_dv=eps;
	eps_cg=0.1; 
	tol_cg=0.000001;

	ratio = divne(d2, d1, niter, rect, ndat, eps_dv, eps_cg, tol_cg,verb);
	ratio1 = divne(d1, d2, niter, rect, ndat, eps_dv, eps_cg, tol_cg,verb);
	simi=np.sqrt(np.abs(ratio*ratio1));
	return simi
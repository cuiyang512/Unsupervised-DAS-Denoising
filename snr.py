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
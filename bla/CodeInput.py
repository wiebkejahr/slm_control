# Please edit the code below to calculate the SLM pattern and save into self.data
# There is currently no sanity checking whether self.data is calculated at all
# or whether it's the correct data type and size - this is your job!

# some default parameters & example code:
mode = 'Gauss' 
size = [1200, 792]
rot = 0.0
rad = 1.0
steps = 1.0
phase = 1.0
slm_scale = 0.2698795180722891
r_thick = 0.15 * slm_scale
r_out = 1 * slm_scale
squeeze_stripes = 1
n_stripes = 5
width = 0.02 * slm_scale
rspot = 0.05 * slm_scale
ampspot = 2
r_in = r_out-r_thick
pos = np.linspace(-(r_in*squeeze_stripes-width/2), r_in*squeeze_stripes-width/2, n_stripes) / 2 * np.mean(self.size)
ring = pcalc.crop(pcalc.create_ring(2*self.size, r_in, r_out), self.size)
spot = pcalc.crop(pcalc.create_ring(2*self.size, 0, rspot), self.size)
stripes = pcalc.crop(pcalc.create_gauss(2*self.size), self.size)
for p in pos:
    stripes = stripes + pcalc.crop(pcalc.create_rect(2*self.size, width, 1, slm_scale), self.size, [p,0])
phasemask = pcalc.crop(pcalc.create_gauss(2*self.size), self.size)
amplitude = ring * stripes + spot * ampspot
amplitude = amplitude / np.max(amplitude) * 0.9999
self.data = amplitude
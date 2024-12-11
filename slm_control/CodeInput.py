# Please edit the code below to calculate the SLM pattern and save into self.data
# There is currently no sanity checking whether self.data is calculated at all
# or whether it's the correct data type and size - this is your job!

# some default parameters & example code:
mode = 'Gauss' 
size = [1200, 792]

slm_scale = 0.3
r_out = 1.0 * slm_scale
r_in = 0.9 * slm_scale
r_spot = 0.1 * slm_scale
amp = 0.5
ring = pcalc.crop(pcalc.create_ring(2*self.size, r_in, r_out), self.size)
spot = pcalc.crop(pcalc.create_ring(2*self.size, 0, r_spot), self.size)
phasemask = pcalc.crop(pcalc.create_gauss(2*self.size), self.size)
pattern = ring + spot
self.data = pattern*amp 
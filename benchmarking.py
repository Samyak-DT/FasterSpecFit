import numpy as np
import matplotlib.pylab as plt

# Accuracy of Parameters

cpu_amps, cpu_vshifts, cpu_sigmas = np.loadtxt('params_cpu.txt', unpack = True)
gpu_amps, gpu_vshifts, gpu_sigmas = np.loadtxt('params_gpu.txt', unpack = True)

amp = []
vshift = []
sigma = []

for cpu_amp, cpu_vshift, cpu_sigma , gpu_amp, gpu_vshift, gpu_sigma in zip(cpu_amps, cpu_vshifts, cpu_sigmas, gpu_amps, gpu_vshifts, gpu_sigmas):
    amp_diff = np.positive(gpu_amp - cpu_amp)
    vshift_diff = np.positive(gpu_vshift - cpu_vshift)
    sigma_diff = np.positive(gpu_sigma - cpu_sigma)
    amp_p = np.positive(amp_diff/cpu_amp)
    vshift_p = np.positive(vshift_diff/cpu_vshift)
    sigma_p = np.positive(sigma_diff/cpu_sigma)
    amp.append(amp_p)
    vshift.append(vshift_p)
    sigma.append(sigma_p)

y_val = np.arange(0, len(amp))
fig, plots = plt.subplots(3)
plt.title("Difference in parameter values between CPU and GPU", fontsize = 15)
plots[0].set_title("Percentage Difference in Amplitudes of a Gaussian", fontsize = 15)
plots[0].scatter(amp, y_val)
plots[0].set_xlim(-0.1, 0.1)
plots[1].set_title("Percentage Difference in center of a Gaussian", fontsize = 15)
plots[1].scatter(vshift, y_val)
plots[2].set_title("Percentage Difference in Std Dev of a Gaussian", fontsize = 15)
plots[2].scatter(sigma, y_val)
fig.tight_layout()
plt.savefig("Difference.png")

# Time Management

cpu_time = np.loadtxt('time_cpu.txt', unpack = True)
gpu_time = np.loadtxt('time_gpu.txt', unpack = True)

runtime = ["CPU", "GPU"]
print(cpu_time)
print(gpu_time)
avg_cpu = np.average(cpu_time) 
avg_gpu = np.average(gpu_time)
avg = [avg_cpu, avg_gpu]

print(avg_gpu)
print(avg_cpu)
fig, ax,  = plt.subplots()
ax.set_title("Average Time for 100 spectra for CPU vs GPU (in seconds)", fontsize = 15)
ax.bar(runtime, avg)
plt.xticks(fontsize=16)
fig.tight_layout()
plt.savefig("Time.png")


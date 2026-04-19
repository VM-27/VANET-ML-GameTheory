import matplotlib.pyplot as plt

methods = ["Static ML", "Dynamic GT"]

PDR = [79.37, 77.88]
Delay = [0.0069, 0.0163]
Throughput = [0.006, 0.002]

# PDR
plt.figure()
plt.bar(methods, PDR)
plt.title("PDR Comparison")
plt.xlabel("Method")
plt.ylabel("PDR (%)")
plt.grid()
plt.savefig("pdr_comparison.png")
plt.show()

# Delay
plt.figure()
plt.bar(methods, Delay)
plt.title("Delay Comparison")
plt.xlabel("Method")
plt.ylabel("Delay (s)")
plt.grid()
plt.savefig("delay_comparison.png")
plt.show()

# Throughput
plt.figure()
plt.bar(methods, Throughput)
plt.title("Throughput Comparison")
plt.xlabel("Method")
plt.ylabel("Throughput (Mbps)")
plt.grid()
plt.savefig("throughput_comparison.png")
plt.show()

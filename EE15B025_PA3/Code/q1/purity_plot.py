import matplotlib.pyplot as plt 
import numpy as np

y = [6.6667, 13.3333, 20, 26.6667, 33.3333, 40, 46.6667, 53.3333, 60, 66.6667, 73.3333, 80, 
		86.1667, 84, 90.6667, 87.8333, 94.6667, 91.5, 88.1667, 85.1667]

x = np.arange(len(y)) 

plt.plot(x,y)
plt.title("Purity vs k")
plt.xlabel("k")
plt.ylabel("Purity (%)")
plt.show()
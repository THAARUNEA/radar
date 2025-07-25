import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
c=3e8
fs=1e6 
target_distances=[100,150,200]
pulse_width=1e-6 

#generate pulse
t=np.arange(0,pulse_width,1/fs)
pulse=signal.square(2*np.pi*1e6*t)

#echo signals
echo_signals=[]
for d in target_distances:
    delay=int((2*d/c)*fs)
    echo=np.pad(pulse,(delay,0),'constant',constant_values=(0,))
    echo=echo[:len(pulse)+max(delay for d in target_distances)]
    echo_signals.append(echo)
    for i, echo in enumerate(echo_signals):
        plt.plot(echo,label=f"{target_distances[i]}m Target")
    plt.title("Radar echo")
    plt.legend()
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

import random
    def create_sample(distance,noise_level,label):
        fs=1e6
        pulse_width=1e-6
        fixed_length=500
        t=np.arange(0,pulse_width,1/fs)
        pulse=signal.square(2*np.pi*1e6*t)
        delay=int((2*distance/3e8)*fs)
        echo=np.pad(pulse,(delay,0),'constant')[:500]
        echo=echo[:fixed_length]
        if len(echo)<fixed_length:
            echo=np.pad(echo,(0,fixed_length-len(echo)))
        echo+=np.random.normal(0,noise_level,len(echo)) 
        return echo,label
    X,y=[],[]
    for _ in range(500): 
        d=random.choice([100,150])
        label=0 if d==100 else 1  
        sample, label=create_sample(d,0.1,label)
        X.append(sample)
        y.append(label)
    np.save('x_radar.npy',np.array(X))
    np.save('y_radar.npy',np.array(y))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X=np.load('x_radar.npy')
y=np.load('y_radar.npy')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
model=RandomForestClassifier()
model.fit(X_train,y_train)
preds=model.predict(X_test)
acc=accuracy_score(y_test,preds)
print(f"Target classification accuracy:{acc*100:.2f}%")


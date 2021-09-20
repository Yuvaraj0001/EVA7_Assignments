# Session 0 Assignment

## 1. What are Channels and Kernels (according to EVA)?
### Channels:
According to EVA, channels are defined using below examples.
1. From the below figure, metallic song or music is produced by combining individual instruments played (drums, guitar, piano, vocals, etc.). In terms of channel, each instrument is a channel such as drum channel, guitar channel, piano channel, vocal channel, etc.
![alt text]()
2. Below text image having 26 channels based on 26 alphabets/characters. Here, each character is considered as a channel. For example, “m-channel” defines all ‘m’ characters with different size and orientation are combined as a channel.

### Kernels:
From the below image, Kernels is a set of 9 values (3x3 matrix – dark violet color) moving on top of channel (4x4 matrix-light violet color) horizontal and vertical directions. Kernels are used to extract edges or features from corresponding channel. Also, kernels are known as feature extractors or filters.


## 2. Why should we (nearly) always use 3x3 kernels?

From the below image, 5x5 channel is convolved by 3x3 kernels and got 3x3 output channel. Again, if we convolve 3x3 channel by 3x3 kernels, we will get 1x1 output. By using 3x3 kernels, we got 9+9=18 parameters to reach 1x1.

Similarly, if we use 5x5 kernels over 5x5 channel, it will require 25 parameters to reach 1x1. 

Therefore, if we using 3x3 kernels the total number of parameters are less when compare to 5x5 kernels. Also, 3x3 Kernels learns large complex features easily, whereas large filters learns simple features.


## 3.	How many times do we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...).

We need to perform 99 times of 3x3 convolutions operations to reach close to 1x1 from 199x199.

199x199 > 197x197 > 195x195 > 193x193 > 191x191 >
189x189 > 187x187 > 185x185 > 183x183 > 181x181>
179x179 > 177x177 > 175x175 > 173x173 > 171x171>
169x169 > 167x167 > 165x165 > 163x163 > 161x161>
159x159 > 157x157 > 155x155 > 153x153 > 151x151>
149x149 > 147x147 > 145x145 > 143x143 > 141x141>
139x139 > 137x137 > 135x135 > 133x133 > 131x131>
129x129 > 127x127 > 125x125 > 123x123 > 121x121>
119x119 > 117x117 > 115x115 > 113x113 > 111x111>
109x109 > 107x107 > 105x105 > 103x103 > 101x101>
99x99 > 97x97 > 95x95 > 93x93 > 91x91 >
89x89 > 87x87 > 85x85 > 83x83 > 81x81>
79x79 > 77x77 > 75x75 > 73x73 > 71x71>
69x69 > 67x67 > 65x65 > 63x63 > 61x61>
59x59 > 57x57 > 55x55 > 53x53 > 51x51>
49x49 > 47x47 > 45x45 > 43x43 > 41x41>
39x39 > 37x37 > 35x35 > 33x33 > 31x31>
29x29 > 27x27 > 25x25 > 23x23 > 21x21>
19x19 > 17x17 > 15x15 > 13x13 > 11x11>
9x9 > 7x7 > 5x5 > 3x3 > 1x1


## 4.	How are kernels initialized? 

The kernels are initialized by small random/ arbitrary values. Then the values are optimized using gradient decent optimizer, so that the kernels solve the problem. There are two general initialization approaches are,

#### He initialization: 
It works better for layers with ReLu (Rectified Linear unit) activation.

#### Xavier initialization:  
It works better for layers with sigmoid activation.

Below conditions are used to prevent the gradients of the activations from vanishing (too small initialization) or exploding (too large initialization), 
1.	The mean of activations should be zero.
2.	The variance of activations should remains same across every layer.


## 5.	What happens during the training of a DNN?
Neural network trying to extract feature such as edges, texture, pattern, parts and objects from input datasets internally. 
Neural network process images and try to learn the pattern so that we can make a better prediction. Let’s take example of our brain, how it will process the images? It uses an edge detector to detect the edges and forms an image based on that. So when an Images comes to the brain it uses edge detector to find the edges and then those edges are converted to texture, pattern, part of an object and finally they are converted to object.



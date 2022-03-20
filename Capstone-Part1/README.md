# EVA7 Capstone - Part 1

Submitted by: Yuvaraj V (yuvaraj100493@gmail.com)

### 1. We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention (FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)

First, we have to pass RGB input image to Resnet50 and getting last layer output size is 2048 x h/32 x w/32. Then this is passed to DETR Encoder and getting the encoded image of size dxH/32xW/32 where d is reduced to 256 from 2048.
This DETR encoded image send it to Multi-Head Attention.

### 2. We also send dxN Box embeddings to the Multi-Head Attention

We also sending dxN Box embeddings with encoded image d x H/32 x W/32 to the multi-head attention.

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Capstone-Part1/Image/arch-1.png)

### 3. We do something here to generate NxMxH/32xW/32 maps

Bounding boxes from DETR decoder and encoded image from DETR encoder are passed to multi-head attention. The output size from this head attention maps is N x M x H/32 x W/32.

### 4. Then we concatenate these maps with Res5 Block

Res5 block is coming from the input projection of the feature map that is obtained from encoder of the architecture.

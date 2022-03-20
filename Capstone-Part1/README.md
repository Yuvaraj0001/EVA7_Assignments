# EVA7 Capstone - Part 1

Submitted by: Yuvaraj V (yuvaraj100493@gmail.com)

### 1. We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention (FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)

First, we have to pass RGB input image to Resnet50 and getting last layer output size is 2048 x h/32 x w/32. Then this is passed to DETR Encoder and getting the encoded image of size dxH/32xW/32 where d is reduced to 256 from 2048.
This DETR encoded image send it to Multi-Head Attention.

### 2. We also send dxN Box embeddings to the Multi-Head Attention

We also sending dxN Box embeddings with encoded image d x H/32 x W/32 to the multi-head attention.

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Capstone-Part1/Image/arch-1.png)

### 3. We do something here to generate NxMxH/32xW/32 maps (WHAT DO WE DO HERE?)

Bounding boxes from DETR decoder and encoded image from DETR encoder are passed to multi-head attention. The output size from this head attention maps is N x M x H/32 x W/32.

### 4. Then we concatenate these maps with Res5 Block (WHERE IS THIS COMING FROM?)

Res5 block is coming from the input projection of the feature map that is obtained from encoder of the architecture.

### 5. Then we perform the above steps (EXPLAIN THESE STEPS)

We train DETR to predict boxes around both objects and its classes from construction dataset using the object detection model.  The mask head takes as input, the output of transformer decoder for each object and computes multi-head (with M heads) attention scores of this embedding over the output of the encoder, generating M attention heatmaps per object in a small resolution. To make the final prediction and increase the resolution, an FPN-like architecture is used. The final resolution of the masks has stride 4 and each mask is supervised separately using the DICE/F-1 loss and Focal loss.

The mask head can be trained either jointly, or in a two steps process, where we train DETR for boxes only, then freeze all the weights and train only the mask head for 25 epochs. To predict the final panoptic segmentation we simply use an argmax over the mask scores at each pixel, and assign the corresponding categories to the resulting masks. This procedure guarantees that the final masks have no overlaps and, therefore, DETR does not require a heuristic that is often used to align different masks.

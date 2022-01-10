# Session 10: ViT - An Image is worth 16x16 words

## PatchEmbeddings(nn.Module)

Conversion of an image size to the patch embeddings

1. Takes an input image, check the dimensions of the image and calculates the number of patches required.
2. Captures the batch size, no.of channels and image dimensions from the pixel_vlues.shape
3. Projects the image to a required size, followed by flatten and transpose.

## ViTEmbeddings(nn.Module)

Construct the CLS token, position and patch embeddings.

1. ViTEmbeddings expects a predefined configuration file with different attributes of image and batch
2. Construction of CLS tokens, with random numbers of size mentioned in the config.hidden_size
3. Construction of Patchembeddings with image_size, patch_size, num_channels, embed_dim as configured in the config.
4. Construction of position embeddings for num_patches+1
5. Concatanation of cls_tokens, patch_embeddings and add the postion_embeddings to all the patch_embeddings.

## ViTSelfAttention(nn.Module)

Costruction of self Attention matrix.

1. Check if the hidden_size is divisible by the num_attention_heads.
2. Initialising the num_attention_heads, attention_head_size, all_head_size as configured in the config.
3. Creating the query, key and Value matrices of size hidden_size, all_head_size
4. Reshaping the matrices to required sizes by Permute operation and make the query, key and Value layers.
5. Multiplication of query and key matrices to get the attention scores and Normalize the attention scores.
6. Making the context_layer which is a Product of attention_probs and value_layer.

## ViTSelfOutput(nn.Module)

This class specifies the Linear layer block.

Making the dense layers, takes hidden_size and gives layer of hidden_size.

## ViTAttention(nn.Module)

Class Specifying the ViT Attention layer.

1. get the ViTSelfAttention object, ViTSelfOutput and pruned_heads(Required to remove the unwanted heads).
2. get the self_outputs from the FC layer.
3. Get the attention_output layer
4. Add the attentions if we return the outputs

## ViTLayer(nn.Module)

This corresponds to the Block class in the timm implementation

1. Get the ViTAttention, ViTIntermediate and ViTOutput objects, layernorm_before and layer_norm after.
2. Get the self_attention_outputs which intakes the layernorm_before, head_mask and output_attentions.
3. Add self attentions and the skip connections.
4. Layernorm is also applied after self-attention

## ViTEncoder(nn.Module)

This Class specifies the overall VIT layer

1. Get all the hidden states
2. Layer outputs are obtained through the layer module using hidden_states, layer_head_masak, ouput_attention

## ViTModel()

This class specifies the actual model for ViT training

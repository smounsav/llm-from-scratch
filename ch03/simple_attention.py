import torch
inputs = torch.tensor( # Input embeddings
    [[0.43, 0.15, 0.89], # Your    (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts  (x^3)
     [0.22, 0.58, 0.33], # with    (x^4)
     [0.77, 0.25, 0.10], # one     (x^5)
     [0.05, 0.80, 0.55]]  # step    (x^6)
)

# # 3.3.1
# # Computation of the attention score between the second token (journey) and the other input tokens
# query = inputs[1]
# attn_scores_2 = torch.empty(inputs.shape[0])
# for i, x_i in enumerate(inputs):
#     attn_scores_2[i] = torch.dot(x_i, query)
# print(attn_scores_2)

# # Normalization
# attn_weight_2_tmp = attn_scores_2 / attn_scores_2.sum()
# print("Attention weights:", attn_weight_2_tmp)
# print("Sum:", attn_weight_2_tmp.sum())

# def softmax_naive(x):
#     return torch.exp(x) / torch.exp(x).sum(dim=0)

# attn_weight_2_naive = softmax_naive(attn_scores_2)
# print("Attention weights:", attn_weight_2_naive)
# print("Sum:", attn_weight_2_naive.sum())

# attn_weight_2 = torch.softmax(attn_scores_2, dim=0)
# print("Attention weights:", attn_weight_2)
# print("Sum:", attn_weight_2.sum())

# query = inputs[1]
# context_vec_2 = torch.zeros(query.shape)
# for i, x_i in enumerate(inputs):
#     context_vec_2 += attn_weight_2[i] * x_i
# print(context_vec_2)

# 3.3.2
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs): # basic for loop
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

# with matrix multiplication
attn_scores = inputs @ inputs.T
print(attn_scores)

#Normalization
attn_weights = torch.softmax(attn_scores, dim = -1)
print(attn_weights)

print("All row sums:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
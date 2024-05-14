import torch
import torch.nn as nn
import torch.nn.functional as F


class new_BiDirectionalAttention(nn.Module):
    def __init__(self, channels, dim):
        super(new_BiDirectionalAttention, self).__init__()
        self.channels = channels
        self.dim = dim
        depth, height, width = dim

        self.image_conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.image_conv2 = nn.Conv3d(channels, channels, kernel_size=1)

        self.prompt_conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.prompt_conv2 = nn.Conv3d(channels, channels, kernel_size=1)

        self.image_norm = nn.LayerNorm([channels, *dim])
        self.prompt_norm = nn.LayerNorm([channels, *dim])

        self.image_addnorm1 = AddNorm(channels, dim)
        self.image_addnorm2 = AddNorm(channels, dim)
        self.prompt_addnorm1 = AddNorm(channels, dim)
        self.prompt_addnorm2 = AddNorm(channels, dim)
        self.prompt_addnorm3 = AddNorm(channels, dim)

        self.image_gelu = nn.GELU()
        self.prompt_gelu = nn.GELU()

        self.image_multiheadattn = MultiHeadSelfAttention(
            channels, 4, depth, height, width
        )
        self.promtp_multiheadattn = MultiHeadSelfAttention(
            channels, 4, depth, height, width
        )

        # MLP layer
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, channels * 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(channels * 4, channels, kernel_size=1),
        )

    def forward(self, image_embedding, prompt_embedding):
        batch_size = image_embedding.size(0)

        updated_image_embedding = self.image_conv1(image_embedding)
        updated_image_embedding = self.image_norm(updated_image_embedding)
        updated_image_embedding = self.image_gelu(updated_image_embedding)
        updated_image_embedding = self.image_conv2(updated_image_embedding)
        updated_image_embedding = self.image_addnorm1(
            updated_image_embedding, image_embedding
        )

        updated_prompt_embedding = self.prompt_conv1(prompt_embedding)
        updated_prompt_embedding = self.prompt_norm(updated_prompt_embedding)
        updated_prompt_embedding = self.prompt_gelu(updated_prompt_embedding)
        updated_prompt_embedding = self.prompt_conv2(updated_prompt_embedding)
        updated_prompt_embedding = self.prompt_addnorm1(
            updated_prompt_embedding, prompt_embedding
        )

        q_p = prompt_embedding + updated_prompt_embedding
        k_p = image_embedding
        v_p = image_embedding

        output_prompt_embedding = self.promtp_multiheadattn(q_p, k_p, v_p)
        output_prompt_embedding = self.prompt_addnorm2(
            output_prompt_embedding, updated_prompt_embedding
        )
        output_prompt_embedding = self.prompt_addnorm3(
            self.mlp(output_prompt_embedding), output_prompt_embedding
        )

        q_i = updated_image_embedding + image_embedding
        k_i = output_prompt_embedding + prompt_embedding
        v_i = prompt_embedding

        output_image_embedding = self.image_multiheadattn(q_i, k_i, v_i)
        output_image_embedding = self.image_addnorm2(
            output_image_embedding, updated_image_embedding
        )

        return output_image_embedding, output_prompt_embedding


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels, num_heads, depth, height, width):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.depth = channels // num_heads
        self.total_features = depth * height * width

        # 确保通道数可以被头数整除
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.wq = nn.Linear(self.total_features, channels)
        self.wk = nn.Linear(self.total_features, channels)
        self.wv = nn.Linear(self.total_features, channels)

        self.fc = nn.Linear(channels, self.total_features)

    def split_heads(self, x, batch_size):
        # 分割最后一个维度到 (num_heads, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # 重排为 (batch_size, num_heads, seq_len, depth)

    def forward(self, q, k, v):
        batch_size, _, depth, height, width = q.size()

        # 展平 spatial dimensions，形状变为 (batch_size, channels, -1)
        q_flat = q.view(batch_size, self.channels, -1)  # [1, 384, 512]
        k_flat = k.view(batch_size, self.channels, -1)
        v_flat = v.view(batch_size, self.channels, -1)

        # tmp = self.wq(q_flat)
        # print(tmp.shape)

        q = self.split_heads(
            self.wq(q_flat), batch_size
        )  # self.wq(q_flat) [1, 384, 384] q [1, 4, 384, 96]
        k = self.split_heads(self.wk(k_flat), batch_size)
        v = self.split_heads(self.wv(v_flat), batch_size)

        # 缩放点积注意力
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(self.depth, dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        output = torch.matmul(attention_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.channels)

        # 通过最后一个全连接层
        output = self.fc(output)
        output = output.view(batch_size, self.channels, depth, height, width)
        return output


class AddNorm(nn.Module):
    def __init__(self, channels, dim):
        super(AddNorm, self).__init__()
        self.channels = channels  # 定义 channels 属性
        self.norm = nn.LayerNorm([channels, *dim])
        self.dim = dim

    def forward(self, embedding1, embedding2, embedding3):
        output = embedding1 + embedding2 + embedding3
        output = self.norm(output)
        return output


class BiDirectionalAttentionNetwork(nn.Module):
    def __init__(self, channels, dim, num_attention_blocks):
        super(BiDirectionalAttentionNetwork, self).__init__()
        self.channels = channels
        self.dim = dim
        depth, height, width = dim

        # N * BiDirectionalAttention
        self.bidirectional_attentions = nn.ModuleList(
            [
                new_BiDirectionalAttention(channels, dim)
                for _ in range(num_attention_blocks)
            ]
        )

        # Muilthead Attention Layer
        self.multi_head_self_attention = MultiHeadSelfAttention(
            channels, num_heads=4, depth=depth, height=height, width=width
        )

        # add and Norm
        self.add_norm = AddNorm(channels, dim)

    def forward(self, image_embedding, prompt_embedding):
        updated_image_embedding = image_embedding
        updated_prompt_embedding = prompt_embedding

        # N * BidirectionalAttentions
        for attention_block in self.bidirectional_attentions:
            updated_image_embedding, updated_prompt_embedding = attention_block(
                updated_image_embedding, updated_prompt_embedding
            )

        q = updated_prompt_embedding + prompt_embedding
        k = updated_image_embedding + image_embedding
        v = image_embedding

        # Muilthead Attention Layer
        output = self.multi_head_self_attention(q, k, v)

        # Add and Norm
        output = self.add_norm(output, prompt_embedding, image_embedding)

        # Softmax layer
        output = nn.functional.softmax(output, dim=-1)

        return output


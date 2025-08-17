def main_repair(adata,df,device,save_path='example.csv'):
    import pandas as pd
    import numpy as np
    import math
    from inspect import isfunction
    from functools import partial
    from typing import Tuple
    from einops import rearrange, reduce
    from einops.layers.torch import Rearrange

    import torch
    from torch import nn, einsum
    import torch.nn.functional as F
    # torch.set_num_threads(256)

    def rotate_every_two(x):
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x = torch.stack([-x2, x1], dim=-1)
        return x.flatten(-2)

    def theta_shift(x, sin, cos):
        return (x * cos) + (rotate_every_two(x) * sin)

    def exisit(x):
        return x is not None


    def default(val, d):
        if exisit(val):
            return val
        return d() if isfunction(d) else d


    class Residual(nn.Module):
        def __init__(self, fn):
            super(Residual, self).__init__()
            self.fn = fn

        def forward(self, x, *args, **kwargs):
            return self.fn(x, *args, **kwargs) + x


    def Upsample(dim, dim_out=None):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dim, default(dim_out, dim), 3, 1, 1)
        )


    def Downsample(dim, dim_out=None):
        return nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(dim * 4, default(dim_out, dim), 1, 1, 0)
        )


    class SinusoidalPositionEmbeddings(nn.Module):
        def __init__(self, dim):
            super(SinusoidalPositionEmbeddings, self).__init__()
            self.dim = dim

        def forward(self, time):
            device = time.device
            half_dim = self.dim // 2
            embeddings = math.log(10000) / (half_dim - 1)
            embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
            embeddings = time[:, None] * embeddings[None, :]
            embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
            return embeddings


    class WeightStandardizedConv2d(nn.Conv2d):
        def forward(self, x):
            eps = 1e-5 if x.dtype == torch.float32 else 1e-3

            weight = self.weight
            mean = reduce(weight, "o ... -> o 1 1 1", "mean")
            var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
            normalized_weight = (weight - mean) * (var + eps).rsqrt()

            return F.conv2d(
                x,
                normalized_weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )


    class Block(nn.Module):
        def __init__(self, dim, dim_out, groups=8):
            super(Block, self).__init__()
            self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
            self.norm = nn.GroupNorm(groups, dim_out)
            self.act = nn.SiLU()

        def forward(self, x, scale_shift=None):
            x = self.proj(x)
            x = self.norm(x)

            if exisit(scale_shift):
                scale, shift = scale_shift
                x = x * (scale + 1) + shift

            x = self.act(x)

            return x


    class ResnetBlock(nn.Module):
        def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
            super(ResnetBlock, self).__init__()
            self.mlp = (
                nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exisit(time_emb_dim) else None
            )
            self.block1 = Block(dim, dim_out, groups=groups)
            self.block2 = Block(dim_out, dim_out, groups=groups)
            self.res_conv = nn.Conv2d(dim, dim_out, 1, 1, 0) if dim != dim_out else nn.Identity()

        def forward(self, x, time_emb=None):
            scale_shift = None
            if exisit(self.mlp) and exisit(time_emb):
                time_emb = self.mlp(time_emb)
                time_emb = rearrange(time_emb, "b c -> b c 1 1")
                scale_shift = time_emb.chunk(2, dim=1)

            h = self.block1(x, scale_shift=scale_shift)
            h = self.block2(h)
            return h + self.res_conv(x)


    class Attention(nn.Module):
        def __init__(self, dim, heads=8, dim_head=32):
            super(Attention, self).__init__()
            self.scale = dim_head ** -0.5
            self.heads = heads
            hidden_dim = dim_head * heads
            self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, 1, 0, bias=False)
            self.to_out = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        def forward(self, x):
            b, c, h, w = x.shape
            qkv = self.to_qkv(x).chunk(3, dim=1)
            q, k, v = map(
                lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
            )
            q = q * self.scale

            sim = einsum("b h d i, b h d j -> b h i j", q, k)
            sim = sim - sim.amax(dim=-1, keepdim=True).detach()
            attn = sim.softmax(dim=-1)

            out = einsum("b h i j, b h d j -> b h i d", attn, v)
            out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
            return self.to_out(out)



    class GateLinearAttentionNoSilu(nn.Module):

        def __init__(self, dim, num_heads):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** (-0.5)
            self.qkvo = nn.Conv2d(dim, dim * 4, 1)
            self.elu = nn.ELU()
            self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
            self.proj = nn.Conv2d(dim, dim, 1)

        def forward(self, x, sin, cos):
            '''
            x: (b c h w)
            sin: ((h w) d1)
            cos: ((h w) d1)
            '''
            B, C, H, W = x.shape
            qkvo = self.qkvo(x)  # (b 3*c h w)
            qkv = qkvo[:, :3 * self.dim, :, :]
            o = qkvo[:, 3 * self.dim:, :, :]
            lepe = self.lepe(qkv[:, 2 * self.dim:, :, :])  # (b c h w)

            q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads)  # (b n (h w) d)

            q = self.elu(q) + 1.0
            k = self.elu(k) + 1.0  # (b n l d)

            q_mean = q.mean(dim=-2, keepdim=True)  # (b n 1 d)
            eff = self.scale * q_mean @ k.transpose(-1, -2)  # (b n 1 l)
            eff = torch.softmax(eff, dim=-1).transpose(-1, -2)  # (b n l 1)
            k = k * eff * (H * W)

            q_rope = theta_shift(q, sin, cos)
            k_rope = theta_shift(k, sin, cos)

            z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)  # (b n l 1)
            kv = (k_rope.transpose(-2, -1) * ((H * W) ** -0.5)) @ (v * ((H * W) ** -0.5))  # (b n d d)

            res = q_rope @ kv * z  # (b n l d)
            res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
            res = res + lepe
            return self.proj(res * o)

    class RoPE(nn.Module):

        def __init__(self, embed_dim, num_heads):
            '''
            recurrent_chunk_size: (clh clw)
            num_chunks: (nch ncw)
            clh * clw == cl
            nch * ncw == nc

            default: clh==clw, clh != clw is not implemented
            '''
            super().__init__()
            angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
            angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
            self.register_buffer('angle', angle)

        def forward(self, slen: Tuple[int]):
            '''
            slen: (h, w)
            h * w == l
            recurrent is not implemented
            '''
            # index = torch.arange(slen[0]*slen[1]).to(self.angle)
            index_h = torch.arange(slen[0]).to(self.angle)
            index_w = torch.arange(slen[1]).to(self.angle)
            sin_h = torch.sin(index_h[:, None] * self.angle[None, :])  # (h d1//2)
            sin_w = torch.sin(index_w[:, None] * self.angle[None, :])  # (w d1//2)
            sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h w d1//2)
            sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h w d1//2)
            sin = torch.cat([sin_h, sin_w], -1)  # (h w d1)
            cos_h = torch.cos(index_h[:, None] * self.angle[None, :])  # (h d1//2)
            cos_w = torch.cos(index_w[:, None] * self.angle[None, :])  # (w d1//2)
            cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h w d1//2)
            cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h w d1//2)
            cos = torch.cat([cos_h, cos_w], -1)  # (h w d1)

            retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))

            return retention_rel_pos

    class PreNorm(nn.Module):
        def __init__(self, dim, fn):
            super(PreNorm, self).__init__()
            self.fn = fn
            self.norm = nn.GroupNorm(1, dim)

        def forward(self, x):
            x = self.norm(x)
            return self.fn(x)


    class Unet_Gate(nn.Module):
        def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, self_condition=False,
                    resnet_block_groups=4, num_heads=8):
            super(Unet_Gate, self).__init__()
            self.num_heads = num_heads
            self.channels = channels
            self.self_condition = self_condition
            input_channels = channels * (2 if self_condition else 1)

            init_dim = default(init_dim, dim)
            self.init_conv = nn.Conv2d(input_channels, init_dim, 1, 1, 0)

            dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
            in_out = list(zip(dims[:-1], dims[1:]))
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )

            self.downs = nn.ModuleList()
            self.ups = nn.ModuleList()
            self.attn_modules = nn.ModuleList()  
            self.rope_modules = nn.ModuleList()  

        
            for ind, (dim_in, dim_out) in enumerate(in_out):
                is_last = ind >= (len(in_out) - 1)
                attn = GateLinearAttentionNoSilu(dim=dim_in, num_heads=self.num_heads)
                rope = RoPE(embed_dim=dim_in, num_heads=self.num_heads)
                self.attn_modules.append(attn)  
                self.rope_modules.append(rope)  

                self.downs.append(
                    nn.ModuleList([
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, attn)),  
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                    ])
                )

            mid_dim = dims[-1]
            self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
            self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))  
            self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        
            for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
                is_last = ind == (len(in_out) - 1)
                attn = GateLinearAttentionNoSilu(dim=dim_out, num_heads=self.num_heads)
                rope = RoPE(embed_dim=dim_out, num_heads=self.num_heads)
                self.attn_modules.append(attn)  # 添加到注意力列表
                self.rope_modules.append(rope)  # 添加到RoPE列表

                self.ups.append(
                    nn.ModuleList([
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, attn)),  # 使用GateLinearAttentionNoSilu
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, 1, 1)
                    ])
                )

            self.out_dim = default(out_dim, channels)
            self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
            self.final_conv = nn.Conv2d(dim, self.out_dim, 1, 1, 0)

        def forward(self, x, time, x_self_cond=None):
            if self.self_condition:
                x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
                x = torch.cat((x_self_cond, x), dim=1)

            x = self.init_conv(x)
            r = x.clone()
            t = self.time_mlp(time)
            h = []

 
            attn_idx = 0  
            for ind, (block1, block2, attn_layer, downsample) in enumerate(self.downs):
                x = block1(x, t)
                h.append(x)
                x = block2(x, t)

              
                attn = self.attn_modules[attn_idx]
                rope = self.rope_modules[attn_idx]
                attn_idx += 1

                h_, w_ = x.shape[2], x.shape[3]
                sin, cos = rope((h_, w_))
                
                # 手动处理PreNorm归一化（与attn_layer中的逻辑一致）
                x_norm = attn_layer.fn.norm(x)  # PreNorm中的归一化层
                x_attn = attn(x_norm, sin, cos)  # 调用注意力模块，传入位置编码
                x = x + x_attn  # 残差连接

                h.append(x)
                x = downsample(x)

            # 中间层（保留原始注意力，无需位置编码）
            x = self.mid_block1(x, t)
            x = self.mid_attn(x)  # 原始注意力，无需位置编码
            x = self.mid_block2(x, t)

            # 解码器正向传播（使用GateLinearAttentionNoSilu）
            for ind, (block1, block2, attn_layer, upsample) in enumerate(self.ups):
                x = torch.cat((x, h.pop()), dim=1)
                x = block1(x, t)
                x = torch.cat((x, h.pop()), dim=1)
                x = block2(x, t)

                # 获取当前层级的注意力和RoPE模块
                attn = self.attn_modules[attn_idx]
                rope = self.rope_modules[attn_idx]
                attn_idx += 1

                h_, w_ = x.shape[2], x.shape[3]
                sin, cos = rope((h_, w_))
                
                x_norm = attn_layer.fn.norm(x)  # 归一化
                x_attn = attn(x_norm, sin, cos)
                x = x + x_attn  # 残差连接

                x = upsample(x)

            x = torch.cat((x, r), dim=1)
            x = self.final_res_block(x, t)
            return self.final_conv(x)

    import os
    import torch
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm
    import numpy as np
    from torchvision import transforms, datasets
    # DDPM模型

    # 定义4种生成β的方法，均需传入总步长T，返回β序列
    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)


    def quadratic_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


    def sigmoid_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


    def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


  
    def q_sample(x_start, t, noise=None):
        std_dev = 0.0001
        if noise is None:
            noise = torch.randn_like(x_start) * std_dev
        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumpord_t = extract(sqrt_one_minus_alphas_cumpord, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumpord_t * noise


 
    def p_losses(denoise_model, x_start, t, noise=None, loss_type="l2"):
        
        std_dev = 0.0001
        if noise is None:
            noise = torch.randn_like(x_start) * std_dev

        x_noisy = q_sample(x_start, t, noise)
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss


    @torch.no_grad()
    def p_sample(model, x, t, t_index):
        betas_t = extract(betas, t, x.shape)
        sqrt_one_minus_alphas_cumpord_t = extract(sqrt_one_minus_alphas_cumpord, t, x.shape)
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumpord_t)
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            std_dev = 0.0001  
            noise = torch.randn_like(x) * std_dev  
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(model, shape):
        device = next(model.parameters()).device
        b = shape[0]
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(0, timesteps)), desc="sampling loop time step", total=timesteps):
            img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu())
        return imgs

    @torch.no_grad()
    def sample(model, image_size, batch_size=16, channels=1):
        return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
    data_matrix = df.values
    scaled_matrix = data_matrix 

    n_samples = scaled_matrix.shape[0] 
    image_size = int(np.sqrt(scaled_matrix.shape[1])) 
    reshaped_matrix = scaled_matrix.reshape(n_samples, image_size, image_size) 
    image_tensors = torch.tensor(reshaped_matrix, dtype=torch.float32).unsqueeze(1) 
    timesteps = 2  
 
    betas = linear_beta_schedule(timesteps=timesteps)
    alphas = 1. - betas 
    alphas_cumprod = torch.cumprod(alphas, axis=0) 
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  
    sqrt_recip_alphas = torch.sqrt(1. / alphas)  
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  
    sqrt_one_minus_alphas_cumpord = torch.sqrt(1. - alphas_cumprod)  
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) 
    total_epochs = 50
    channels = 1
    batch_size = 256
    lr = 1e-4


    result_dict = {}

    for i in tqdm(range(len(image_tensors)), desc='Processing'):
        spot_dict = {}
        image = image_tensors[i].to(device)
        image = image.unsqueeze(0)  

    
        model = Unet_Gate(dim=image_size, channels=channels, dim_mults=(1, 2, 4), num_heads=8).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(total_epochs):
                optimizer.zero_grad()

             
                t = torch.randint(0, timesteps, (1,), device=device).long()

             
                noise = torch.randn_like(image)
                x_noisy = q_sample(image, t, noise)

          
                predicted_noise = model(x_noisy, t)
                loss = F.mse_loss(noise, predicted_noise)
     
                loss.backward()
                optimizer.step()

                spot_dict[epoch] = (x_noisy - predicted_noise).squeeze().detach().cpu().numpy()

        
        result_dict[i] = spot_dict

    def mse_similarity(A, B):

      
        diff = A - B
        sq_diff = np.square(diff)

       
        return np.mean(sq_diff)

    NumDiff=len(result_dict[0])

    average_result = {}

 
    for ij in range(len(result_dict)):
        
        original_matrix = image_tensors[ij].squeeze().cpu().numpy()
       
        similarities = {}
        for i in range(NumDiff):
            similarities[i] = mse_similarity(original_matrix, result_dict[ij][i])

        most_similar_matrices = []
        for i in range(2):
            min_similarity = float("inf")
            min_idx = None
            for j, similarity in similarities.items():
                if similarity < min_similarity:
                    min_similarity = similarity
                    min_idx = j

        most_similar_matrices.append(min_idx)
    
        average_matrix = np.zeros((64, 64))
        for i in most_similar_matrices:
            average_matrix += result_dict[ij][i]

        average_matrix /= 2
    
        average_result[ij] = average_matrix
      
        print(average_matrix)
        
        print(ij, ':', average_result[ij])

    revector_result = {}

    for ij in range(len(result_dict)):
        revector_result[ij] = average_result[ij].flatten()
  
    repaired_images_matrix = np.vstack([revector_result[i].flatten() for i in range(len(revector_result))])
    np.savetxt(save_path, repaired_images_matrix, delimiter=',')

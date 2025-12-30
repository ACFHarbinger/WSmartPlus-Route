import os
import torch
import argparse
import numpy as np
import seaborn as sns
import loss_landscapes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from torch.utils.tensorboard.writer import SummaryWriter
from logic.src.problems.wcvrp.problem_wcvrp import CWCVRP
from logic.src.models.attention_model import AttentionModel
from logic.src.models.subnets.gat_encoder import GraphAttentionEncoder
from logic.src.policies.single_vehicle import local_search_2opt_vectorized

# --- UTILS ---

def get_batch(device, size=50, batch_size=32):
    all_coords = torch.rand(batch_size, size+1, 2, device=device)
    depot = all_coords[:, 0, :]
    loc = all_coords[:, 1:, :]
    dist_tensor = torch.cdist(all_coords, all_coords)
    waste = torch.rand(batch_size, size, device=device)
    max_waste = torch.ones(batch_size, device=device)
    
    return {
        'depot': depot, 'loc': loc, 'dist': dist_tensor,
        'demand': waste, 'waste': waste, 'max_waste': max_waste
    }

class MyModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input, cost_weights=None, return_pi=False, pad=False, mask=None, expert_pi=None):
        return self.model(input, cost_weights, return_pi, pad, mask, expert_pi)

def load_model_instance(model_path, device, size=100):
    model = AttentionModel(
        embedding_dim=128, hidden_dim=512, problem=CWCVRP(),
        encoder_class=GraphAttentionEncoder, n_encode_layers=3,
        mask_inner=True, mask_logits=True, normalization='instance',
        tanh_clipping=10.0, checkpoint_encoder=False, shrink_size=None,
        n_heads=8, n_encode_sublayers=1, n_decode_layers=2, predictor_layers=2
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

# --- VISUALIZATION FUNCTIONS ---

def plot_weight_trajectories(checkpoint_dir, output_file):
    print("Computing Weight Trajectories...")
    # Create output dir if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    # Sort files by epoch number
    files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]) if '-' in x else 0)
    
    weights = []
    epochs = []
    
    for f in files:
        path = os.path.join(checkpoint_dir, f)
        checkpoint = torch.load(path, map_location='cpu')
        flat_weight = torch.cat([p.flatten() for p in checkpoint['model'].values()]).numpy()
        weights.append(flat_weight)
        epochs.append(f.replace('.pt', ''))
        
    weights = np.array(weights)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(weights)
    
    plt.figure(figsize=(10, 8))
    plt.plot(projected[:, 0], projected[:, 1], '-o', alpha=0.6)
    for i, txt in enumerate(epochs):
        plt.annotate(txt, (projected[i, 0], projected[i, 1]), size=8)
    
    plt.title('Training Trajectory (PCA Projection of Model Weights)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.savefig(output_file)
    print(f"Trajectory saved to {output_file}")

def log_weight_distributions(checkpoint_dir, log_dir):
    print("Logging Weight Distributions to TensorBoard...")
    writer = SummaryWriter(log_dir)
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]) if '-' in x else 0)
    
    for f in files:
        epoch = int(f.split('-')[1].split('.')[0]) if '-' in f else 0
        path = os.path.join(checkpoint_dir, f)
        checkpoint = torch.load(path, map_location='cpu')
        for name, param in checkpoint['model'].items():
            writer.add_histogram(name, param, epoch)
    writer.close()
    print(f"Distributions logged to {log_dir}")

def project_node_embeddings(model_path, x_batch, log_dir):
    print("Projecting Node Embeddings to TensorBoard...")
    device = torch.device('cpu')
    model = load_model_instance(model_path, device)
    
    writer = SummaryWriter(log_dir)
    with torch.no_grad():
        # Get embeddings from the embedder
        # model.embedder(model._init_embed(x_batch))
        # But AttentionModel._init_embed is internal. 
        # Let's run a partial forward.
        nodes = model._init_embed(x_batch)
        embeddings = model.embedder(nodes) # (B, G, D)
        
        # Take the first sample in batch
        sample_embeddings = embeddings[0] # (G, D)
        labels = [f"Node_{i}" for i in range(sample_embeddings.size(0))]
        labels[0] = "Depot"
        
        writer.add_embedding(sample_embeddings, metadata=labels, tag="Node_Embeddings")
    writer.close()
    print(f"Embeddings projected to {log_dir}")

def plot_attention_heatmaps(model_path, output_dir):
    print("Plotting Attention Heatmaps...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    device = torch.device('cpu')
    model = load_model_instance(model_path, device)
    
    # Encoder attention weights
    # Structure: model.embedder.layers[i].att.module.W_query.weight etc.
    # Looking at gat_encoder.py: self.layers[i].att.module is MultiHeadAttention
    
    for i, layer in enumerate(model.embedder.layers):
        # att is SkipConnection, att.module is MultiHeadAttention
        mha = layer.att.module
        
        # W_query, W_key, W_val are typically the names in MultiHeadAttention
        for name in ['W_query', 'W_key', 'W_val']:
            if hasattr(mha, name):
                param = getattr(mha, name)
                if hasattr(param, 'weight'):
                    weight = param.weight.data.numpy()
                else:
                    weight = param.data.numpy()
                
                # Handle 3D weights (Multi-Head Attention)
                if weight.ndim == 3:
                    # (Heads, Input, Output) -> (Input, Heads * Output)
                    h, input_dim, o = weight.shape
                    weight = weight.transpose(1, 0, 2).reshape(input_dim, h * o)
                elif weight.ndim > 2:
                    # Generic flatten to 2D
                    weight = weight.reshape(weight.shape[0], -1)
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(weight, cmap='RdBu_r', center=0)
                plt.title(f'Layer {i} - {name} Weights')
                plt.savefig(os.path.join(output_dir, f'layer_{i}_{name}.png'))
                plt.close()
                
    print(f"Heatmaps saved to {output_dir}")

# --- LOSS LANDSCAPE FUNCTIONS (Existing) ---

def imitation_loss_fn(m, x_batch, pi_target):
    model_to_call = m.modules[0]
    if hasattr(model_to_call, 'model'):
        model_to_call = model_to_call.model
    model_to_call.eval()
    with torch.no_grad():
        res = model_to_call(x_batch, return_pi=False, expert_pi=pi_target)
        log_likelihood = res[1]
    return -log_likelihood.mean().item()

def rl_loss_fn(m, x_batch):
    model_to_call = m.modules[0]
    if hasattr(model_to_call, 'model'):
        model_to_call = model_to_call.model
    model_to_call.eval()
    with torch.no_grad():
        model_to_call.set_decode_type('greedy')
        cost, _, _, _, _ = model_to_call(x_batch, return_pi=False)
    return cost.float().mean().item()

def plot_loss_landscape(model_path, args):
    device = torch.device('cpu')
    model = load_model_instance(model_path, device, size=args.size)
    x_batch = get_batch(device, size=args.size, batch_size=args.batch_size)
    
    print("Generating expert targets...")
    model.set_decode_type('greedy')
    with torch.no_grad():
        _, _, _, pi, _ = model(x_batch, return_pi=True)
        x_dist = x_batch['dist']
        if x_dist.dim() == 2: x_dist = x_dist.unsqueeze(0)
        if x_dist.size(0) == 1: x_dist = x_dist.expand(pi.size(0), -1, -1)
        pi_with_depot = torch.cat([torch.zeros((pi.size(0), 1), dtype=torch.long, device=device), pi], dim=1)
        pi_opt = local_search_2opt_vectorized(pi_with_depot, x_dist, max_iterations=100)
        pi_target = pi_opt[:, 1:]

    wrapped_model = MyModelWrapper(model)

    if args.mode in ['imitation', 'both']:
        print("Computing Imitation Landscape...")
        metric = lambda m: imitation_loss_fn(m, x_batch, pi_target)
        data = loss_landscapes.random_plane(wrapped_model, metric, distance=args.span, steps=args.resolution, deepcopy_model=True)
        
        plt.figure()
        plt.contour(data, levels=50)
        plt.title('Imitation Loss Landscape')
        plt.savefig('landscape_imitation.png')
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.arange(args.resolution), np.arange(args.resolution))
        ax.plot_surface(X, Y, np.array(data), cmap='viridis')
        plt.title('Imitation Loss Surface')
        plt.savefig('surface_imitation.png')

    if args.mode in ['rl', 'both']:
        print("Computing RL Cost Landscape...")
        metric = lambda m: rl_loss_fn(m, x_batch)
        data = loss_landscapes.random_plane(wrapped_model, metric, distance=args.span, steps=args.resolution, deepcopy_model=True)
        
        plt.figure()
        plt.contour(data, levels=50)
        plt.title('RL Cost Landscape')
        plt.savefig('landscape_rl.png')
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.arange(args.resolution), np.arange(args.resolution))
        ax.plot_surface(X, Y, np.array(data), cmap='magma')
        plt.title('RL Cost Surface')
        plt.savefig('surface_rl.png')

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory with multiple checkpoints')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory')
    parser.add_argument('--log_dir', type=str, default='runs/viz', help='TensorBoard log directory')
    
    parser.add_argument('--size', type=int, default=100, help='Problem size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--resolution', type=int, default=10, help='Resolution for landscapes')
    parser.add_argument('--span', type=float, default=1.0, help='Span for landscapes')
    
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['trajectory', 'distributions', 'embeddings', 'heatmaps', 'loss', 'both'],
                        help='Visualization mode')
    
    args = parser.parse_args()
    
    if args.mode == 'trajectory':
        if not args.checkpoint_dir: raise ValueError("--checkpoint_dir required for trajectory")
        plot_weight_trajectories(args.checkpoint_dir, os.path.join(args.output_dir, 'trajectory.png'))
        
    elif args.mode == 'distributions':
        if not args.checkpoint_dir: raise ValueError("--checkpoint_dir required for distributions")
        log_weight_distributions(args.checkpoint_dir, args.log_dir)
        
    elif args.mode == 'embeddings':
        if not args.model_path: raise ValueError("--model_path required for embeddings")
        x_batch = get_batch(torch.device('cpu'), size=args.size, batch_size=1)
        project_node_embeddings(args.model_path, x_batch, args.log_dir)
        
    elif args.mode == 'heatmaps':
        if not args.model_path: raise ValueError("--model_path required for heatmaps")
        plot_attention_heatmaps(args.model_path, args.output_dir)
        
    elif args.mode == 'loss' or args.mode == 'both':
        if not args.model_path: raise ValueError("--model_path required for loss landscape")
        plot_loss_landscape(args.model_path, args)

if __name__ == "__main__":
    main()

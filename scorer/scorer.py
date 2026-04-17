import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from collections import defaultdict
import json
import os
import argparse 


def load_data(input_path='', normalize=True):

    with open(os.path.join(input_path, 'train_data_group.pkl'), 'rb') as f:
        print(f"Load train_data_group.pkl from {os.path.join(input_path, 'train_data_group.pkl')}")
        train_data = pickle.load(f)
    
    with open(os.path.join(input_path, 'test_data_group.pkl'), 'rb') as f:
        print(f"Load test_data_group.pkl from {os.path.join(input_path, 'test_data_group.pkl')}")
        test_data = pickle.load(f)

    
    train_X = np.array([item[0] for item in train_data])
    train_y = np.array([item[1] for item in train_data])
    
    if len(train_data[0]) > 2:
        train_group = np.array([item[2] for item in train_data])
    else:
        train_group = np.zeros(len(train_data))
    
    test_X = np.array([item[0] for item in test_data])
    test_y = np.array([item[1] for item in test_data])
    
    if len(test_data[0]) > 2:
        test_group = np.array([item[2] for item in test_data])
    else:
        test_group = np.zeros(len(test_data))
    
    if normalize:
        original_shape = train_X.shape
        train_X_flat = train_X.reshape(train_X.shape[0], -1)
        test_X_flat = test_X.reshape(test_X.shape[0], -1)
        
        scaler = StandardScaler()
        train_X_flat = scaler.fit_transform(train_X_flat)

        scaler_params = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }

        test_X_flat = scaler.transform(test_X_flat)
        
        train_X = train_X_flat.reshape(original_shape)
        test_X = test_X_flat.reshape(test_X.shape)
    
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    train_group = torch.tensor(train_group, dtype=torch.int64)
    
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)
    test_group = torch.tensor(test_group, dtype=torch.int64)

    
    return train_X, train_y, train_group, test_X, test_y, test_group, scaler_params

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bottleneck_size: int = 32, 
        conv_lengths: list = [10, 20, 40], 
        conv_filters: int = 32 
    ):
        super().__init__()
        self.relu = nn.ReLU()

        self.bottleneck = nn.Conv1d(
            in_channels=in_channels,
            out_channels=bottleneck_size,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.conv1 = nn.Conv1d(
            in_channels=bottleneck_size,
            out_channels=conv_filters,
            kernel_size=conv_lengths[0],
            stride=1,
            padding=(conv_lengths[0]-1) // 2
        )
        self.conv2 = nn.Conv1d(
            in_channels=bottleneck_size,
            out_channels=conv_filters,
            kernel_size=conv_lengths[1],
            stride=1,
            padding=(conv_lengths[1]-1) // 2
        )
        self.conv3 = nn.Conv1d(
            in_channels=bottleneck_size,
            out_channels=conv_filters,
            kernel_size=conv_lengths[2],
            stride=1,
            padding=(conv_lengths[2]-1) // 2
        )
        
        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stride=1,
            padding=0
        )
        self.pool_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=conv_filters,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.out_channels = 4 * conv_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_length = x.shape
        
        bottleneck_out = self.relu(self.bottleneck(x))
        conv1_out = self.relu(self.conv1(bottleneck_out)) 
        conv2_out = self.relu(self.conv2(bottleneck_out))
        conv3_out = self.relu(self.conv3(bottleneck_out))
        
        pool_out = self.pool(x)
        pool_conv_out = self.relu(self.pool_conv(pool_out))
        
        out = torch.cat([conv1_out, conv2_out, conv3_out, pool_conv_out], dim=1)
        return out

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bottleneck_size: int = 32,
        conv_lengths: list = [10, 20, 40],
        conv_filters: int = 32
    ):
        super().__init__()
        
        self.inception1 = InceptionModule(
            in_channels=in_channels,
            bottleneck_size=bottleneck_size,
            conv_lengths=conv_lengths,
            conv_filters=conv_filters
        )
        self.inception2 = InceptionModule(
            in_channels=self.inception1.out_channels,
            bottleneck_size=bottleneck_size,
            conv_lengths=conv_lengths,
            conv_filters=conv_filters
        )
        self.inception3 = InceptionModule(
            in_channels=self.inception2.out_channels,
            bottleneck_size=bottleneck_size,
            conv_lengths=conv_lengths,
            conv_filters=conv_filters
        )
        
        self.out_channels = self.inception3.out_channels
        self.shortcut = nn.Sequential()
        if in_channels != self.out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inception1(x)
        out = self.inception2(out)
        out = self.inception3(out)
        
        residual = self.shortcut(x)
        residual = nn.functional.adaptive_avg_pool1d(residual, out.shape[-1])
        
        out += residual
        out = self.relu(out)
        return out

class InceptionNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1, 
        num_classes: int = 1, 
        num_residual_blocks: int = 2, 
        bottleneck_size: int = 32,
        conv_lengths: list = [10, 20, 40],
        conv_filters: int = 32
    ):
        super().__init__()
        
        self.residual_blocks = nn.Sequential()
        in_ch = in_channels
        for i in range(num_residual_blocks):
            rb = ResidualBlock(
                in_channels=in_ch,
                bottleneck_size=bottleneck_size,
                conv_lengths=conv_lengths,
                conv_filters=conv_filters
            )
            self.residual_blocks.add_module(f"res_block_{i}", rb)
            in_ch = rb.out_channels
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(in_ch, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.residual_blocks(x)
        out = self.global_avg_pool(out)
        out = out.squeeze(-1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


class InceptionTime(nn.Module):
    def __init__(self, num_ensemble: int = 5, **kwargs):
        super().__init__()
        
        self.models = nn.ModuleList([InceptionNet(** kwargs) for _ in range(num_ensemble)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([model(x) for model in self.models], dim=0)
        avg_output = torch.mean(outputs, dim=0)
        return avg_output

    def get_individual_models(self) -> list:
        return self.models



def compute_group_auc(y_true, y_pred_proba, groups):
    group_data = defaultdict(lambda: {'y_true': [], 'y_pred_proba': []})
    for true, pred, group in zip(y_true, y_pred_proba, groups):
        group_data[group]['y_true'].append(true)
        group_data[group]['y_pred_proba'].append(pred)
    
    group_aucs = {}
    valid_groups = 0
    
    for group_id, data in group_data.items():
        y_true_group = data['y_true']
        y_pred_proba_group = data['y_pred_proba']
        
        try:
            if len(set(y_true_group)) > 1:
                auc = roc_auc_score(y_true_group, y_pred_proba_group)
                group_aucs[group_id] = auc
                valid_groups += 1
        except ValueError:
            continue
    

    if valid_groups > 0:
        avg_group_auc = np.mean(list(group_aucs.values()))
        return avg_group_auc    
    else:
        return None


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device) -> float:
    model.eval()
    y_pred = []
    y_true = []
    y_prob = []
    groups = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                X, y, group_ids = batch
                groups.extend(group_ids.cpu().numpy())
            else:
                X, y = batch
            
            X, y = X.to(device), y.to(device).unsqueeze(1)
            outputs = model(X)
            predicted = (outputs > 0.5).float().squeeze()
            
            y_pred.extend(predicted.cpu().numpy().tolist())
            y_true.extend(y.cpu().numpy().flatten().tolist())
            y_prob.extend(outputs.cpu().numpy().flatten().tolist())
    
    accuracy = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_prob)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Overall AUC Score: {auc_score:.4f}') 
    

    if groups:
        print('Calculating Group AUC...')
        group_auc = compute_group_auc(y_true, y_prob, groups)
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred))
    
    return accuracy, group_auc


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = 50) -> list[float]:
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            if len(batch) == 3:
                X, y, _ = batch
            else:
                X, y = batch
                
            X, y = X.to(device), y.to(device).unsqueeze(1)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f}")
    return train_losses

def main():
    parser = argparse.ArgumentParser(description='scorer')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_ensemble', type=int, default=3)
    
    parser.add_argument('--num_residual_blocks', type=int, default=2)
    parser.add_argument('--bottleneck_size', type=int, default=8)
    parser.add_argument('--conv_filters', type=int, default=8)
    parser.add_argument('--conv_lengths', type=int, nargs=3, default=[10, 20, 40])
    parser.add_argument('--exp_name', type=str, default='exp')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading data...")
    train_X, train_y, train_group, test_X, test_y, test_group, scaler_params = load_data(input_path=args.input_path)
    if len(train_X.shape) == 2:
        train_X = train_X.unsqueeze(1) 
        test_X = test_X.unsqueeze(1) 
    

    train_dataset = TensorDataset(train_X, train_y, train_group)
    test_dataset = TensorDataset(test_X, test_y, test_group)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    in_channels = train_X.shape[1]
    num_classes = 1
    
    # 初始化模型
    inception_time = InceptionTime(
        num_ensemble=args.num_ensemble,
        in_channels=in_channels,
        num_classes=num_classes,
        num_residual_blocks=args.num_residual_blocks,
        bottleneck_size=args.bottleneck_size,
        conv_lengths=args.conv_lengths,
        conv_filters=args.conv_filters
    ).to(device)

    print(f"Total trainable parameters: {count_parameters(inception_time):,}")
    

    criterion = nn.BCELoss()
    individual_models = inception_time.get_individual_models()
    individual_losses = []

    print("\n===== Train =====")
    
    for i, model in enumerate(individual_models, 1):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        losses = train_single_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=epochs
        )
        individual_losses.append(losses)

    print("\n===== Test =====")
    individual_accs = []
    for i, model in enumerate(individual_models, 1):
        acc, group_auc = evaluate_model(model, test_loader, device)
        individual_accs.append(acc)
        print(f"{i}-th model test acc: {acc:.4f}")
        if group_auc is not None:
            print(f"{i}-th model test group_auc: {group_auc:.4f}")
        
    ensemble_acc, ensemble_group_auc = evaluate_model(inception_time, test_loader, device)

    print(f"{args.exp_name} group_auc: {ensemble_group_auc:.4f}\
        \n test acc: {ensemble_acc:.4f} \
        \n num_residual_blocks: {args.num_residual_blocks} \
        \n bottleneck_size: {args.bottleneck_size} \
        \n conv_filters: {args.conv_filters} \
        \n conv_lengths: {args.conv_lengths} \
        \n epochs: {args.epochs}")

    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(os.path.join(args.output_path, args.exp_name)):
        os.makedirs(os.path.join(args.output_path, args.exp_name))

    torch.save(inception_time.state_dict(), f'{args.output_path}/{args.exp_name}/model.pth')

    config = {
        'num_ensemble': args.num_ensemble,
        'num_residual_blocks': args.num_residual_blocks,
        'bottleneck_size': args.bottleneck_size,
        'conv_filters': args.conv_filters,
        'conv_lengths': args.conv_lengths,
        'scaler_params': scaler_params
    }
    with open(f'{args.output_path}/{args.exp_name}/config.json', 'w') as f:
        json.dump(config, f, indent=4)



if __name__ == "__main__":
    main()
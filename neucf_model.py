# neucf_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuCF(nn.Module):
    """
    NeuCF: GMF + MLP fusion with final prediction.
    Accepts either user/item indices (and looks up embeddings)
    OR accepts precomputed user/item embedding tensors (for pseudo users).
    """
    def __init__(self, n_users, n_items, emb_size=64, mlp_layers=[128,64,32], dropout=0.2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_size = emb_size

        # GMF embeddings
        self.user_gmf = nn.Embedding(n_users, emb_size)
        self.item_gmf = nn.Embedding(n_items, emb_size)

        # MLP embeddings 
        self.user_mlp = nn.Embedding(n_users, emb_size)
        self.item_mlp = nn.Embedding(n_items, emb_size)

        # MLP layers
        mlp_input = emb_size * 2
        mlp_modules = []
        for h in mlp_layers:
            mlp_modules.append(nn.Linear(mlp_input, h))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_input = h
        self.mlp = nn.Sequential(*mlp_modules)

        # Final prediction layers
        predict_size = emb_size + mlp_layers[-1]
        self.predict = nn.Sequential(
            nn.Linear(predict_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_gmf.weight, std=0.01)
        nn.init.normal_(self.item_gmf.weight, std=0.01)
        nn.init.normal_(self.user_mlp.weight, std=0.01)
        nn.init.normal_(self.item_mlp.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        for m in self.predict:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user_idx=None, item_idx=None, user_emb=None, item_emb=None):
        """
        Either provide (user_idx, item_idx) as LongTensors
        OR provide (user_emb, item_emb) as FloatTensors (already embedded).
        Embedding shapes: (batch, emb_size)
        """
        if user_emb is None or item_emb is None:
            # use indices and embeddings
            assert user_idx is not None and item_idx is not None
            g_u = self.user_gmf(user_idx)
            g_i = self.item_gmf(item_idx)
            m_u = self.user_mlp(user_idx)
            m_i = self.item_mlp(item_idx)
        else:
            # provided embeddings (float tensors)
            g_u = user_emb
            g_i = item_emb
            m_u = user_emb
            m_i = item_emb

        # GMF part: elementwise product
        gmf_out = g_u * g_i  # (batch, emb)

        # MLP part: concat and feed MLP
        mlp_in = torch.cat([m_u, m_i], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # concat GMF and MLP outputs
        x = torch.cat([gmf_out, mlp_out], dim=-1)
        out = self.predict(x).squeeze(-1)  # (batch,)
        return out

    def user_embedding(self, user_idx):
        """Return the trained user embedding (combined if needed)."""
        # return GMF user emb (could also average GMF+MLP)
        return (self.user_gmf(user_idx).detach().cpu().numpy())

    def item_embedding_matrix(self):
        """Return numpy array of all item embeddings (GMF part)."""
        with torch.no_grad():
            return self.item_gmf.weight.detach().cpu().numpy()

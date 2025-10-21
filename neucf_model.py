# neucf_model.py (replace existing file)
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuCF(nn.Module):
    """
    NeuCF with item side-features (tags + rating).
    item_feat: FloatTensor (batch, n_item_features)
    """
    def __init__(self, n_users, n_items, emb_size=64, mlp_layers=[128,64,32],
                 dropout=0.2, n_item_features=0, feat_emb_size=None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_size = emb_size
        self.n_item_features = n_item_features
        if feat_emb_size is None:
            feat_emb_size = emb_size  # default project features to same emb size
        self.feat_emb_size = feat_emb_size

        # GMF embeddings
        self.user_gmf = nn.Embedding(n_users, emb_size)
        self.item_gmf = nn.Embedding(n_items, emb_size)

        # MLP embeddings (smaller or same)
        self.user_mlp = nn.Embedding(n_users, emb_size)
        self.item_mlp = nn.Embedding(n_items, emb_size)

        # feature projector (from item features to vector)
        if n_item_features > 0:
            self.feat_proj = nn.Sequential(
                nn.Linear(n_item_features, feat_emb_size),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )
        else:
            self.feat_proj = None

        # MLP layers
        # mlp input is user_mlp + item_mlp + feat_emb (if any)
        mlp_input = emb_size * 2 + (feat_emb_size if self.feat_proj is not None else 0)
        mlp_modules = []
        for h in mlp_layers:
            mlp_modules.append(nn.Linear(mlp_input, h))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_input = h
        self.mlp = nn.Sequential(*mlp_modules)

        # Final prediction layers
        predict_size = emb_size + (mlp_layers[-1] if mlp_layers else 0)
        self.predict = nn.Sequential(
            nn.Linear(predict_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # keep sigmoid if using BCE; remove if using pairwise loss
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_gmf.weight, std=0.01)
        nn.init.normal_(self.item_gmf.weight, std=0.01)
        nn.init.normal_(self.user_mlp.weight, std=0.01)
        nn.init.normal_(self.item_mlp.weight, std=0.01)
        if self.feat_proj is not None:
            # init linear inside feat_proj
            for m in self.feat_proj:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        for m in self.predict:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user_idx=None, item_idx=None, user_emb=None, item_emb=None, item_feat=None):
        """
        Either provide (user_idx, item_idx) as LongTensors and item_feat (FloatTensor), OR
        provide (user_emb, item_emb) as FloatTensors (already embedded) along with item_feat optionally.
        item_feat shape: (batch, n_item_features)
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

        # project item features if available
        feat_vec = None
        if self.feat_proj is not None:
            assert item_feat is not None, "item_feat required when model constructed with n_item_features>0"
            feat_vec = self.feat_proj(item_feat)  # (batch, feat_emb_size)

        # GMF part: elementwise product
        gmf_out = g_u * g_i  # (batch, emb)

        # MLP part: concat and feed MLP (include feat_vec if present)
        if feat_vec is not None:
            mlp_in = torch.cat([m_u, m_i, feat_vec], dim=-1)
        else:
            mlp_in = torch.cat([m_u, m_i], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # concat GMF and MLP outputs
        x = torch.cat([gmf_out, mlp_out], dim=-1)
        out = self.predict(x).squeeze(-1)  # (batch,)
        return out

    def user_embedding(self, user_idx):
        """Return the trained user GMF embedding (numpy)."""
        return self.user_gmf(user_idx).detach().cpu().numpy()

    def item_embedding_matrix(self):
        """Return numpy array of all item GMF embeddings (GMF part)."""
        with torch.no_grad():
            return self.item_gmf.weight.detach().cpu().numpy()

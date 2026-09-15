"""
model.py
--------
Model GraphSAGE pentru graf eterogen (2 tipuri de noduri: patient, episode).

Folosim torch_geometric.nn.SAGEConv în interiorul unui HeteroConv, ceea ce
înseamnă că pentru fiecare tip de muchie avem un SAGEConv separat, iar
rezultatele pentru fiecare tip de nod se agregă (sum) la finalul fiecărui
layer. Asta e echivalentul heterogen standard al GraphSAGE în PyG.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=2,
                 num_layers=2, dropout=0.3):
        """
        metadata: data.metadata() -> (node_types, edge_types), obținut din
                  HeteroData; necesar ca HeteroConv să știe ce relații există.
        out_channels: numărul de clase (2 pentru anomalie binară).
        """
        super().__init__()
        self.dropout = dropout

        # proiectăm fiecare tip de nod la aceeași dimensiune ascunsă,
        # pentru că 'patient' și 'episode' au numar diferit de features brute
        self.input_proj = torch.nn.ModuleDict({
            node_type: Linear(-1, hidden_channels)
            for node_type in metadata[0]
        })

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((-1, -1), hidden_channels)
                    for edge_type in metadata[1]
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.input_proj[node_type](x).relu()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                node_type: F.dropout(x.relu(), p=self.dropout, training=self.training)
                for node_type, x in x_dict.items()
            }

        # ne interesează doar predicția pe nodurile de tip 'episode'
        out = self.classifier(x_dict["episode"])
        return out

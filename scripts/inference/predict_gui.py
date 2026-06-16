import sys
import os
import torch
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import configs.config_reg as cfg

from atomprop.dataloader.dataloader import SMILESToInputs
from atomprop.models.gnns import Embedder, GNNAggr
from atomprop.models.geat import GeATNet
from atomprop.utils.mlp import MLP

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


class PredictWorker(QThread):
    result_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, smiles_list, model_path, device):
        super().__init__()
        self.smiles_list = [s.strip() for s in smiles_list if s.strip()]
        self.model_path = model_path
        self.device = device

    def run(self):
        try:
            embedding_layer = Embedder(num_atom_types=120, embed_dim=cfg.embed_dim)
            backbone = GeATNet(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                global_num_heads=cfg.global_num_heads,
                output_negative_slope=cfg.output_negative_slope,
                dropout=cfg.geat_dropout,
                geat_num_layers=cfg.geat_num_layers,
                aggr_num_layers=cfg.aggr_num_layers,
                FFN_type=cfg.FFN_type,
                FFN_hidden_dim=cfg.FFN_hidden_dim,
                FFN_num_experts=cfg.FFN_num_experts,
                FFN_num_layers=cfg.FFN_num_layers,
                FFN_top_k=cfg.FFN_top_k,
                use_edge_embedding=cfg.use_edge_embedding
            )
            aggrmodel = GNNAggr(embed_dim=cfg.embed_dim, aggr=cfg.aggr, layers=1)
            head = MLP(
                input_dim=cfg.embed_dim,
                hidden_dim=cfg.head_hidden_dim,
                output_dim=1,
                num_layers=cfg.head_layers,
                dropout=cfg.head_dropout,
                batch_norm=True,
                output_activation=None
            )

            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            embedding_layer.load_state_dict(ckpt['embedding_layer_state_dict'])
            backbone.load_state_dict(ckpt['backbone_state_dict'])
            head.load_state_dict(ckpt['head_state_dict'])
            if cfg.aggr == 'attention' and ckpt.get('aggr_state_dict'):
                aggrmodel.load_state_dict(ckpt['aggr_state_dict'])

            scaler_stats = ckpt.get('scaler_stats')
            embedding_layer.to(self.device).eval()
            backbone.to(self.device).eval()
            head.to(self.device).eval()
            aggrmodel.to(self.device).eval()

            dataset = []
            valid_smiles = []
            for smi in self.smiles_list:
                atom_info, edge_info, _ = SMILESToInputs.convert(smi, sanitize=False)
                if atom_info is None or edge_info is None:
                    continue
                if edge_info.dim() == 2 and edge_info.size(1) == 4:
                    edge_index = edge_info[:, :2].t().contiguous()
                    edge_attr = edge_info[:, 2:]
                else:
                    edge_index = torch.tensor([[], []], dtype=torch.long)
                    edge_attr = torch.tensor([], dtype=torch.long).view(0, 2)
                dataset.append(Data(x=atom_info, edge_index=edge_index, edge_attr=edge_attr))
                valid_smiles.append(smi)

            loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=Batch.from_data_list)
            preds = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    emb = embedding_layer(batch.x.squeeze())
                    emb = backbone(Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr), batch=batch.batch)
                    g_emb = aggrmodel(emb, batch.batch)
                    out = head(g_emb).cpu().numpy().flatten()
                    preds.extend(out)

            if scaler_stats is not None:
                preds = np.array(preds) * scaler_stats["scale"][0] + scaler_stats["mean"][0]

            self.result_ready.emit(list(zip(valid_smiles, preds)))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))


class PredictGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMILES Prediction Tool")
        self.setGeometry(100, 100, 900, 720)
        self.setAcceptDrops(True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = None
        self.results = []
        self.initUI()

    def initUI(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        title = QLabel("SMILES Property Predictor")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        model_row = QHBoxLayout()
        self.model_label = QLabel("No model loaded")
        self.model_btn = QPushButton("Load Model (.pth)")
        model_row.addWidget(QLabel("Model:"))
        model_row.addWidget(self.model_label)
        model_row.addWidget(self.model_btn)
        layout.addLayout(model_row)

        layout.addWidget(QLabel("Input SMILES (one per line) or drag TXT/CSV here:"))
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Drag a file or paste SMILES...")
        layout.addWidget(self.text_input)

        btn_row = QHBoxLayout()
        self.predict_btn = QPushButton("Predict")
        self.export_btn = QPushButton("Export CSV")
        btn_row.addWidget(self.predict_btn)
        btn_row.addWidget(self.export_btn)
        layout.addLayout(btn_row)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["SMILES", "Predicted Value"])
        layout.addWidget(self.table)

        self.log_label = QLabel("Ready")
        layout.addWidget(self.log_label)

        self.model_btn.clicked.connect(self.load_model)
        self.predict_btn.clicked.connect(self.run_prediction)
        self.export_btn.clicked.connect(self.export_csv)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(filter="Model Files (*.pth)")
        if path:
            self.model_path = path
            self.model_label.setText(os.path.basename(path))

    def run_prediction(self):
        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please load a model first!")
            return
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Warning", "Please input SMILES!")
            return

        self.log_label.setText("Predicting...")
        self.predict_btn.setEnabled(False)
        self.worker = PredictWorker(text.splitlines(), self.model_path, self.device)
        self.worker.result_ready.connect(self.show_results)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.start()

    def show_results(self, results):
        self.results = results
        self.table.setRowCount(len(results))
        for i, (smi, val) in enumerate(results):
            self.table.setItem(i, 0, QTableWidgetItem(smi))
            self.table.setItem(i, 1, QTableWidgetItem(f"{val:.6f}"))
        self.log_label.setText(f"Done: {len(results)} molecules")
        self.predict_btn.setEnabled(True)

    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.log_label.setText("Error")
        self.predict_btn.setEnabled(True)

    def export_csv(self):
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to export!")
            return
        path, _ = QFileDialog.getSaveFileName(filter="CSV Files (*.csv)")
        if path:
            df = pd.DataFrame(self.results, columns=["SMILES", "predicted_value"])
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Success", "CSV saved!")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            urls = e.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if file_path.startswith('file://'):
                    file_path = file_path[7:]
                if file_path.lower().endswith(('.txt', '.csv')):
                    e.accept()
                    return
        e.ignore()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if not urls:
            return
        
        file_path = urls[0].toLocalFile()
        if file_path.startswith('file://'):
            file_path = file_path[7:]
        
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Warning", f"File not found:\n{file_path}")
            return
        
        if file_path.lower().endswith((".txt", ".csv")):
            try:
                if file_path.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, encoding='gbk')
                    
                    smiles_col = None
                    for col in df.columns:
                        col_lower = col.lower()
                        if col_lower in ['smiles', 'smi', 'canonical_smiles', 'smiles_string']:
                            smiles_col = col
                            break
                    
                    if smiles_col is None and len(df.columns) > 0:
                        smiles_col = df.columns[0]
                    
                    if smiles_col:
                        smiles_list = df[smiles_col].astype(str).tolist()
                        smiles_list = [s.strip() for s in smiles_list if s and s.lower() != 'nan' and s.strip()]
                        smiles_text = '\n'.join(smiles_list)
                        self.text_input.setPlainText(smiles_text)
                        self.log_label.setText(f"Loaded {len(smiles_list)} SMILES from {os.path.basename(file_path)}")
                    else:
                        self.log_label.setText(f"Failed to find data in CSV")
                        
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self.text_input.setPlainText(content)
                    self.log_label.setText(f"Loaded: {os.path.basename(file_path)}")
                
                e.accept()
                
            except Exception as ex:
                error_msg = f"Failed to load file:\n{str(ex)}"
                self.log_label.setText("Error loading file")
                QMessageBox.critical(self, "Error", error_msg)
        else:
            QMessageBox.warning(self, "Warning", "Please drop a .txt or .csv file")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PredictGUI()
    gui.show()
    sys.exit(app.exec_())
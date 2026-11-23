"""
RumahPoint - Web Service Deployment (Flask + Keras + Scikit-Learn)
Model Servis Prediksi Kelas Harga Properti Multi-Kota
Port: 7860 (Hugging Face Spaces Default)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

# Inisialisasi Flask App
app = Flask(__name__)

# Konfigurasi Path Artefak Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_rumahpoint.keras")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")

# Fallback lokasi Top-15 jika metadata belum dimuat
DEFAULT_TOP15 = [
    'Kota Jakarta Selatan', 'Kota Surabaya', 'Kota Bandung', 'Kota Bekasi',
    'Kota Tangerang', 'Kota Depok', 'Kota Semarang', 'Kota Medan',
    'Kota Makassar', 'Kota Tangerang Selatan', 'Kota Bogor', 'Kota Malang',
    'Kab. Sleman', 'Kab. Badung', 'Kota Denpasar'
]

# Variabel Global Artefak
model = None
preprocessor = None
metadata = None

def load_artifacts():
    """Memuat artefak model, preprocessor, dan metadata dari disk."""
    global model, preprocessor, metadata
    
    # 1. Pemuatan Metadata
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"[INFO] Metadata berhasil dimuat dari {METADATA_PATH}")
        except Exception as e:
            print(f"[ERROR] Gagal membaca metadata.json: {e}")
            metadata = None
    else:
        print(f"[WARN] File metadata tidak ditemukan pada {METADATA_PATH}")
        metadata = None

    # 2. Pemuatan Preprocessor (ColumnTransformer)
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print(f"[INFO] Preprocessor berhasil dimuat dari {PREPROCESSOR_PATH}")
        except Exception as e:
            print(f"[ERROR] Gagal memuat preprocessor.joblib: {e}")
            preprocessor = None
    else:
        print(f"[WARN] File preprocessor tidak ditemukan pada {PREPROCESSOR_PATH}")
        preprocessor = None

    # 3. Pemuatan Model Keras 3 Native (.keras)
    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            # Gunakan tf.keras.models.load_model atau keras.models.load_model
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[INFO] Model Keras berhasil dimuat dari {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Gagal memuat model Keras: {e}")
            model = None
    else:
        print(f"[WARN] File model tidak ditemukan pada {MODEL_PATH}")
        model = None

# Muat artefak saat modul diinisialisasi
load_artifacts()

# Template HTML Antarmuka Web Interaktif
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RumahPoint - Prediksi Kelas Harga Properti</title>
    <style>
        :root {
            --primary: #1e3a8a;
            --primary-light: #3b82f6;
            --secondary: #0d9488;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --success: #16a34a;
            --warning: #ca8a04;
            --danger: #dc2626;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            line-height: 1.6;
            padding: 24px 16px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 28px;
            padding: 24px;
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #0d9488 100%);
            color: white;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
        }
        .header h1 { font-size: 28px; font-weight: 800; margin-bottom: 8px; letter-spacing: -0.5px; }
        .header p { font-size: 15px; opacity: 0.9; }
        .status-badge {
            display: inline-block;
            margin-top: 12px;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .badge-ready { background: #dcfce7; color: #166534; }
        .badge-warning { background: #fef3c7; color: #92400e; }
        .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 24px;
        }
        @media (min-width: 768px) {
            .grid { grid-template-columns: 1.1fr 0.9fr; }
        }
        .card {
            background: var(--card-bg);
            border-radius: 14px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border);
        }
        .card h2 { font-size: 18px; font-weight: 700; margin-bottom: 16px; color: var(--primary); }
        .form-group { margin-bottom: 16px; }
        .form-group label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-main);
        }
        .form-control {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.15s ease;
        }
        .form-control:focus {
            outline: none;
            border-color: var(--primary-light);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
        }
        .btn-submit {
            width: 100%;
            padding: 12px 18px;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s ease;
        }
        .btn-submit:hover { background: #1e40af; }
        .result-box {
            background: #f1f5f9;
            border-radius: 10px;
            padding: 18px;
            margin-top: 16px;
            text-align: center;
        }
        .result-tier {
            font-size: 24px;
            font-weight: 800;
            margin: 8px 0;
        }
        .tier-murah { color: var(--success); }
        .tier-sedang { color: var(--warning); }
        .tier-mahal { color: var(--danger); }
        .prob-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            margin-top: 6px;
        }
        .prob-bar-container {
            flex-grow: 1;
            margin: 0 10px;
            background: #e2e8f0;
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
        }
        .prob-bar { height: 100%; background: var(--primary-light); border-radius: 4px; }
        .ref-info {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 14px;
            line-height: 1.5;
            background: #f8fafc;
            padding: 10px;
            border-radius: 6px;
            border: 1px dashed var(--border);
        }
        .api-doc {
            margin-top: 24px;
            padding: 18px;
            background: #0f172a;
            color: #e2e8f0;
            border-radius: 10px;
            font-family: monospace;
            font-size: 12px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RumahPoint AI — Price Tier Classifier</h1>
            <p>Sistem Inferensi Produksi Klasifikasi Harga Properti Multi-Kota Indonesia</p>
            <div>
                {% if model_ready %}
                    <span class="status-badge badge-ready">● Server & Model Keras Siap</span>
                {% else %}
                    <span class="status-badge badge-warning">▲ Model Belum Dimuat (Jalankan Colab)</span>
                {% endif %}
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Formulir Spesifikasi Properti</h2>
                <form id="predictionForm" method="POST" action="/predict">
                    <div class="form-group">
                        <label for="lokasi_bersih">Lokasi / Kota</label>
                        <select id="lokasi_bersih" name="lokasi_bersih" class="form-control" required>
                            {% for loc in locations %}
                                <option value="{{ loc }}" {% if form_data and form_data.lokasi_bersih == loc %}selected{% endif %}>{{ loc }}</option>
                            {% endfor %}
                            <option value="Lainnya" {% if form_data and form_data.lokasi_bersih == 'Lainnya' %}selected{% endif %}>Kota Lainnya (Di luar Top-15)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="luas_m2">Luas Tanah (m²)</label>
                        <input type="number" step="1" min="10" max="5000" id="luas_m2" name="luas_m2" class="form-control" required value="{{ form_data.luas_m2 if form_data else '120' }}" placeholder="Contoh: 120">
                    </div>

                    <div class="form-group">
                        <label for="luas_bangunan">Luas Bangunan (m²)</label>
                        <input type="number" step="1" min="0" max="5000" id="luas_bangunan" name="luas_bangunan" class="form-control" value="{{ form_data.luas_bangunan if form_data else '90' }}" placeholder="Contoh: 90 (Opsional)">
                    </div>

                    <div class="form-group">
                        <label for="jumlah_kamar">Jumlah Kamar Tidur</label>
                        <input type="number" step="1" min="0" max="30" id="jumlah_kamar" name="jumlah_kamar" class="form-control" required value="{{ form_data.jumlah_kamar if form_data else '3' }}" placeholder="Contoh: 3 (0 untuk Komersil/Ruko)">
                    </div>

                    <div class="form-group">
                        <label for="jumlah_kamar_mandi">Jumlah Kamar Mandi</label>
                        <input type="number" step="1" min="0" max="20" id="jumlah_kamar_mandi" name="jumlah_kamar_mandi" class="form-control" required value="{{ form_data.jumlah_kamar_mandi if form_data else '2' }}" placeholder="Contoh: 2">
                    </div>

                    <div class="form-group">
                        <label for="kondisi">Kondisi Bangunan</label>
                        <select id="kondisi" name="kondisi" class="form-control" required>
                            <option value="Baru" {% if form_data and form_data.kondisi == 'Baru' %}selected{% endif %}>Baru (≤ 3 tahun)</option>
                            <option value="Bekas" {% if form_data and form_data.kondisi == 'Bekas' %}selected{% elif not form_data %}selected{% endif %}>Bekas (> 3 tahun)</option>
                            <option value="Tidak diketahui" {% if form_data and form_data.kondisi == 'Tidak diketahui' %}selected{% endif %}>Tidak Diketahui</option>
                        </select>
                    </div>

                    <button type="submit" class="btn-submit" {% if not model_ready %}disabled style="opacity:0.6; cursor:not-allowed;"{% endif %}>
                        Prediksi Kelas Harga Properti
                    </button>
                </form>
            </div>

            <div class="card">
                <h2>Hasil Estimasi Model</h2>
                <div id="resultContainer">
                    {% if result %}
                        <div class="result-box">
                            <span style="font-size: 13px; color: var(--text-muted);">Kelas Harga Prediksi:</span>
                            <div class="result-tier {% if result.prediksi.kelas_harga == 'Murah' %}tier-murah{% elif result.prediksi.kelas_harga == 'Sedang' %}tier-sedang{% else %}tier-mahal{% endif %}">
                                {{ result.prediksi.kelas_harga }}
                            </div>
                            <p style="font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 12px;">
                                Tingkat Keyakinan: {{ result.prediksi.confidence }}%
                            </p>

                            <div style="text-align: left; margin-top: 16px;">
                                <div style="font-size: 12px; font-weight: 600; margin-bottom: 8px;">Distribusi Probabilitas Model:</div>
                                {% for cls_name, prob in result.prediksi.probabilitas.items() %}
                                    <div class="prob-item">
                                        <span style="width: 50px;">{{ cls_name }}</span>
                                        <div class="prob-bar-container">
                                            <div class="prob-bar" style="width: {{ (prob * 100)|round(1) }}%;"></div>
                                        </div>
                                        <span style="width: 45px; text-align: right; font-weight: 600;">{{ (prob * 100)|round(1) }}%</span>
                                    </div>
                                {% endfor %}
                            </div>
                        </div>
                    {% else %}
                        <div style="text-align: center; padding: 40px 10px; color: var(--text-muted);">
                            <p>Masukkan parameter properti di sebelah kiri lalu klik <strong>Prediksi Kelas Harga Properti</strong>.</p>
                        </div>
                    {% endif %}
                </div>

                <div class="ref-info">
                    <strong>Kriteria Batas Kuantil Kelas Harga:</strong><br>
                    • <strong>Murah:</strong> &lt; Rp 483,82 Juta (Kuantil 33,3%)<br>
                    • <strong>Sedang:</strong> Rp 483,82 Juta – Rp 1,235 Miliar (Kuantil 33,3%–66,7%)<br>
                    • <strong>Mahal:</strong> &gt; Rp 1,235 Miliar (Kuantil 66,7%)
                </div>
            </div>
        </div>

        <div class="api-doc">
            <strong>Endpoint REST API JSON:</strong><br>
            POST /predict<br>
            Content-Type: application/json<br><br>
            {<br>
            &nbsp;&nbsp;"luas_m2": 120,<br>
            &nbsp;&nbsp;"luas_bangunan": 90,<br>
            &nbsp;&nbsp;"jumlah_kamar": 3,<br>
            &nbsp;&nbsp;"jumlah_kamar_mandi": 2,<br>
            &nbsp;&nbsp;"lokasi_bersih": "Kota Jakarta Selatan",<br>
            &nbsp;&nbsp;"kondisi": "Bekas"<br>
            }
        </div>
    </div>

    <script>
        // Tangani form submit dengan AJAX JSON untuk UX responsif tanpa reload halaman
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const btn = this.querySelector('button[type="submit"]');
            btn.disabled = true;
            btn.innerText = 'Memproses Inferensi...';

            const payload = {
                lokasi_bersih: document.getElementById('lokasi_bersih').value,
                luas_m2: parseFloat(document.getElementById('luas_m2').value),
                luas_bangunan: document.getElementById('luas_bangunan').value ? parseFloat(document.getElementById('luas_bangunan').value) : null,
                jumlah_kamar: parseFloat(document.getElementById('jumlah_kamar').value),
                jumlah_kamar_mandi: parseFloat(document.getElementById('jumlah_kamar_mandi').value),
                kondisi: document.getElementById('kondisi').value
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const resData = await response.json();

                if (response.ok && resData.status === 'success') {
                    const p = resData.prediksi;
                    const tierClass = p.kelas_harga === 'Murah' ? 'tier-murah' : (p.kelas_harga === 'Sedang' ? 'tier-sedang' : 'tier-mahal');
                    
                    let probHtml = '';
                    for (const [cls, prob] of Object.entries(p.probabilitas)) {
                        const pct = (prob * 100).toFixed(1);
                        probHtml += `
                            <div class="prob-item">
                                <span style="width: 50px;">${cls}</span>
                                <div class="prob-bar-container">
                                    <div class="prob-bar" style="width: ${pct}%;"></div>
                                </div>
                                <span style="width: 45px; text-align: right; font-weight: 600;">${pct}%</span>
                            </div>
                        `;
                    }

                    document.getElementById('resultContainer').innerHTML = `
                        <div class="result-box">
                            <span style="font-size: 13px; color: var(--text-muted);">Kelas Harga Prediksi:</span>
                            <div class="result-tier ${tierClass}">${p.kelas_harga}</div>
                            <p style="font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 12px;">
                                Tingkat Keyakinan: ${p.confidence}%
                            </p>
                            <div style="text-align: left; margin-top: 16px;">
                                <div style="font-size: 12px; font-weight: 600; margin-bottom: 8px;">Distribusi Probabilitas Model:</div>
                                ${probHtml}
                            </div>
                        </div>
                    `;
                } else {
                    document.getElementById('resultContainer').innerHTML = `
                        <div class="result-box" style="background: #fee2e2; color: #991b1b;">
                            <strong>Gagal Memproses:</strong> ${resData.message || 'Terjadi kesalahan pada server.'}
                        </div>
                    `;
                }
            } catch (err) {
                document.getElementById('resultContainer').innerHTML = `
                    <div class="result-box" style="background: #fee2e2; color: #991b1b;">
                        <strong>Kesalahan Jaringan:</strong> ${err.message}
                    </div>
                `;
            } finally {
                btn.disabled = false;
                btn.innerText = 'Prediksi Kelas Harga Properti';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    """Endpoint antarmuka web interaktif RumahPoint."""
    locations = metadata.get('top15_locations', DEFAULT_TOP15) if metadata else DEFAULT_TOP15
    model_ready = (model is not None and preprocessor is not None and metadata is not None)
    return render_template_string(HTML_TEMPLATE, locations=locations, model_ready=model_ready, result=None, form_data=None)

@app.route('/health', methods=['GET'])
def health():
    """Endpoint Health Check server dan status ketersediaan artefak."""
    is_ready = (model is not None and preprocessor is not None and metadata is not None)
    status_code = 200 if is_ready else 503
    return jsonify({
        'status': 'ok' if is_ready else 'degraded',
        'service': 'RumahPoint Price Tier Prediction Service',
        'artifacts': {
            'model_loaded': model is not None,
            'preprocessor_loaded': preprocessor is not None,
            'metadata_loaded': metadata is not None
        },
        'versions': metadata.get('versions', {}) if metadata else {},
        'price_thresholds': metadata.get('price_thresholds', {}) if metadata else {}
    }), status_code

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint inferensi prediksi kelas harga properti via REST API JSON."""
    if model is None or preprocessor is None or metadata is None:
        return jsonify({
            'status': 'error',
            'message': 'Artefak model belum dimuat di server. Pastikan folder artifacts/ berisi model_rumahpoint.keras, preprocessor.joblib, dan metadata.json dari Colab.'
        }), 503

    try:
        # Deteksi format request (JSON atau Web Form)
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        if not data:
            return jsonify({'status': 'error', 'message': 'Payload request kosong atau format tidak valid.'}), 400

        # Validasi luas_m2
        try:
            luas_m2 = float(data.get('luas_m2', 0))
            if luas_m2 <= 0:
                return jsonify({'status': 'error', 'message': 'Field luas_m2 harus berupa angka lebih besar dari 0.'}), 400
        except (ValueError, TypeError):
            return jsonify({'status': 'error', 'message': 'Format angka luas_m2 tidak valid.'}), 400

        # Validasi luas_bangunan (opsional, imputasi median jika kosong)
        try:
            val_lb = data.get('luas_bangunan')
            luas_bangunan = float(val_lb) if val_lb not in [None, ''] else np.nan
        except (ValueError, TypeError):
            luas_bangunan = np.nan

        # Validasi jumlah kamar
        try:
            jumlah_kamar = float(data.get('jumlah_kamar', 0))
            if jumlah_kamar < 0:
                return jsonify({'status': 'error', 'message': 'Field jumlah_kamar tidak boleh bernilai negatif.'}), 400
        except (ValueError, TypeError):
            return jsonify({'status': 'error', 'message': 'Format angka jumlah_kamar tidak valid.'}), 400

        # Validasi jumlah kamar mandi
        try:
            jumlah_kamar_mandi = float(data.get('jumlah_kamar_mandi', 0))
            if jumlah_kamar_mandi < 0:
                return jsonify({'status': 'error', 'message': 'Field jumlah_kamar_mandi tidak boleh bernilai negatif.'}), 400
        except (ValueError, TypeError):
            return jsonify({'status': 'error', 'message': 'Format angka jumlah_kamar_mandi tidak valid.'}), 400

        # Normalisasi teks lokasi dan kondisi
        lokasi_raw = str(data.get('lokasi_bersih', 'Lainnya')).strip()
        kondisi = str(data.get('kondisi', 'Tidak diketahui')).strip()
        if kondisi not in ['Baru', 'Bekas', 'Tidak diketahui']:
            kondisi = 'Tidak diketahui'

        # Pemetaan Top-15 Lokasi Anti-Leakage
        top15 = metadata.get('top15_locations', DEFAULT_TOP15)
        lokasi_mapped = lokasi_raw if lokasi_raw in top15 else 'Lainnya'

        # Konstruksi DataFrame single-row persis seperti pipeline training
        df_input = pd.DataFrame([{
            'luas_m2': luas_m2,
            'luas_bangunan': luas_bangunan,
            'jumlah_kamar': jumlah_kamar,
            'jumlah_kamar_mandi': jumlah_kamar_mandi,
            'lokasi_bersih': lokasi_mapped,
            'kondisi': kondisi
        }])

        # Transformasi fitur via ColumnTransformer
        X_proc = preprocessor.transform(df_input).astype(np.float32)

        # Prediksi probabilitas kelas via Keras Model
        preds = model.predict(X_proc, verbose=0)
        probs = preds[0]
        class_idx = int(np.argmax(probs))
        label_classes = metadata.get('label_classes', ['Murah', 'Sedang', 'Mahal'])
        predicted_class = label_classes[class_idx]

        prob_dict = {
            cls_name: float(round(float(probs[i]), 4))
            for i, cls_name in enumerate(label_classes)
        }

        response_data = {
            'status': 'success',
            'prediksi': {
                'kelas_harga': predicted_class,
                'kategori_indeks': class_idx,
                'probabilitas': prob_dict,
                'confidence': float(round(float(probs[class_idx]) * 100, 2))
            },
            'ambang_batas_referensi': metadata.get('price_thresholds', {
                'Murah': '< Rp 483,82 Juta',
                'Sedang': 'Rp 483,82 Juta - Rp 1,235 Miliar',
                'Mahal': '> Rp 1,235 Miliar'
            }),
            'input_terverifikasi': {
                'luas_m2': luas_m2,
                'luas_bangunan': luas_bangunan if not np.isnan(luas_bangunan) else None,
                'jumlah_kamar': jumlah_kamar,
                'jumlah_kamar_mandi': jumlah_kamar_mandi,
                'lokasi_bersih': lokasi_raw,
                'lokasi_mapped': lokasi_mapped,
                'kondisi': kondisi
            }
        }

        # Jika request berasal dari form HTML standar tanpa fetch
        if not request.is_json and request.accept_mimetypes.accept_html and not request.headers.get('X-Requested-With'):
            return render_template_string(HTML_TEMPLATE, locations=top15, model_ready=True, result=response_data, form_data=data)

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Terjadi kesalahan internal saat inferensi: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Standar port 7860 untuk Hugging Face Spaces Docker
    app.run(host='0.0.0.0', port=7860, debug=False)

# -*- coding: utf-8 -*-
"""
共通ユーティリティ関数（ALS版 / クリーン版）
"""

from pathlib import Path
from typing import Optional, Tuple
import io

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, eye, diags

# =============================================================================
# 基本ヘルパー
# =============================================================================

def get_file_name(uploaded_file, file_name: Optional[str] = None) -> str:
    """アップロードされたファイルから安全にファイル名を取得"""
    if file_name:
        return file_name
    try:
        if hasattr(uploaded_file, 'name') and uploaded_file.name:
            return uploaded_file.name
        if hasattr(uploaded_file, 'filename') and uploaded_file.filename:
            return uploaded_file.filename
    except Exception:
        pass
    return "unknown_file"


def safe_seek(file_obj):
    """ファイルオブジェクトを安全に先頭へ"""
    try:
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
    except Exception:
        pass


def read_csv_file(uploaded_file, file_extension: str):
    """CSV/TSVを読み込む（UTF-8→Shift_JISフォールバック）"""
    sep = ',' if str(file_extension).lower() == "csv" else '\t'
    try:
        safe_seek(uploaded_file)
        return pd.read_csv(uploaded_file, sep=sep, header=0, index_col=None, on_bad_lines='skip')
    except UnicodeDecodeError:
        safe_seek(uploaded_file)
        try:
            return pd.read_csv(uploaded_file, sep=sep, encoding='shift_jis',
                               header=0, index_col=None, on_bad_lines='skip')
        except Exception:
            return None
    except Exception:
        return None


def detect_file_type(data: pd.DataFrame) -> str:
    """ファイル先頭列名から簡易タイプ判定"""
    try:
        first_column = str(data.columns[0])
        if first_column.startswith("# Laser Wavelength"):
            return "ramaneye_new"
        if first_column == "WaveNumber":
            return "ramaneye_old"
        if first_column == "Timestamp":
            return "ramaneye_old_old"
        if first_column == "Pixels":
            return "eagle"
        if first_column == "ENLIGHTEN Version" or "enlighten" in first_column.lower():
            return "wasatch"
        return "unknown"
    except Exception:
        return "unknown"


def find_index(rs_array, rs_focused):
    """rs_array から rs_focused に最も近いインデックス"""
    diff = [abs(float(element) - float(rs_focused)) for element in rs_array]
    return int(np.argmin(diff))


# =============================================================================
# スペクトル・前処理
# =============================================================================

def lift_min_to_one(y: np.ndarray) -> Tuple[np.ndarray, float]:
    """配列の最小値が1以上になるように常に非負へ持ち上げ"""
    y = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y_min = float(np.min(y))
    offset = 0.0 if y_min >= 1.0 else (1.0 - y_min)
    return y + offset, offset


def WhittakerSmooth(x, w, lambda_, differences=1):
    """Whittaker平滑化（重みw付き / 差分階数 differences）"""
    X = np.array(x, dtype=np.float64)
    m = X.size
    E = eye(m, format='csc')
    for _ in range(differences):
        E = E[1:] - E[:-1]
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * E.T * E))
    B = csc_matrix(W * X.T).toarray().flatten()
    background = spsolve(A, B)
    return np.array(background)


def asymmetric_least_squares(y, lam, p=0.01, niter=10, differences=2):
    """
    ALS（Asymmetric Least Squares）ベースライン推定
    - lam: 平滑化強度（λ）
    - p:   非対称重み（小さいほど上側点を弱く信頼）
    - niter: 反復回数
    - differences: 差分階数（通常は2）
    """
    y = np.asarray(y, dtype=np.float64)
    m = y.size
    w = np.ones(m, dtype=np.float64)
    lam = float(lam) if np.isfinite(lam) and lam > 0 else 1e4

    for _ in range(int(niter)):
        z = WhittakerSmooth(y, w, lam, differences)
        w = np.where(y > z, p, 1.0 - p).astype(np.float64)
    return z


def remove_outliers_and_interpolate(spectrum, window_size=10, threshold_factor=3):
    """スパイク（外れ値）検出→近傍補完"""
    spectrum = np.asarray(spectrum, dtype=float)
    n = len(spectrum)
    cleaned = spectrum.copy()

    for i in range(n):
        left = max(i - window_size, 0)
        right = min(i + window_size + 1, n)
        window = spectrum[left:right]
        median = np.median(window)
        std = np.std(window)
        if std == 0:
            continue
        if abs(spectrum[i] - median) > threshold_factor * std:
            if 0 < i < n - 1:
                cleaned[i] = (spectrum[i - 1] + spectrum[i + 1]) / 2.0
            elif i == 0 and n > 1:
                cleaned[i] = spectrum[i + 1]
            elif i == n - 1 and n > 1:
                cleaned[i] = spectrum[i - 1]
    return cleaned


# =============================================================================
# ファイル読込のバリエーション
# =============================================================================

def try_read_wasatch_file(uploaded_file, skiprows_list=None):
    """
    Wasatch (ENLIGHTEN) CSVの本体部を推定して読み込む。
    戻り値: (DataFrame{"Wavelength","Processed"}, used_skiprows) or (None, None)
    """
    if skiprows_list is None:
        skiprows_list = [44, 45, 46, 47, 48, 49, 50, 52]
    wavelength_candidates = ["Wavelength", "Wavelength (nm)", "wavelength", "wavelength (nm)"]

    for skiprows in skiprows_list:
        try:
            safe_seek(uploaded_file)
            try:
                df = pd.read_csv(uploaded_file, skiprows=skiprows, engine="python")
            except UnicodeDecodeError:
                safe_seek(uploaded_file)
                df = pd.read_csv(uploaded_file, skiprows=skiprows, encoding="shift-jis", engine="python")

            wl_col = None
            for c in df.columns:
                cn = str(c).strip()
                if cn in wavelength_candidates or "wavelength" in cn.lower():
                    wl_col = c
                    break
            if wl_col is None:
                continue

            processed_like = [c for c in df.columns if str(c).lower().startswith("processed")]
            if processed_like:
                spec_col = processed_like[-1]
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if str(c) != str(wl_col)]
                spec_col = numeric_cols[-1] if numeric_cols else None
            if wl_col is None or spec_col is None:
                continue

            wl_s = pd.to_numeric(df[wl_col], errors="coerce")
            sp_s = pd.to_numeric(df[spec_col], errors="coerce")
            if wl_s.notna().sum() >= 5 and sp_s.notna().sum() >= 5:
                out = pd.DataFrame({"Wavelength": wl_s, "Processed": sp_s})
                return out, skiprows
        except Exception:
            continue
    return None, None


# =============================================================================
# メイン：統合版 読み取り・前処理
# =============================================================================

def process_spectrum_file(
    uploaded_file,
    start_wavenum,
    end_wavenum,
    dssn_th,
    savgol_wsize,
    prelift_to_one: bool = True,
    file_name: Optional[str] = None,
):
    """
    スペクトルファイルを処理（切り出し・スパイク処理・ベースライン除去）
    戻り値: (wavenum, spectra, BSremoval_spectra_pos, Averemoval_spectra_pos, file_type, original_file_name)
    """
    original_file_name = get_file_name(uploaded_file, file_name)
    file_extension = original_file_name.split('.')[-1].lower() if '.' in original_file_name else ''

    if uploaded_file is None:
        return None, None, None, None, None, original_file_name

    # bytes / BytesIO -> StringIO に統一
    if isinstance(uploaded_file, bytes):
        try:
            uploaded_file = io.StringIO(uploaded_file.decode('utf-8'))
        except UnicodeDecodeError:
            try:
                uploaded_file = io.StringIO(uploaded_file.decode('shift-jis'))
            except UnicodeDecodeError:
                return None, None, None, None, None, original_file_name
    elif isinstance(uploaded_file, io.BytesIO):
        try:
            uploaded_file.seek(0)
            data_bytes = uploaded_file.read()
            try:
                uploaded_file = io.StringIO(data_bytes.decode('utf-8'))
            except UnicodeDecodeError:
                uploaded_file = io.StringIO(data_bytes.decode('shift-jis'))
        except Exception:
            return None, None, None, None, None, original_file_name

    # 一次読込
    data = read_csv_file(uploaded_file, file_extension)
    if data is None:
        return None, None, None, None, None, original_file_name

    # タイプ判定
    file_type = detect_file_type(data)
    safe_seek(uploaded_file)
    if file_type == "unknown":
        return None, None, None, None, None, original_file_name

    try:
        # ===== フォーマット別読み出し =====
        if file_type == "wasatch":
            data2, _used_skip = try_read_wasatch_file(uploaded_file)
            if data2 is None:
                return None, None, None, None, None, original_file_name

            lambda_ex = 785.0  # nm（必要に応じて拡張）
            pre_wavelength = pd.to_numeric(data2["Wavelength"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(pre_wavelength)
            if valid.sum() < 5:
                return None, None, None, None, None, original_file_name

            pre_wavelength = pre_wavelength[valid]
            wavenum_full = (1e7 / lambda_ex) - (1e7 / pre_wavelength)

            if "Processed" in data2.columns:
                pre_spectra_full = pd.to_numeric(data2["Processed"], errors="coerce").to_numpy(dtype=float)[valid]
            else:
                numeric_cols = data2.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c != "Wavelength"]
                if not numeric_cols:
                    return None, None, None, None, None, original_file_name
                pre_spectra_full = pd.to_numeric(data2[numeric_cols[-1]], errors="coerce").to_numpy(dtype=float)[valid]

            if wavenum_full[0] > wavenum_full[-1]:
                wavenum_full = wavenum_full[::-1]
                pre_spectra_full = pre_spectra_full[::-1]

            pre_wavenum = wavenum_full
            pre_spectra = pre_spectra_full

        elif file_type == "ramaneye_old_old":
            data_wo_ts = data.drop("Timestamp", axis=1) if "Timestamp" in data.columns else data
            df_t = data_wo_ts.set_index(data_wo_ts.columns[0]).T
            df_t.columns = ["intensity"]
            df_t.index = df_t.index.astype(float)
            df_t = df_t.sort_index()
            pre_wavenum = df_t.index.to_numpy(dtype=float)
            pre_spectra = df_t["intensity"].to_numpy(dtype=float)

        elif

# -*- coding: utf-8 -*-
"""
共通ユーティリティ関数（ALS版）
"""

from pathlib import Path
from typing import Optional, Tuple
import io

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import savgol_filter, find_peaks, peak_prominences
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
    """拡張子から区切りを推定して DataFrame に読み込み"""
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
    """先頭列名からファイルタイプを推定"""
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
        if first_column == "ENLIGHTEN Version":
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
    - p:   非対称重み（小さいほど上側点を弱く信頼）※固定値
    - niter: 反復回数 ※固定値
    - differences: 差分階数（通常は2）
    """
    y = np.asarray(y, dtype=np.float64)
    m = y.size
    w = np.ones(m, dtype=np.float64)
    lam = float(lam) if np.isfinite(lam) and lam > 0 else 1e4  # フォールバック

    for _ in range(int(niter)):
        z = WhittakerSmooth(y, w, lam, differences)
        # y > z ならピーク側 → 小さい重み p、y <= z なら背景側 → 大きい重み (1-p)
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


def find_peak_width(spectra, first_dev, peak_position, window_size=20):
    """ピーク近傍の開始/終了インデックス"""
    start_idx = max(peak_position - window_size, 0)
    end_idx   = min(peak_position + window_size, len(first_dev) - 1)
    local_start_idx = int(np.argmax(first_dev[start_idx:end_idx+1]) + start_idx)
    local_end_idx   = int(np.argmin(first_dev[start_idx:end_idx+1]) + start_idx)
    return local_start_idx, local_end_idx


def find_peak_area(spectra, local_start_idx, local_end_idx):
    """ピーク面積を台形則で計算"""
    return float(np.trapz(spectra[local_start_idx:local_end_idx+1], dx=1))


def calculate_peak_width(spectrum, peak_idx, wavenum):
    """半値幅（FWHM）を計算"""
    spectrum = np.asarray(spectrum, dtype=float)
    wavenum  = np.asarray(wavenum, dtype=float)
    if peak_idx <= 0 or peak_idx >= len(spectrum) - 1:
        return 0.0

    peak_intensity = spectrum[peak_idx]
    half_max = peak_intensity / 2.0

    left_idx = peak_idx
    while left_idx > 0 and spectrum[left_idx] > half_max:
        left_idx -= 1
    if left_idx < peak_idx and spectrum[left_idx] <= half_max < spectrum[left_idx + 1]:
        ratio = (half_max - spectrum[left_idx]) / (spectrum[left_idx + 1] - spectrum[left_idx] + 1e-12)
        left_w = wavenum[left_idx] + ratio * (wavenum[left_idx + 1] - wavenum[left_idx])
    else:
        left_w = wavenum[left_idx] if left_idx >= 0 else wavenum[0]

    right_idx = peak_idx
    while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_max:
        right_idx += 1
    if right_idx > peak_idx and spectrum[right_idx] <= half_max < spectrum[right_idx - 1]:
        ratio = (half_max - spectrum[right_idx]) / (spectrum[right_idx - 1] - spectrum[right_idx] + 1e-12)
        right_w = wavenum[right_idx] + ratio * (wavenum[right_idx - 1] - wavenum[right_idx])
    else:
        right_w = wavenum[right_idx] if right_idx < len(wavenum) else wavenum[-1]

    return float(abs(right_w - left_w))


# =============================================================================
# ファイル読込のバリエーション
# =============================================================================

def try_read_wasatch_file(uploaded_file, skiprows_list=None):
    """
    Wasatchファイルを複数の skiprows で試行読み込み
    """
    if skiprows_list is None:
        skiprows_list = [46, 47, 45, 44, 48, 49, 50]

    for skiprows in skiprows_list:
        try:
            safe_seek(uploaded_file)
            try:
                data = pd.read_csv(uploaded_file, skiprows=skiprows, nrows=None)
            except UnicodeDecodeError:
                safe_seek(uploaded_file)
                data = pd.read_csv(uploaded_file, encoding='shift-jis', skiprows=skiprows, nrows=None)

            if 'Wavelength' in data.columns and (
                'Processed' in data.columns or data.select_dtypes(include=[np.number]).shape[1] >= 2
            ):
                return data, skiprows
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

    備考:
    - dssn_th: ALSの平滑化強度（λ）として解釈（>0 を推奨）
    - p=0.01, niter=10 は内部固定
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
            lambda_ex = 785
            data2, _ = try_read_wasatch_file(uploaded_file)
            if data2 is None:
                return None, None, None, None, None, original_file_name

            pre_wavelength = np.array(data2["Wavelength"].values, dtype=float)
            pre_wavenum = (1e7 / float(lambda_ex)) - (1e7 / pre_wavelength)

            if "Processed" in data2.columns:
                pre_spectra = np.array(data2["Processed"].values, dtype=float)
            else:
                numeric_columns = data2.select_dtypes(include=[np.number]).columns.tolist()
                spectrum_columns = [col for col in numeric_columns if col != 'Wavelength']
                if not spectrum_columns:
                    return None, None, None, None, None, original_file_name
                pre_spectra = np.array(data2[spectrum_columns[-1]].values, dtype=float)

        elif file_type == "ramaneye_old_old":
            data_wo_ts = data.drop("Timestamp", axis=1) if "Timestamp" in data.columns else data
            df_t = data_wo_ts.set_index(data_wo_ts.columns[0]).T
            df_t.columns = ["intensity"]
            df_t.index = df_t.index.astype(float)
            df_t = df_t.sort_index()
            pre_wavenum = df_t.index.to_numpy(dtype=float)
            pre_spectra = df_t["intensity"].to_numpy(dtype=float)

        elif file_type == "ramaneye_old":
            pre_wavenum = np.array(data["WaveNumber"].values, dtype=float)
            pre_spectra = np.array(data.iloc[:, -1].values, dtype=float)

        elif file_type == "ramaneye_new":
            # ヘッダ行（WaveNumber を含む行）を動的探索し header=その行で再読込
            sep = ',' if file_extension == 'csv' else '\t'
            safe_seek(uploaded_file)
            text = uploaded_file.read()
            safe_seek(uploaded_file)
            lines = text.splitlines()

            header_idx = None
            for i, line in enumerate(lines[:200]):
                cols = line.split(sep)
                if any(c.strip() == "WaveNumber" for c in cols):
                    header_idx = i
                    break

            if header_idx is None:
                # フォールバック（既知の既定行）
                safe_seek(uploaded_file)
                data2 = pd.read_csv(uploaded_file, sep=sep, header=9, engine="python")
            else:
                safe_seek(uploaded_file)
                data2 = pd.read_csv(uploaded_file, sep=sep, header=header_idx, engine="python")

            if "WaveNumber" not in data2.columns:
                return None, None, None, None, None, original_file_name

            pre_wavenum = np.array(data2["WaveNumber"].values, dtype=float)
            pre_spectra = np.array(data2.iloc[:, -1].values, dtype=float)

        elif file_type == "eagle":
            data_t = data.transpose()
            header = data_t.iloc[:3]
            reversed_data = data_t.iloc[3:].iloc[::-1]
            data_t = pd.concat([header, reversed_data], ignore_index=True)
            pre_wavenum = np.array(data_t.iloc[3:, 0], dtype=float)
            pre_spectra = np.array(data_t.iloc[3:, 1], dtype=float)

        else:
            return None, None, None, None, None, original_file_name

        # 空チェック
        if pre_wavenum.size == 0 or pre_spectra.size == 0:
            return None, None, None, None, None, original_file_name

        # 降順なら反転して昇順へ
        if pre_wavenum.size > 1 and pre_wavenum[0] >= pre_wavenum[1]:
            pre_wavenum = pre_wavenum[::-1]
            pre_spectra = pre_spectra[::-1]

        # ===== 範囲切り出し =====
        start_index = find_index(pre_wavenum, start_wavenum)
        end_index   = find_index(pre_wavenum, end_wavenum)
        if end_index < start_index:
            start_index, end_index = end_index, start_index

        wavenum = np.array(pre_wavenum[start_index:end_index+1], dtype=float)
        spectra = np.array(pre_spectra[start_index:end_index+1], dtype=float)

        # ===== スパイク除去 → 移動平均（medfilt） =====
        spectra_spikerm = remove_outliers_and_interpolate(spectra)
        mveAve_spectra  = signal.medfilt(spectra_spikerm, savgol_wsize)

        # ===== ベースライン推定（ALS、必要なら事前に最小値を1へ） =====
        if prelift_to_one:
            mveAve_spectra, offset = lift_min_to_one(mveAve_spectra)
            spectra_spikerm = spectra_spikerm + offset

        # dssn_th を ALS の λ として利用（p=0.01, niter=10 は内部固定）
        baseline = asymmetric_least_squares(mveAve_spectra, lam=dssn_th*100000000000, p=0.1, niter=10, differences=2)

        # ===== ベースライン除去（負値を持ち上げた版も用意） =====
        BSremoval_spectra     = spectra_spikerm - baseline
        BSremoval_spectra_pos = BSremoval_spectra + abs(np.minimum(BSremoval_spectra, 0))

        Averemoval_spectra     = mveAve_spectra - baseline
        Averemoval_spectra_pos = Averemoval_spectra + abs(np.minimum(Averemoval_spectra, 0))

        return wavenum, spectra, BSremoval_spectra_pos, Averemoval_spectra_pos, file_type, original_file_name

    except Exception:
        return None, None, None, None, None, original_file_name


def process_spectrum_file_from_path(
    file_path: str,
    start_wavenum,
    end_wavenum,
    dssn_th,
    savgol_wsize,
    prelift_to_one: bool = True,
):
    """ファイルパス→bytes→共通処理に回すラッパー"""
    try:
        p = Path(file_path)
        with open(p, 'rb') as f:
            data = f.read()
        return process_spectrum_file(
            data, start_wavenum, end_wavenum, dssn_th, savgol_wsize,
            prelift_to_one=prelift_to_one, file_name=p.name
        )
    except Exception:
        return None, None, None, None, None, None

# -*- coding: utf-8 -*-
"""
ラマン分光：ピーク高さ比による検量線作成・分析モジュール
- 片方を信号ピーク（①）、もう片方を基準ピーク（②）として高さ比 H1/H2 を説明変数に一次回帰
- 既存の「面積」「PLS（多変量）」機能は削除
- 解析範囲は 2 区間（①と②）を指定・可視化
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io

# ページ設定：横幅いっぱいに表示
st.sidebar.markdown("### 設定")
st.set_page_config(layout="wide", initial_sidebar_state='expanded')

# 共通ユーティリティ関数
from common_utils import (
    detect_file_type, read_csv_file, find_index, WhittakerSmooth, 
    asymmetric_least_squares, remove_outliers_and_interpolate, process_spectrum_file
)

# 日本語フォント
plt.rcParams['font.family'] = 'DejaVu Sans'


class CalibrationAnalyzer:
    def __init__(self):
        self.spectra_data = []
        self.concentrations = []
        self.wavenumbers = None
        self.calibration_model = None  # ダミー（今回は未使用）
        self.calibration_type = None
        self.wave_ranges = None  # [[start1, end1], [start2, end2]]
        self.fitted_params = None
        self.slope_ratio = None
        self.intercept_ratio = None

    def process_spectra_files(self, uploaded_files, start_wavenum, end_wavenum, dssn_th, savgol_wsize):
        """複数のスペクトルファイルを処理（ベースライン補正と平滑を実施）"""
        self.spectra_data = []
        processed_files = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"ファイル処理中: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
            progress_bar.progress((i + 1) / len(uploaded_files))
            try:
                wavenum, raw_spectrum, corrected_spectrum, smoothed_spectrum, file_type, file_name = process_spectrum_file(
                    uploaded_file, start_wavenum, end_wavenum, dssn_th, savgol_wsize
                )
                if wavenum is not None:
                    self.spectra_data.append({
                        'filename': file_name,
                        'wavenumbers': wavenum,
                        'raw_spectrum': raw_spectrum,
                        'corrected_spectrum': smoothed_spectrum,
                        'file_type': file_type,
                    })
                    processed_files.append(file_name)
                else:
                    st.warning(f"ファイル {file_name} の処理に失敗しました")
            except Exception as e:
                st.error(f"ファイル処理エラー ({uploaded_file.name}): {str(e)}")
                continue

        progress_bar.empty()
        status_text.empty()

        if self.spectra_data:
            self.wavenumbers = self.spectra_data[0]['wavenumbers']
            st.success(f"{len(self.spectra_data)}個のファイルを正常に処理しました")
            return processed_files
        else:
            st.error("処理可能なファイルがありませんでした")
            return []

    # ---- 基本ユーティリティ ----
    def linear_baseline_correction(self, x, y):
        """指定範囲の左右端点群による一次ベースラインを推定し除去"""
        n_points = min(10, max(2, len(x) // 10))
        x_base = np.concatenate([x[:n_points], x[-n_points:]])
        y_base = np.concatenate([y[:n_points], y[-n_points:]])
        coeffs = np.polyfit(x_base, y_base, 1)
        baseline = np.polyval(coeffs, x)
        return y - baseline, baseline

    def calculate_peak_height(self, y):
        return float(np.max(y))

    # ---- 検量線（ピーク比） ----
    def create_peak_ratio_calibration(self, wave1_start, wave1_end, wave2_start, wave2_end):
        """ピーク高さ比 (H1/H2) による検量線作成"""
        h1_list, h2_list, ratio_list, fitting_results = [], [], [], []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, spectrum_data in enumerate(self.spectra_data):
            status_text.text(f"ピーク解析中: {spectrum_data['filename']} ({i+1}/{len(self.spectra_data)})")
            progress_bar.progress((i + 1) / len(self.spectra_data))

            wavenum = spectrum_data['wavenumbers']
            spectrum = spectrum_data['corrected_spectrum']

            # 範囲①
            s1 = find_index(wavenum, wave1_start)
            e1 = find_index(wavenum, wave1_end)
            x1 = wavenum[s1:e1 + 1]
            y1 = spectrum[s1:e1 + 1]
            y1_corr, b1 = self.linear_baseline_correction(x1, y1)
            h1 = self.calculate_peak_height(y1_corr)

            # 範囲②
            s2 = find_index(wavenum, wave2_start)
            e2 = find_index(wavenum, wave2_end)
            x2 = wavenum[s2:e2 + 1]
            y2 = spectrum[s2:e2 + 1]
            y2_corr, b2 = self.linear_baseline_correction(x2, y2)
            h2 = self.calculate_peak_height(y2_corr)

            h1_list.append(h1)
            h2_list.append(h2)
            ratio_list.append(h1 / h2 if h2 != 0 else np.nan)

            fitting_results.append({
                'filename': spectrum_data['filename'],
                'x1': x1, 'y1': y1, 'y1_corr': y1_corr, 'baseline1': b1, 'h1': h1,
                'x2': x2, 'y2': y2, 'y2_corr': y2_corr, 'baseline2': b2, 'h2': h2,
            })

        progress_bar.empty()
        status_text.empty()

        self.calibration_type = 'peak_ratio'
        self.wave_ranges = [[wave1_start, wave1_end], [wave2_start, wave2_end]]
        self.fitted_params = fitting_results

        return np.array(h1_list), np.array(h2_list), np.array(ratio_list), fitting_results

    def predict_concentration_ratio(self, new_spectrum_data, wave1_start, wave1_end, wave2_start, wave2_end,
                                    slope_ratio, intercept_ratio):
        """ピーク比から濃度予測"""
        wavenum = new_spectrum_data['wavenumbers']
        spectrum = new_spectrum_data['corrected_spectrum']

        # 範囲①
        s1 = find_index(wavenum, wave1_start)
        e1 = find_index(wavenum, wave1_end)
        x1 = wavenum[s1:e1 + 1]
        y1 = spectrum[s1:e1 + 1]
        y1_corr, b1 = self.linear_baseline_correction(x1, y1)
        h1 = self.calculate_peak_height(y1_corr)

        # 範囲②
        s2 = find_index(wavenum, wave2_start)
        e2 = find_index(wavenum, wave2_end)
        x2 = wavenum[s2:e2 + 1]
        y2 = spectrum[s2:e2 + 1]
        y2_corr, b2 = self.linear_baseline_correction(x2, y2)
        h2 = self.calculate_peak_height(y2_corr)

        ratio = h1 / h2 if h2 != 0 else np.nan
        if np.isnan(ratio):
            return None, h1, h2, ratio, x1, y1, y1_corr, b1, x2, y2, y2_corr, b2

        concentration = slope_ratio * ratio + intercept_ratio
        return float(concentration), h1, h2, ratio, x1, y1, y1_corr, b1, x2, y2, y2_corr, b2


# ---- UI 補助表示 ----
def display_calibration_equation(results):
    """検量線の数式を表示（ピーク比）"""
    st.subheader("検量線数式")
    if results['type'] == 'peak_ratio':
        m = results.get('slope_ratio')
        b = results.get('intercept_ratio')
        if m is not None and b is not None:
            eq = f"濃度 = {m:.6f} × (高さ①/高さ②) + {b:.6f}" if b >= 0 else \
                 f"濃度 = {m:.6f} × (高さ①/高さ②) - {abs(b):.6f}"
            st.markdown(f"- 比の式: **{eq}**")

# ---- タブ：検量線作成 ----
def calibration_creation_tab(analyzer: CalibrationAnalyzer):
    st.subheader("検量線作成（ピーク高さ比）")

    uploaded_files = st.file_uploader(
        "ラマンスペクトルをアップロードしてください（複数可）",
        type=['csv', 'txt'],
        accept_multiple_files=True,
        key="calibration_uploader",
    )

    if uploaded_files:
        # サイドバー：データ処理設定
        with st.sidebar:
            st.subheader("データ処理設定")
            start_wavenum = st.number_input("波数（開始）を入力してください:", value=400, min_value=0, max_value=4000)
            end_wavenum = st.number_input("波数（終了）を入力してください:", value=2000, min_value=start_wavenum + 1, max_value=4000)
            dssn_th = st.number_input("ベースラインパラメーターを入力してください:", value=1000, min_value=1, max_value=10000) / 1e7
            savgol_wsize = st.number_input("移動平均のウィンドウサイズを入力してください:", value=5, min_value=3, max_value=101, step=2)

        processed_files = analyzer.process_spectra_files(
            uploaded_files, start_wavenum, end_wavenum, dssn_th, savgol_wsize
        )

        if processed_files:
            # スペクトル表示（データ処理範囲） - Plotly
            st.subheader("スペクトル確認")
            fig_proc = go.Figure()
            for spectrum_data in analyzer.spectra_data:
                fig_proc.add_trace(go.Scatter(
                    x=spectrum_data['wavenumbers'],
                    y=spectrum_data['corrected_spectrum'],
                    mode='lines',
                    name=spectrum_data['filename']
                ))
            fig_proc.update_layout(
                xaxis_title='Raman Shift (cm⁻¹)',
                yaxis_title='Intensity (a.u.)',
                height=420
            )
            fig_proc.update_xaxes(range=[start_wavenum, end_wavenum])
            st.plotly_chart(fig_proc, use_container_width=True)

            # 濃度データ入力
            st.subheader("濃度データ入力")
            key_len = f"concentration_data_{len(processed_files)}"
            if key_len not in st.session_state:
                st.session_state[key_len] = pd.DataFrame({
                    'ファイル名': processed_files,
                    '濃度': [0.0] * len(processed_files),
                    '単位': ['mg/L'] * len(processed_files),
                })

            concentration_df = st.data_editor(
                st.session_state[key_len],
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "ファイル名": st.column_config.TextColumn(disabled=True),
                    # ここを修正
                    "濃度": st.column_config.NumberColumn(
                        "濃度",
                        help="各サンプルの濃度を入力してください",
                        min_value=0.0,
                        step=0.0001,
                        format="%.4f",
                    ),
                    "単位": st.column_config.TextColumn("単位", help="濃度の単位を入力してください"),
                },
                key=f"concentration_editor_{len(processed_files)}",
            )

            # 濃度データ確定
            col_btn, col_status = st.columns([1, 2])
            with col_btn:
                concentration_confirmed = st.button("濃度データ確定", type="secondary")
            with col_status:
                if concentration_confirmed:
                    st.session_state[key_len] = concentration_df
                    analyzer.concentrations = concentration_df['濃度'].values
                    st.success("濃度データを確定しました")
                    st.session_state.concentration_confirmed = True
                elif st.session_state.get('concentration_confirmed', False):
                    st.info("濃度データ確定済み")
                else:
                    st.warning("濃度データを入力して確定ボタンを押してください")

            # 検量線設定（2 つの解析範囲）
            if st.session_state.get('concentration_confirmed', False):
                analyzer.concentrations = st.session_state[key_len]['濃度'].values

                with st.sidebar:
                    st.subheader("検量線設定（ピーク比）")
                    # 波数範囲①（信号）
                    analysis1_start = st.number_input(
                        "解析開始波数①:",
                        value=1695,
                        min_value=int(analyzer.wavenumbers.min()) if analyzer.wavenumbers is not None else start_wavenum,
                        max_value=int(analyzer.wavenumbers.max()) if analyzer.wavenumbers is not None else end_wavenum,
                    )
                    _a1_end_default = 1730
                    _a1_end_default = max(_a1_end_default, analysis1_start)  # min_value 以上に調整
                    analysis1_end = st.number_input(
                        "解析終了波数①:",
                        value=_a1_end_default,
                        min_value=analysis1_start,
                        max_value=int(analyzer.wavenumbers.max()) if analyzer.wavenumbers is not None else end_wavenum,
                    )
                    # 波数範囲②（基準）
                    analysis2_start = st.number_input(
                        "解析開始波数②:",
                        value=1605,
                        min_value=int(analyzer.wavenumbers.min()) if analyzer.wavenumbers is not None else start_wavenum,
                        max_value=int(analyzer.wavenumbers.max()) if analyzer.wavenumbers is not None else end_wavenum,
                    )
                    _a2_end_default = 1625
                    _a2_end_default = max(_a2_end_default, analysis2_start)  # min_value 以上に調整
                    analysis2_end = st.number_input(
                        "解析終了波数②:",
                        value=_a2_end_default,
                        min_value=analysis2_start,
                        max_value=int(analyzer.wavenumbers.max()) if analyzer.wavenumbers is not None else end_wavenum,
                    )

                # 入力バリデーション（①開始 < ①終了、②開始 < ②終了 のみ）
                if analysis1_start >= analysis1_end:
                    st.error("解析範囲①は『開始 < 終了』にしてください。")
                    return
                if analysis2_start >= analysis2_end:
                    st.error("解析範囲②は『開始 < 終了』にしてください。")
                    return

                # 可視化：解析範囲（①と②）
                st.subheader("スペクトル確認（解析範囲①・②）")
                fig2 = go.Figure()
                for spectrum_data in analyzer.spectra_data:
                    fig2.add_trace(go.Scatter(
                        x=spectrum_data['wavenumbers'],
                        y=spectrum_data['corrected_spectrum'],
                        mode='lines',
                        name=spectrum_data['filename']
                    ))
                # ①の目印（赤）
                fig2.add_vline(x=analysis1_start, line_dash="dash", line_color="red", annotation_text=f"①開始: {analysis1_start} cm⁻¹")
                fig2.add_vline(x=analysis1_end, line_dash="dash", line_color="red", annotation_text=f"①終了: {analysis1_end} cm⁻¹")
                # ②の目印（青）
                fig2.add_vline(x=analysis2_start, line_dash="dash", line_color="blue", annotation_text=f"②開始: {analysis2_start} cm⁻¹")
                fig2.add_vline(x=analysis2_end, line_dash="dash", line_color="blue", annotation_text=f"②終了: {analysis2_end} cm⁻¹")

                fig2.update_layout(
                    xaxis_title='Raman Shift (cm⁻¹)',
                    yaxis_title='Intensity (a.u.)',
                    height=420
                )
                fig2.update_xaxes(range=[start_wavenum, end_wavenum])
                st.plotly_chart(fig2, use_container_width=True)

                # 検量線作成実行
                if st.button("検量線作成実行", type="primary"):
                    current_conc = st.session_state[key_len]['濃度'].values
                    analyzer.concentrations = current_conc
                    if len(set(analyzer.concentrations)) < 2:
                        st.error("少なくとも2つの異なる濃度が必要です")
                    else:
                        with st.spinner("検量線作成中..."):
                            h1, h2, ratios, fitting_results = analyzer.create_peak_ratio_calibration(
                                analysis1_start, analysis1_end, analysis2_start, analysis2_end
                            )
                            n = min(len(h1), len(h2), len(analyzer.concentrations))
                            h1 = h1[:n]
                            h2 = h2[:n]
                            ratios = ratios[:n]
                            conc_aligned = np.array(analyzer.concentrations)[:n]

                            valid = (~np.isnan(ratios)) & (h1 > 0) & (h2 > 0)
                            v_ratio = ratios[valid]
                            v_conc = conc_aligned[valid]

                            if len(v_ratio) >= 2:
                                slope_ratio, intercept_ratio = np.polyfit(v_ratio, v_conc, 1)
                                y_pred = slope_ratio * v_ratio + intercept_ratio
                                # 指標
                                ss_res = float(np.sum((v_conc - y_pred) ** 2))
                                ss_tot = float(np.sum((v_conc - np.mean(v_conc)) ** 2))
                                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
                                rmse = float(np.sqrt(np.mean((v_conc - y_pred) ** 2)))

                                st.session_state.calibration_results = {
                                    'type': 'peak_ratio',
                                    'heights1': h1,
                                    'heights2': h2,
                                    'ratios': ratios,
                                    'concentrations': conc_aligned,
                                    'slope_ratio': slope_ratio,
                                    'intercept_ratio': intercept_ratio,
                                    'r2': r2,
                                    'rmse': rmse,
                                    'fitting_results': fitting_results,
                                    'wave_ranges': [[analysis1_start, analysis1_end], [analysis2_start, analysis2_end]],
                                    'dssn_th': dssn_th,
                                    'savgol_wsize': savgol_wsize,
                                    'proc_range': [start_wavenum, end_wavenum],
                                    'analyzer': analyzer,
                                }
                            else:
                                st.error("有効データ点が不足しています（高さが正で、かつ②の高さ≠0が最低2点必要）。")

            # 結果表示
            if 'calibration_results' in st.session_state:
                results = st.session_state.calibration_results

                # 検量線数式
                display_calibration_equation(results)

                # 統計指標
                st.subheader("統計指標")
                if results['type'] == 'peak_ratio':
                    st.info(
                        f"選択方法: ピーク比（①/②） / 解析波数範囲①: {results['wave_ranges'][0][0]}–{results['wave_ranges'][0][1]} cm⁻¹ / "
                        f"解析波数範囲②: {results['wave_ranges'][1][0]}–{results['wave_ranges'][1][1]} cm⁻¹ / "
                        f"ベースライン: dssn_th={results['dssn_th']:.8f}, savgol_wsize={results['savgol_wsize']}"
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("R²", f"{results['r2']:.4f}")
                    with c2:
                        st.metric("RMSE", f"{results['rmse']:.4f}")

                    # 検量線プロット（比）
                    ratios = results['ratios']
                    concentrations = results['concentrations']
                    valid = (~np.isnan(ratios)) & (results['heights1'] > 0) & (results['heights2'] > 0)
                    vr, vc = ratios[valid], concentrations[valid]

                    fig_cal = go.Figure()
                    if len(vr) >= 1:
                        fig_cal.add_trace(go.Scatter(x=vr, y=vc, mode='markers', name='データ (比)', marker=dict(size=8)))
                    if len(vr) >= 2:
                        xl = np.linspace(vr.min(), vr.max(), 100)
                        yl = results['slope_ratio'] * xl + results['intercept_ratio']
                        fig_cal.add_trace(go.Scatter(x=xl, y=yl, mode='lines', name='回帰 (比)', line=dict(dash='dash')))
                    fig_cal.update_xaxes(title_text="高さ比 (①/②)")
                    fig_cal.update_yaxes(title_text="濃度")
                    fig_cal.update_layout(height=420)
                    st.plotly_chart(fig_cal, use_container_width=True)

                # エクスポート（作成タブ）
                st.subheader("結果エクスポート")
                if results['type'] == 'peak_ratio':
                    export_df = pd.DataFrame({
                        'ファイル名': [d['filename'] for d in analyzer.spectra_data],
                        '濃度': results['concentrations'],
                        '高さ①': results['heights1'],
                        '高さ②': results['heights2'],
                        '比(①/②)': results['ratios'],
                    })
                    w1s, w1e = results['wave_ranges'][0]
                    w2s, w2e = results['wave_ranges'][1]
                    csv_buffer = io.StringIO()
                    csv_buffer.write("# 検量線解析結果（ピーク比）\n")
                    csv_buffer.write(f"# 解析波数範囲①: {w1s}-{w1e} cm⁻¹\n")
                    csv_buffer.write(f"# 解析波数範囲②: {w2s}-{w2e} cm⁻¹\n")
                    csv_buffer.write(f"# ベースライン: dssn_th={results['dssn_th']:.8f}, savgol_wsize={results['savgol_wsize']}\n")
                    csv_buffer.write(f"# 比の式: y = {results['slope_ratio']:.6f}x + {results['intercept_ratio']:.6f}\n")
                    csv_buffer.write(f"# R2={results['r2']:.4f}, RMSE={results['rmse']:.4f}\n#\n")
                    export_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="結果をCSVでダウンロード",
                        data=csv_buffer.getvalue(),
                        file_name="calibration_results_peak_ratio.csv",
                        mime="text/csv",
                    )


# ---- タブ：スペクトル分析 ----
def spectrum_analysis_tab():
    """スペクトル分析タブ（作成時の方法・波数範囲・ベースラインで固定）"""
    st.subheader("スペクトル分析")

    if 'calibration_results' not in st.session_state:
        st.warning("まず検量線作成タブで検量線を作成してください。")
        return

    results = st.session_state.calibration_results
    analyzer: CalibrationAnalyzer = results['analyzer']

    # 固定情報
    if results['type'] == 'peak_ratio':
        w1s, w1e = results['wave_ranges'][0]
        w2s, w2e = results['wave_ranges'][1]
        st.info(
            f"使用中の検量線: ピーク比（①/②） / 解析波数範囲①: {w1s}–{w1e} cm⁻¹ / 解析波数範囲②: {w2s}–{w2e} cm⁻¹ / "
            f"ベースライン: dssn_th={results['dssn_th']:.8f}, savgol_wsize={results['savgol_wsize']}"
        )

    display_calibration_equation(results)

    # 新規スペクトル
    st.subheader("濃度算出用ファイル")
    uploaded_spectrum = st.file_uploader(
        "分析対象のラマンスペクトルをアップロードしてください",
        type=['csv', 'txt'],
        key="analysis_uploader",
    )

    if uploaded_spectrum:
        try:
            # 作成時に保存したベースライン設定を引き継ぎ、両範囲をカバーする処理範囲で読み込み
            w1s, w1e = results['wave_ranges'][0]
            w2s, w2e = results['wave_ranges'][1]
            wave_start = min(w1s, w2s)
            wave_end = max(w1e, w2e)
            baseline_dssn = results.get('dssn_th', 1000/1e7)
            baseline_win  = results.get('savgol_wsize', 5)

            wavenum, raw_spectrum, corrected_spectrum, smoothed_spectrum, file_type, file_name = process_spectrum_file(
                uploaded_spectrum, wave_start, wave_end, baseline_dssn, baseline_win
            )
            if wavenum is None:
                st.error("スペクトルファイルの処理に失敗しました。")
                return

            new_spectrum_data = {
                'filename': file_name,
                'wavenumbers': wavenum,
                'raw_spectrum': raw_spectrum,
                'corrected_spectrum': smoothed_spectrum,
            }

            # 表示（全体）
            st.subheader("分析スペクトル")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wavenum, y=smoothed_spectrum, mode='lines', name=file_name))
            fig.add_vline(x=w1s, line_dash="dash", line_color="red", annotation_text="①開始")
            fig.add_vline(x=w1e, line_dash="dash", line_color="red", annotation_text="①終了")
            fig.add_vline(x=w2s, line_dash="dash", line_color="blue", annotation_text="②開始")
            fig.add_vline(x=w2e, line_dash="dash", line_color="blue", annotation_text="②終了")
            fig.update_layout(xaxis_title='Raman Shift (cm⁻¹)', yaxis_title='Intensity (a.u.)', height=420)
            st.plotly_chart(fig, use_container_width=True)

            if results['type'] == 'peak_ratio':
                m = results['slope_ratio']
                b = results['intercept_ratio']
                (concentration, h1, h2, ratio,
                 x1, y1, y1_corr, b1,
                 x2, y2, y2_corr, b2) = analyzer.predict_concentration_ratio(
                    new_spectrum_data, w1s, w1e, w2s, w2e, m, b
                )

                if concentration is None or np.isnan(ratio):
                    st.error("比の計算に失敗しました（②の高さが0、または NaN）。解析範囲や基準ピークの選定を見直してください。")
                    return

                st.subheader("分析結果")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("予測濃度", f"{concentration:.4f}")
                with c2:
                    st.metric("高さ①", f"{h1:.4f}")
                with c3:
                    st.metric("高さ②", f"{h2:.4f}")
                with c4:
                    st.metric("比(①/②)", f"{ratio:.4f}")

                # ①のベースライン除去可視化
                fig_fit1 = go.Figure()
                fig_fit1.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='元データ①'))
                fig_fit1.add_trace(go.Scatter(x=x1, y=b1, mode='lines', name='一次ベースライン①', line=dict(dash='dot')))
                fig_fit1.add_trace(go.Scatter(x=x1, y=y1_corr, mode='lines', name='ベースライン除去後①'))
                fig_fit1.update_layout(title='解析範囲①のフィッティング', xaxis_title='Raman Shift (cm⁻¹)', yaxis_title='Intensity (a.u.)', height=380)
                st.plotly_chart(fig_fit1, use_container_width=True)

                # ②のベースライン除去可視化
                fig_fit2 = go.Figure()
                fig_fit2.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name='元データ②'))
                fig_fit2.add_trace(go.Scatter(x=x2, y=b2, mode='lines', name='一次ベースライン②', line=dict(dash='dot')))
                fig_fit2.add_trace(go.Scatter(x=x2, y=y2_corr, mode='lines', name='ベースライン除去後②'))
                fig_fit2.update_layout(title='解析範囲②のフィッティング', xaxis_title='Raman Shift (cm⁻¹)', yaxis_title='Intensity (a.u.)', height=380)
                st.plotly_chart(fig_fit2, use_container_width=True)

                # エクスポート（分析タブ）
                st.subheader("結果エクスポート")
                base_name = os.path.splitext(file_name)[0]
                df_out = pd.DataFrame({
                    'ファイル名': [base_name],
                    '予測濃度': [concentration],
                    '高さ①': [h1],
                    '高さ②': [h2],
                    '比(①/②)': [ratio],
                    '解析開始波数①': [w1s],
                    '解析終了波数①': [w1e],
                    '解析開始波数②': [w2s],
                    '解析終了波数②': [w2e],
                    'dssn_th': [baseline_dssn],
                    'savgol_wsize': [baseline_win],
                })
                csv_buffer = io.StringIO()
                csv_buffer.write("# スペクトル分析結果（ピーク比）\n")
                csv_buffer.write(f"# 解析波数範囲①: {w1s}-{w1e} cm⁻¹\n")
                csv_buffer.write(f"# 解析波数範囲②: {w2s}-{w2e} cm⁻¹\n")
                csv_buffer.write(f"# ベースライン: dssn_th={baseline_dssn:.8f}, savgol_wsize={baseline_win}\n")
                csv_buffer.write(f"# 比の式: y = {m:.6f}x + {b:.6f}\n#\n")
                df_out.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="分析結果をCSVでダウンロード",
                    data=csv_buffer.getvalue(),
                    file_name=f"analysis_result_{base_name}.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"ファイル処理エラー: {str(e)}")


# ---- 時系列表示タブ ----
def time_series_tab():
    """時系列データ表示タブ（サイドバー=共通設定、ここでは変更不可）"""
    st.subheader("時系列表示")

    if 'calibration_results' not in st.session_state:
        st.warning("まず検量線作成タブで検量線を作成してください。")
        return

    results = st.session_state.calibration_results
    analyzer: CalibrationAnalyzer = results['analyzer']

    # 共通設定（このタブでは変更不可）
    proc = results.get('proc_range', None)
    if not proc or len(proc) != 2:
        st.warning("処理範囲情報が見つかりません。いったん検量線作成を実行してください。")
        return
    proc_start, proc_end = proc
    dssn_th = results.get('dssn_th')
    savgol_wsize = results.get('savgol_wsize')
    (w1s, w1e), (w2s, w2e) = results['wave_ranges']

    st.info(
        f"表示範囲: {proc_start}–{proc_end} cm⁻¹ / "
        f"検量線範囲①: {w1s}–{w1e} cm⁻¹, 範囲②: {w2s}–{w2e} cm⁻¹ / "
        f"ベースライン: dssn_th={dssn_th:.8f}, savgol_wsize={savgol_wsize}"
    )

    uploaded_files = st.file_uploader(
        "時系列用のラマンスペクトルをアップロード（複数可）",
        type=['csv', 'txt'],
        accept_multiple_files=True,
        key="timeseries_uploader",
    )

    if not uploaded_files:
        return

    def parse_timeseries(uploaded_file):
        file_name = uploaded_file.name
        ext = file_name.split('.')[-1] if '.' in file_name else ''
        data = read_csv_file(uploaded_file, ext)
        file_type = detect_file_type(data)
        uploaded_file.seek(0)

        if file_type == "wasatch":
            try:
                df = pd.read_csv(uploaded_file, encoding='shift-jis', skiprows=46)
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, skiprows=46)
            lambda_ex = 785
            pre_wavelength = np.array(df["Wavelength"].values)
            wavenum_full = (1e7 / lambda_ex) - (1e7 / pre_wavelength)
            processed_cols = [c for c in df.columns if str(c).startswith("Processed")]
            if len(processed_cols) == 0:
                return None
            spectra_full = df[processed_cols].to_numpy()
            labels = processed_cols
        elif file_type in ("ramaneye_old", "ramaneye_new"):
            if file_type == "ramaneye_new":
                df = pd.read_csv(uploaded_file, skiprows=9)
            else:
                df = data
            wavenum_full = np.array(df["WaveNumber"].values)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cols = [c for c in num_cols if c != "WaveNumber"]
            if len(cols) == 0:
                cols = [df.columns[-1]]
            spectra_full = df[cols].to_numpy()
            labels = [str(c) for c in cols]
        else:
            return None

        # 昇順に統一
        if wavenum_full[0] > wavenum_full[-1]:
            wavenum_full = wavenum_full[::-1]
            spectra_full = spectra_full[::-1, :]

        # 処理範囲へ切り出し
        s = find_index(wavenum_full, proc_start)
        e = find_index(wavenum_full, proc_end)
        wn = np.array(wavenum_full[s:e+1])
        mat = np.array(spectra_full[s:e+1, :])  # 形状: (N_wavenum, N_time)
        return file_name, wn, mat, labels

    # 各ファイルごとに可視化
    for up in uploaded_files:
        
        parsed = parse_timeseries(up)
        if parsed is None:
            st.error(f"{up.name} を時系列として解釈できませんでした。対応形式（Wasatch/RamanEye）をご利用ください。")
            continue
        file_name, wn, mat, labels = parsed
        base_name = os.path.splitext(file_name)[0]

        # スペクトル時系列（Heatmap）
        st.subheader(f"時系列スペクトル: {base_name}")
        fig_hm = go.Figure(data=go.Heatmap(z=mat.T, x=wn, y=list(range(mat.shape[1])), colorbar=dict(title='Intensity')))
        fig_hm.update_layout(xaxis_title='Raman Shift (cm⁻¹)', yaxis_title='Index (time)')
        st.plotly_chart(fig_hm, use_container_width=True)

        # 範囲①/②のピーク高さを時系列で算出
        h1_series, h2_series = [], []
        s1 = find_index(wn, w1s)
        e1 = find_index(wn, w1e)
        s2 = find_index(wn, w2s)
        e2 = find_index(wn, w2e)
        x1 = wn[s1:e1+1]
        x2 = wn[s2:e2+1]
        for j in range(mat.shape[1]):
            y1 = mat[s1:e1+1, j]
            y2 = mat[s2:e2+1, j]
            y1_corr, _ = analyzer.linear_baseline_correction(x1, y1)
            y2_corr, _ = analyzer.linear_baseline_correction(x2, y2)
            h1 = analyzer.calculate_peak_height(y1_corr)
            h2 = analyzer.calculate_peak_height(y2_corr)
            h1_series.append(h1)
            h2_series.append(h2)

        # 時系列プロット（①・②）
        st.subheader("範囲①・②の強度（時系列）")
        idx = list(range(len(h1_series)))
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=idx, y=h1_series, mode='lines+markers', name='高さ①'))
        fig_ts.add_trace(go.Scatter(x=idx, y=h2_series, mode='lines+markers', name='高さ②'))
        fig_ts.update_layout(xaxis_title='Index (time)', yaxis_title='Peak height (a.u.)', height=420)
        st.plotly_chart(fig_ts, use_container_width=True)
        df_h = pd.DataFrame({
            'Index': idx,
            '高さ①': h1_series,
            '高さ②': h2_series,
        })
        csv_h = io.StringIO()
        df_h.to_csv(csv_h, index=False)
        st.download_button(
            label="CSVダウンロード（①・②の強度）",
            data=csv_h.getvalue(),
            file_name=f"{base_name}_heights.csv",
            mime="text/csv",
        )
        
        # ピーク比（①/②）の時系列
        st.subheader("ピーク比 (①/②) の時系列")
        ratio_series = [ (h1_series[i] / h2_series[i]) if h2_series[i] != 0 else np.nan for i in range(len(idx)) ]
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(x=idx, y=ratio_series, mode='lines+markers', name='比(①/②)'))
        fig_ratio.update_layout(xaxis_title='Index (time)', yaxis_title='Ratio', height=380)
        st.plotly_chart(fig_ratio, use_container_width=True)
        df_r = pd.DataFrame({
            'Index': idx,
            '比(①/②)': ratio_series,
        })
        csv_r = io.StringIO()
        df_r.to_csv(csv_r, index=False)
        st.download_button(
            label="CSVダウンロード（ピーク比）",
            data=csv_r.getvalue(),
            file_name=f"{base_name}_ratio.csv",
            mime="text/csv",
        )
        
        # 濃度換算の時系列（検量線: C = m * (H1/H2) + b）
        st.subheader("濃度換算の時系列")
        m = results.get('slope_ratio')
        b = results.get('intercept_ratio')
        if m is None or b is None:
            st.warning("検量線の係数が見つからないため、濃度換算を表示できません。")
        else:
            conc_series = [ (m * r + b) if not np.isnan(r) else np.nan for r in ratio_series ]
            fig_conc = go.Figure()
            fig_conc.add_trace(go.Scatter(x=idx, y=conc_series, mode='lines+markers', name='濃度換算'))
            fig_conc.update_layout(xaxis_title='Index (time)', yaxis_title='Concentration (estimated)', height=380)
            st.plotly_chart(fig_conc, use_container_width=True)
            
        df_c = pd.DataFrame({
            'Index': idx,
            '濃度(推定)': conc_series,
        })
        csv_c = io.StringIO()
        df_c.to_csv(csv_c, index=False)
        st.download_button(
            label="CSVダウンロード（濃度換算）",
            data=csv_c.getvalue(),
            file_name=f"{base_name}_concentration.csv",
            mime="text/csv",
        )

# ---- 画面構成 ----
def calibration_mode():
    """検量線作成モード（タブ版）"""
    st.header("検量線作成・分析システム（ピーク高さ比）")

    tab1, tab2, tab3 = st.tabs(["検量線作成", "スペクトル分析", "時系列表示"])

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = CalibrationAnalyzer()

    with tab1:
        calibration_creation_tab(st.session_state.analyzer)
    with tab2:
        spectrum_analysis_tab()
    with tab3:
        time_series_tab()


if __name__ == "__main__":
    calibration_mode()

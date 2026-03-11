import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Fatigue Rate Finder")
st.subheader("Based off Jim's model")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    xl = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select a sheet", xl.sheet_names)

    df = pd.read_excel(xl, sheet_name=sheet_name, usecols=['Time', 'Cadence', 'Power'])
    df["Torque"] = df["Power"] / (df["Cadence"] *  np.pi / 30)

    first_15 = df[df['Time'] <= 15]
    if len(first_15) > 1:
        biggest_gap_idx = first_15['Power'].diff().abs().idxmax()
        default_sit_time = float(first_15.loc[biggest_gap_idx, 'Time'])
    else:
        default_sit_time = 0.1

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        standing_Tmax = st.number_input("Standing Tmax", value=250, step=1)
    with c2:
        standing_Cmax = st.number_input("Standing Cmax", value=250, step=1)
    with c3:
        seated_Tmax = st.number_input("Seated Tmax", value=250, step=1)
    with c4:
        seated_Cmax = st.number_input("Seated Cmax", value=250, step=1)
    with c5:
        time_at_sit = st.number_input("Time at sit", value=default_sit_time, step=0.01)

    ot1, ot2, ot3 = st.columns(3)
    with ot1:
        opt_start_time = st.number_input("Optimise from time (s)", value=0.0, step=0.1, min_value=0.0)
    with ot2:
        opt_end_time = st.number_input("Optimise to time (s)", value=float(df["Time"].max()), step=0.1, min_value=0.0)
    with ot3:
        opt_range = st.number_input("Optimise range (±)", value=10, step=1, min_value=1)

    def _model_power(st_Tmax, st_Cmax, se_Tmax, se_Cmax):
        if st_Cmax <= 0 or se_Cmax <= 0:
            return None, None
        st_slope = st_Tmax / st_Cmax
        se_slope = se_Tmax / se_Cmax
        tc_slope = np.where(df["Time"] < time_at_sit, st_slope, se_slope)
        Tmax_calc = df["Torque"].values + tc_slope * df["Cadence"].values
        standing_mask = (df["Time"] < time_at_sit).values
        time_vals = df["Time"].values
        with np.errstate(invalid='ignore', divide='ignore'):
            st_time, st_tmax = time_vals[standing_mask], Tmax_calc[standing_mask]
            st_valid = np.isfinite(st_tmax)
            st_dT = np.polyfit(st_time[st_valid], st_tmax[st_valid], 1)[0] if st_valid.sum() >= 2 else 0.0
            se_time, se_tmax = time_vals[~standing_mask], Tmax_calc[~standing_mask]
            se_valid = np.isfinite(se_tmax)
            se_dT = np.polyfit(se_time[se_valid], se_tmax[se_valid], 1)[0] if se_valid.sum() >= 2 else 0.0
        Tmax_at_sit_val = Tmax_calc[standing_mask][-1] if standing_mask.any() else st_Tmax
        Model_Tmax = np.where(
            df["Time"] < time_at_sit,
            st_Tmax + df["Time"].values * st_dT,
            Tmax_at_sit_val + df["Time"].values * se_dT
        )
        Model_Power = (Model_Tmax - tc_slope * df["Cadence"].values) * df["Cadence"].values * np.pi / 30
        return Model_Power, tc_slope

    def compute_standing_sse(params):
        st_Tmax, st_Cmax = params
        if st_Cmax <= 0:
            return 1e18
        Model_Power, _ = _model_power(st_Tmax, st_Cmax, seated_Tmax, seated_Cmax)
        if Model_Power is None:
            return 1e18
        mask = (df["Time"].values >= opt_start_time) & (df["Time"].values <= opt_end_time) & (df["Time"].values < time_at_sit)
        sse = np.nansum((df["Power"].values[mask] - Model_Power[mask]) ** 2)
        return float(sse) if np.isfinite(sse) else 1e18

    def compute_seated_sse(params):
        se_Tmax, se_Cmax = params
        if se_Cmax <= 0:
            return 1e18
        Model_Power, _ = _model_power(standing_Tmax, standing_Cmax, se_Tmax, se_Cmax)
        if Model_Power is None:
            return 1e18
        mask = (df["Time"].values >= opt_start_time) & (df["Time"].values <= opt_end_time) & (df["Time"].values >= time_at_sit)
        sse = np.nansum((df["Power"].values[mask] - Model_Power[mask]) ** 2)
        return float(sse) if np.isfinite(sse) else 1e18

    btn_col, toggle_col = st.columns([1, 2])
    with btn_col:
        if st.button("Optimise"):
            st_bounds = [(standing_Tmax - opt_range, standing_Tmax + opt_range),
                         (standing_Cmax - opt_range, standing_Cmax + opt_range)]
            se_bounds = [(seated_Tmax - opt_range, seated_Tmax + opt_range),
                         (seated_Cmax - opt_range, seated_Cmax + opt_range)]
            st_result = minimize(
                compute_standing_sse,
                x0=[standing_Tmax, standing_Cmax],
                method='Nelder-Mead',
                bounds=st_bounds,
                options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 10000}
            )
            opt_st_Tmax_r, opt_st_Cmax_r = st_result.x

            def _compute_seated_sse_opt(params):
                se_Tmax, se_Cmax = params
                if se_Cmax <= 0:
                    return 1e18
                Model_Power, _ = _model_power(opt_st_Tmax_r, opt_st_Cmax_r, se_Tmax, se_Cmax)
                if Model_Power is None:
                    return 1e18
                mask = (df["Time"].values >= opt_start_time) & (df["Time"].values <= opt_end_time) & (df["Time"].values >= time_at_sit)
                sse = np.nansum((df["Power"].values[mask] - Model_Power[mask]) ** 2)
                return float(sse) if np.isfinite(sse) else 1e18

            se_result = minimize(
                _compute_seated_sse_opt,
                x0=[seated_Tmax, seated_Cmax],
                method='Nelder-Mead',
                bounds=se_bounds,
                options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 10000}
            )
            st.session_state["opt_params"] = [st_result.x[0], st_result.x[1], se_result.x[0], se_result.x[1]]
    with toggle_col:
        use_optimised = st.toggle(
            "Use optimised inputs",
            value=False,
            disabled="opt_params" not in st.session_state
        )

    if "opt_params" in st.session_state:
        opt_st_Tmax, opt_st_Cmax, opt_se_Tmax, opt_se_Cmax = st.session_state["opt_params"]
        oc1, oc2, oc3, oc4 = st.columns(4)
        oc1.metric("Standing Tmax", f"{opt_st_Tmax:.1f}", delta=f"{opt_st_Tmax - standing_Tmax:.1f}")
        oc2.metric("Standing Cmax", f"{opt_st_Cmax:.1f}", delta=f"{opt_st_Cmax - standing_Cmax:.1f}")
        oc3.metric("Seated Tmax", f"{opt_se_Tmax:.1f}", delta=f"{opt_se_Tmax - seated_Tmax:.1f}")
        oc4.metric("Seated Cmax", f"{opt_se_Cmax:.1f}", delta=f"{opt_se_Cmax - seated_Cmax:.1f}")
    else:
        use_optimised = False
        opt_st_Tmax, opt_st_Cmax, opt_se_Tmax, opt_se_Cmax = standing_Tmax, standing_Cmax, seated_Tmax, seated_Cmax

    if not use_optimised:
        opt_st_Tmax, opt_st_Cmax, opt_se_Tmax, opt_se_Cmax = standing_Tmax, standing_Cmax, seated_Tmax, seated_Cmax

    pmax = opt_st_Tmax * opt_st_Cmax * 0.25 * np.pi / 30
    opt_st_slope = opt_st_Tmax / opt_st_Cmax
    opt_se_slope = opt_se_Tmax / opt_se_Cmax
  
    df["tc_slope"] = np.where(df["Time"] < time_at_sit, opt_st_slope, opt_se_slope)
    df["Tmax_calc"] = df["Torque"] + df["tc_slope"] * df["Cadence"]

    standing_mask = df["Time"] < time_at_sit
    seated_mask = ~standing_mask
    with np.errstate(invalid='ignore', divide='ignore'):
        st_fit = df.loc[standing_mask, ["Time", "Tmax_calc"]].dropna()
        standing_dtMax_dt = np.polyfit(st_fit["Time"], st_fit["Tmax_calc"], 1)[0] if len(st_fit) >= 2 else 0.0
        se_fit = df.loc[seated_mask, ["Time", "Tmax_calc"]].dropna()
        seated_dtMax_dt = np.polyfit(se_fit["Time"], se_fit["Tmax_calc"], 1)[0] if len(se_fit) >= 2 else 0.0
    Tmax_at_sit = df.loc[standing_mask, "Tmax_calc"].iloc[-1] if standing_mask.any() else opt_st_Tmax
  
    df["Model_Tmax"] = np.where(
        df["Time"] < time_at_sit,
        opt_st_Tmax + df["Time"] * standing_dtMax_dt,
        Tmax_at_sit + df["Time"] * seated_dtMax_dt
    )
    df["Model_Power"] = (df["Model_Tmax"] - df["tc_slope"] * df["Cadence"]) * df["Cadence"] * np.pi / 30
    df["SSE"] = (df["Power"] - df["Model_Power"]) ** 2
    opt_time_mask = (df["Time"] >= opt_start_time) & (df["Time"] <= opt_end_time)
    standing_sse = df.loc[opt_time_mask & (df["Time"] < time_at_sit), "SSE"].sum()
    seated_sse = df.loc[opt_time_mask & (df["Time"] >= time_at_sit), "SSE"].sum()
    sse = standing_sse + seated_sse

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Standing Pmax", f"{pmax:.2f} W")
    m2.metric("Standing dT/dt", f"{standing_dtMax_dt:.4f} Nm/s")
    m3.metric("Standing TC slope", f"{opt_st_slope:.4f} Nm/s")
    m4.metric("Seated Pmax", f"{opt_se_Tmax * opt_se_Cmax * 0.25 * np.pi / 30:.2f} W")
    m5.metric("Seated dT/dt", f"{seated_dtMax_dt:.4f} Nm/s")
    m6.metric("Seated TC slope", f"{opt_se_slope:.4f} Nm/s")
    m7, m8, m9 = st.columns(3)
    m7.metric("Total SSE", f"{sse:.2f}")
    m8.metric("Standing SSE", f"{standing_sse:.2f}")
    m9.metric("Seated SSE", f"{seated_sse:.2f}")

    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Time"], y=df["Power"], mode="lines", name="Power"))
    fig.add_trace(go.Scatter(x=df["Time"], y=df["Model_Power"], mode="lines", name="Model Power"))
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Power (W)")
    st.plotly_chart(fig, width='stretch')

    st.dataframe(df)
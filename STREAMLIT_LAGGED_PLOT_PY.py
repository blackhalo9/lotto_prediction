import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide", page_title="Enhanced Lagged Plot Dashboard")
st.title("Raw Material Prices and Variables")
PASSWORD = "solution"

if 'password_verified' not in st.session_state:
    st.session_state['password_verified'] = False

def check_password():
    """Validate password."""
    if st.session_state.password == PASSWORD:
        st.session_state.password_verified = True
    else:
        st.error("Incorrect password, please try again. (if password needed, contact to eunsuk.ko@lgensol.com)")

if not st.session_state.password_verified:
    st.text_input("Enter the password:", type="password", on_change=check_password, key="password")
else:
    FIXED_TARGETS = ['PRICE_LI2CO3_EXW_FAST_KG', 
                     'PRICE_LI2CO3_CIF_FAST_KG_SPOT_EXCHNG', 
                     'PRICE_LIOH_EXW_FAST_KG', 
                     'PRICE_LIOH_CIF_FAST_KG_SPOT_EXCHNG',
                     'PRICE_AG_HIGH_ICC_RMB',
                     'PRICE_COKE_GNR_GRN_ICC_RMB',
                     'PRICE_SV_LIPF6_ICC_RMB']

    def load_and_scale_data(file_path=None, uploaded_file=None):
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, parse_dates=[0])
        elif file_path is not None:
            df = pd.read_csv(file_path, parse_dates=[0])
        else:
            return None

        df.columns = ['date'] + list(df.columns)[1:]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df.iloc[:, 1:])
        df_scaled = pd.DataFrame(scaled_features, columns=df.columns[1:])
        df_scaled['date'] = df['date']
        df_scaled = df_scaled[['date'] + [col for col in df_scaled.columns if col != 'date']]
        return df_scaled

    def plot_lagged_data(df, selected_targets, variable_lags):
        fig = go.Figure()

        for target in selected_targets:
            fig.add_trace(go.Scatter(x=df['date'][:-128], y=df[target], mode='lines', name=target, line=dict(width=4)))

        for variable, lag in variable_lags.items():
            lagged_column_name = f'lagged_{variable}'
            df[lagged_column_name] = df[variable].shift(lag)
            fig.add_trace(go.Scatter(x=df['date'][:-128], y=df[lagged_column_name], mode='lines', name=f'{variable} (lagged {lag})'))

        fig.update_layout(
            title='Lagged Plot with Optional Target Variables',
            xaxis=dict(
                title='Date',
                tickmode='linear',
                dtick="M12",  # Yearly intervals
                rangeselector=dict(
                    buttons=list([
                        dict(count=2, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="2Y", step="year", stepmode="backward"),
                        dict(count=4, label="3Y", step="year", stepmode="backward"),
                        dict(count=5, label="4Y", step="year", stepmode="backward"),
                        dict(count=6, label="5Y", step="year", stepmode="backward"),
                        dict(step="all",label="reset")
                    ])
                ),
                type='date'
            ),
            yaxis_title='Scaled Value',
            autosize=False,
            width=1800,
            height=600
        )
        st.plotly_chart(fig)

    default_data_file = '/home/eunsuk.ko/METAL_FCST/DATA/data202402/INTEGR_METAL_NON_METAL_MONTHLY_2024-02.csv'
    uploaded_file = st.file_uploader("(Optional) choose a CSV file (default = INTEGR_METAL_NON_METAL_MONTHLY_2024-04.csv)", type="csv")
    df = load_and_scale_data(file_path=default_data_file, uploaded_file=uploaded_file)

    if df is not None:
        pass 

    selected_targets = st.multiselect("STEP 1) Select target 'y' to display (multiple selection available)", options=FIXED_TARGETS, default=FIXED_TARGETS[:1])
    comparison_variables = [col for col in df.columns if col not in FIXED_TARGETS + ['date']] if df is not None else []
    selected_variables = st.multiselect("STEP 2) Select variables to compare with the targets (multiple selection available)", options=comparison_variables, default=None)

    variable_lags = {}
    for variable in selected_variables:
        if variable not in st.session_state:
            st.session_state[variable] = 1
        lag = st.slider(f"STEP 3) Select the monthly lag for {variable}", min_value=1, max_value=12, value=st.session_state[variable], key=variable)
        variable_lags[variable] = lag

    if selected_targets and selected_variables:
        plot_lagged_data(df, selected_targets, variable_lags)
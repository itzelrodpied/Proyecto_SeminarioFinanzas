import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import scipy.optimize as op

# =============================
# Configuración interactiva
# =============================
st.title("Simulador de Estrategias de Inversión con Optimización")

# Parámetros configurables en la barra lateral
st.sidebar.header("Configuración")
start_date = st.sidebar.date_input("Fecha de inicio", value=pd.to_datetime("2000-01-01"))
end_date = st.sidebar.date_input("Fecha de fin", value=pd.to_datetime("2025-12-01"))
rf = st.sidebar.number_input("Tasa libre de riesgo anual (%)", 
                             value=2.0, step=0.25, min_value=0.0, max_value=10.0) / 100
n_mc = st.sidebar.number_input("Número de portafolios para simulación", value=5000, min_value=1000, step=1000)

menu = st.sidebar.selectbox("Seleccione la estrategia:", ["Regiones del Mundo", "Sectores de EE.UU."])

# =============================
# Descargar datos históricos
# =============================
regiones = ['SPLG', 'EWC', 'IEUR', 'EEM', 'EWJ']
sectores = ['XLC','XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLRE','XLK','XLU']

@st.cache_data(show_spinner=False)
def descargar_datos(tickers, start_date, end_date):
    df = yf.download(tickers, start=start_date, end=end_date)['Close']
    return df.pct_change().dropna()

returns_regiones = descargar_datos(regiones, start_date, end_date)
returns_sectores = descargar_datos(sectores, start_date, end_date)

# =============================
# Benchmarks
# =============================
pesos_regiones = np.array([0.7062, 0.0323, 0.1176, 0.0902, 0.0537])
benchmark_regiones = returns_regiones[regiones] @ pesos_regiones

pesos_sectores = np.array([0.0999,0.1025,0.0482,0.0295,0.1307,
                           0.0958,0.0809,0.0166,0.0187,0.3535,0.0237])
benchmark_sectores = returns_sectores[sectores] @ pesos_sectores

# =============================
# Funciones de métricas
# =============================
periods_per_year = 252

def sharpe_ratio(r, rf_rate=rf/periods_per_year, periods=252):
    excess = r - rf_rate
    return np.sqrt(periods) * np.mean(excess) / np.std(excess)

def sortino_ratio(r, rf_rate=rf/periods_per_year, target=0, periods=252):
    excess = r - rf_rate
    downside = r[r < target] - target
    downside_std = np.sqrt(np.mean(downside**2))
    return np.sqrt(periods) * np.mean(excess) / downside_std

def treynor_ratio(r, benchmark, rf_rate=rf/periods_per_year, periods=252):
    excess_r = r - rf_rate
    excess_b = benchmark - rf_rate
    beta = np.cov(excess_r, excess_b)[0,1] / np.var(excess_b)
    return np.mean(excess_r) * periods / beta

def information_ratio(r, benchmark, periods=252):
    active = r - benchmark
    return np.sqrt(periods) * np.mean(active) / np.std(active)

def calmar_ratio(r, periods=252):
    cumulative = (1 + r).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min())
    cagr = (cumulative[-1])**(periods / len(r)) - 1
    return cagr / max_dd if max_dd != 0 else np.nan

def calcular_var(returns, confidence=0.95):
    VaR = returns.quantile(1 - confidence)
    return VaR

def calcular_cvar(returns, confidence=0.95):
    VaR = returns.quantile(1 - confidence)
    CVaR = returns[returns <= VaR].mean()
    return CVaR


def calcular_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calcular_beta(asset_returns, market_returns):
    if isinstance(asset_returns, pd.DataFrame):
        asset_returns = asset_returns.iloc[:, 0]
    if isinstance(market_returns, pd.DataFrame):
        market_returns = market_returns.iloc[:, 0]
    common_index = asset_returns.index.intersection(market_returns.index)
    if len(common_index) < 2:
        return np.nan
    cov = np.cov(asset_returns, market_returns)[0, 1]
    var = np.var(market_returns)

    if var == 0:
        return np.nan

    return cov / var

    
def calcular_metricas_completas(returns, weights, benchmark):
    port_ret = returns @ weights
    return {
        "VaR 95%": calcular_var(port_ret),
        "CVaR 95%": calcular_cvar(port_ret),
        "Beta": calcular_beta(port_ret,benchmark),
        "Drawdow":calcular_drawdown(port_ret),
        "Sharpe": sharpe_ratio(port_ret),
        "Sortino": sortino_ratio(port_ret),
        "Treynor": treynor_ratio(port_ret, benchmark),
        "Information Ratio": information_ratio(port_ret, benchmark),
        "Calmar": calmar_ratio(port_ret) 

    }

# =============================
# Optimización
# =============================
def performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(weights @ cov_matrix @ weights.T)
    return ret, vol

def constraint_sum_weights():
    return {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

def optimize_portfolio(mean_returns, cov_matrix):
    n = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(n))
    x0 = np.ones(n) / n
    results = {}

    # Mínima varianza
    result_min_var = op.minimize(lambda w: performance(w, mean_returns, cov_matrix)[1],
                                 x0, method='SLSQP',
                                 constraints=[constraint_sum_weights()],
                                 bounds=bounds)
    if result_min_var.success:
        w_min = result_min_var.x
        ret_min, vol_min = performance(w_min, mean_returns, cov_matrix)
        results['min_var'] = {'weights': w_min, 'return': ret_min, 'vol': vol_min}

    # Máximo retorno
    result_max_ret = op.minimize(lambda w: -performance(w, mean_returns, cov_matrix)[0],
                                 x0, method='SLSQP',
                                 constraints=[constraint_sum_weights()],
                                 bounds=bounds)
    if result_max_ret.success:
        w_max = result_max_ret.x
        ret_max, vol_max = performance(w_max, mean_returns, cov_matrix)
        results['max_ret'] = {'weights': w_max, 'return': ret_max, 'vol': vol_max}

    # Máximo Sharpe
    def neg_sharpe(w):
        ret, vol = performance(w, mean_returns, cov_matrix)
        return -(ret - rf) / vol
    result_sharpe = op.minimize(neg_sharpe, x0, method='SLSQP',
                                constraints=[constraint_sum_weights()],
                                bounds=bounds)
    if result_sharpe.success:
        w_sharpe = result_sharpe.x
        ret_sharpe, vol_sharpe = performance(w_sharpe, mean_returns, cov_matrix)
        results['max_sharpe'] = {'weights': w_sharpe, 'return': ret_sharpe, 'vol': vol_sharpe}

    return results

def efficient_frontier(mean_returns, cov_matrix, n_points=40):
    n = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(n))
    x0 = np.ones(n) / n
    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret*0.8, max_ret*1.2, n_points)
    frontier_vols, frontier_rets = [], []
    for tr in target_returns:
        constraints = [{'type':'eq','fun':lambda w: np.sum(w)-1},
                       {'type':'eq','fun':lambda w: np.dot(w, mean_returns)-tr}]
        result = op.minimize(lambda w: performance(w, mean_returns, cov_matrix)[1],
                             x0, method='SLSQP',
                             constraints=constraints, bounds=bounds)
        if result.success:
            ret, vol = performance(result.x, mean_returns, cov_matrix)
            frontier_rets.append(ret)
            frontier_vols.append(vol)
    return np.array(frontier_vols), np.array(frontier_rets)

# =============================
# Selección de estrategia y datos activos
# =============================
if menu == "Regiones del Mundo":
    datos = returns_regiones
    universo = regiones
    benchmark = benchmark_regiones
else:
    datos = returns_sectores
    universo = sectores
    benchmark = benchmark_sectores

st.subheader(f"Estrategia seleccionada: {menu}")

# =============================
# Portafolio arbitrario definido por el usuario
# =============================
st.write("### Definir ponderaciones del portafolio")
user_weights = []

for col in datos.columns:
    w = st.slider(f"Ponderación {col}", 0.0, 1.0, 0.1, 0.01)
    user_weights.append(w)

user_weights = np.array(user_weights)
suma = user_weights.sum()

if suma == 0:
    user_weights = np.repeat(1/len(datos.columns), len(datos.columns))
else:
    user_weights = user_weights / suma
    st.success(f"Pesos: Suma total = {user_weights.sum():.4f}")

st.write("#### Pesos finales")
pesos_dict = {col: round(w, 4) for col, w in zip(datos.columns, user_weights)}
st.dataframe(pesos_dict)

# =============================

# Distribución de rendimientos del portafolio del usuario
# =============================
st.write("### Distribución de rendimientos del portafolio arbitrario")
port_ret = datos @ user_weights
fig_hist = px.histogram(port_ret, nbins=40, title='Distribución de rendimientos diarios',
                        labels={'value':'Rendimiento diario'}, color_discrete_sequence=['orange'])
fig_hist.update_layout(showlegend=False)

st.plotly_chart(fig_hist, use_container_width=True)

# =============================
# Métricas del portafolio arbitrario
# =============================
metricas = calcular_metricas_completas(datos, user_weights, benchmark)
st.write("### Métricas del portafolio arbitrario")
st.dataframe(pd.DataFrame(metricas, index=["Portafolio"]).round(4).T, use_container_width=True)

# =============================
# Estadísticas y optimización exacta
# =============================
mean_returns = datos.mean() * periods_per_year
cov_matrix = datos.cov() * periods_per_year



st.write("### Optimización")
results = optimize_portfolio(mean_returns.values, cov_matrix.values)

# Mostrar tablas de pesos y resumen ejecutivo
def format_portfolio_result(name, res, tickers):
    df = pd.DataFrame({'Ticker': tickers, 'Peso': res['weights']})
    summary = {
        'Portafolio': name,
        'Retorno anual': f"{res['return']:.2%}",
        'Volatilidad anual': f"{res['vol']:.2%}",
        'Sharpe (rf)': f"{(res['return'] - rf)/res['vol']:.2f}"
    }
    return df, summary

tables = []
summaries = []
for key, label in [('min_var', 'Mínima varianza'),
                   ('max_sharpe', 'Máximo Sharpe'),
                   ('max_ret', 'Máximo retorno')]:
    if key in results:
        df_res, summary = format_portfolio_result(label, results[key], universo)
        tables.append((label, df_res))
        summaries.append(summary)


st.write("### Resumen de portafolios óptimos")
st.dataframe(pd.DataFrame(summaries))

# =============================
# Frontera eficiente exacta y CML
# =============================
st.write("### Frontera eficiente y línea de mercado de capital (CML)")
front_vols, front_rets = efficient_frontier(mean_returns.values, cov_matrix.values, n_points=40)

fig_front = px.scatter(x=front_vols, y=front_rets,
                       labels={'x': 'Volatilidad anual', 'y': 'Retorno anual'},
                       title='Frontera eficiente')
if 'min_var' in results:
    fig_front.add_scatter(x=[results['min_var']['vol']], y=[results['min_var']['return']],
                          mode='markers', name='Mínima varianza',
                          marker=dict(size=10, color='red'))
if 'max_sharpe' in results:
    fig_front.add_scatter(x=[results['max_sharpe']['vol']], y=[results['max_sharpe']['return']],
                          mode='markers', name='Máximo Sharpe',
                          marker=dict(size=12, color='green', symbol='star'))
    x = np.linspace(0, max(front_vols.max(), results['max_sharpe']['vol'] * 1.2), 300)
    y_cml = rf + ((results['max_sharpe']['return'] - rf) / results['max_sharpe']['vol']) * x
    fig_front.add_scatter(x=x, y=y_cml, mode='lines', name='CML',
                          line=dict(color='green', dash='dash'))
if 'max_ret' in results:
    fig_front.add_scatter(x=[results['max_ret']['vol']], y=[results['max_ret']['return']],
                          mode='markers', name='Máximo retorno',
                          marker=dict(size=10, color='purple'))
st.plotly_chart(fig_front, use_container_width=True)


# =============================
# visualizaciones
# =============================
exp_returns = mean_returns.values
cov_values = cov_matrix.values
num_assets = len(exp_returns)

mc_results = []
for _ in range(int(n_mc)):
    w = np.random.random(num_assets)
    w /= w.sum()
    ret = np.dot(w, exp_returns)
    vol = np.sqrt(w @ cov_values @ w.T)
    sharpe = (ret - rf) / vol
    mc_results.append([ret, vol, sharpe, w])
df_mc = pd.DataFrame(mc_results, columns=['Retorno','Volatilidad','Sharpe','Pesos'])

best_sharpe = df_mc.loc[df_mc['Sharpe'].idxmax()]
min_vol = df_mc.loc[df_mc['Volatilidad'].idxmin()]

# =============================
#  Portafolio vs Benchmark
# =============================
st.write("###  Portafolios vs Benchmark")
capital_port = (1 + port_ret).cumprod()

capital_bmk = (1 + benchmark.loc[port_ret.index]).cumprod()
tabs = st.tabs(["Portafolio Arbitrario", "Max Sharpe", "Minima Volatilidad"])

with tabs[0]:
    st.write("### Evolución del Portafolio Arbitrario vs Benchmark")

    df_capital = pd.DataFrame({
        'Portafolio': capital_port,
        'Benchmark': capital_bmk
    })

    fig_capital = px.line(df_capital,
                          title="Portafolio Arbitrario vs Benchmark",
                          labels={'value': 'Capital acumulado'})
    st.plotly_chart(fig_capital, use_container_width=True)

with tabs[1]:
    st.write("### Portafolio de Máximo Sharpe vs Benchmark")
    capital_sharpe = (1 + datos.dot(best_sharpe["Pesos"])).cumprod()
    df_sharpe = pd.DataFrame({
        'Max Sharpe': capital_sharpe,
        'Benchmark': capital_bmk
    })
    fig_sharpe = px.line(df_sharpe,
                         title="Portafolio de Máximo Sharpe vs Benchmark",
                         labels={'value': 'Capital acumulado'})
    st.plotly_chart(fig_sharpe, use_container_width=True)

with tabs[2]:
    st.write("### Portafolio de Mínima Volatilidad vs Benchmark")
    capital_minvol = (1 + datos.dot(min_vol["Pesos"])).cumprod()
    df_minvol = pd.DataFrame({
        'Mínima Volatilidad': capital_minvol,
        'Benchmark': capital_bmk
    })
    fig_minvol = px.line(df_minvol,
                         title="Portafolio de Mínima Volatilidad vs Benchmark",
                         labels={'value': 'Capital acumulado'})
    st.plotly_chart(fig_minvol, use_container_width=True)

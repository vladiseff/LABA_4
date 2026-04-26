import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Исходные данные для Варианта 18 ---
S0 = 30.0
r = 0.1
T = 4.0
h = 0.004
N = int(T / h)
t_arr = np.linspace(0, T, N + 1)

# Ваши недельные котировки (Приложение 1)
weekly_prices = [
    50.00, 49.67, 49.63, 49.74, 50.22, 48.36, 48.74, 50.08, 49.63, 48.39,
    48.19, 48.95, 48.57, 48.34, 49.02, 48.85, 49.45, 49.46, 48.48, 47.87,
    46.51, 48.55, 47.96, 48.26, 47.74, 48.80, 48.31, 49.16, 48.18, 47.65,
    46.19, 46.17, 46.05, 46.28, 44.47, 44.02, 43.00, 42.59, 43.23, 43.52,
    45.20, 46.11, 46.24, 46.50, 45.64, 46.90, 46.43, 46.67, 46.91, 47.64, 48.21
]


# Функции из условия
def mu_func(t): return -0.1 * t


def sigma_func(t): return 0.05 * t + 0.1  # 5t+10% -> 0.05t + 0.1


def avg_sigma_squared(t1, t2):
    def integral(u):
        return (0.0025 * u ** 3) / 3 + (0.01 * u ** 2) / 2 + 0.01 * u

    if t2 == t1: return (sigma_func(t1)) ** 2
    return (integral(t2) - integral(t1)) / (t2 - t1)


# Симуляция базового актива (Лаб 3, п.4) [cite: 6]
np.random.seed(42)
S_path = np.zeros(N + 1)
S_path[0] = S0
for i in range(1, N + 1):
    dW = np.sqrt(h) * np.random.randn()
    t_prev = t_arr[i - 1]
    S_path[i] = S_path[i - 1] * np.exp((mu_func(t_prev) - 0.5 * sigma_func(t_prev) ** 2) * h + sigma_func(t_prev) * dW)

# === ЗАДАНИЕ 1: Цена опциона продавца (Put) ===
print("--- ЗАДАНИЕ 1 ---")
E_put = 7 * S0 / 8
node = 250
t_curr = node * h  # 1.0 год
tau = T - t_curr  # 3.0 года
S_curr = S_path[node]
vol_sq = avg_sigma_squared(t_curr, T)
vol_avg = np.sqrt(vol_sq)

d1 = (np.log(S_curr / E_put) + (r + vol_sq / 2) * tau) / (vol_avg * np.sqrt(tau))
d2 = d1 - vol_avg * np.sqrt(tau)
P = E_put * np.exp(-r * tau) * norm.cdf(-d2) - S_curr * norm.cdf(-d1)
print(f"Цена Put-опциона (T-t=3): {P:.4f}\n")

# === ЗАДАНИЕ 2 & 3: Хеджирование и Портфель ===
print("--- ЗАДАНИЕ 2 & 3 ---")
t_steps = np.arange(0, T + 0.5, 0.5)
delta_C = []
portfolio_value = []

for t_i in t_steps:
    idx = int(t_i / h)
    S_i = S_path[idx]
    tau_i = T - t_i
    if tau_i > 0:
        v_sq = avg_sigma_squared(t_i, T)
        d1_i = (np.log(S_i / E_put) + (r + v_sq / 2) * tau_i) / (np.sqrt(v_sq) * np.sqrt(tau_i))
        d2_i = d1_i - np.sqrt(v_sq) * np.sqrt(tau_i)
        Call_i = S_i * norm.cdf(d1_i) - E_put * np.exp(-r * tau_i) * norm.cdf(d2_i)
        delta_i = norm.cdf(d1_i)
    else:
        Call_i = max(S_i - E_put, 0)
        delta_i = 1.0 if S_i > E_put else 0.0

    delta_C.append(delta_i)
    portfolio_value.append(delta_i * S_i - Call_i)

plt.figure(figsize=(8, 4))
plt.plot(t_steps, portfolio_value, 'b-o', label=r'$\Pi_t = \Delta_t S_t - C_t$')
plt.title('Стоимость риск-нейтрального портфеля')
plt.xlabel('t (годы)')
plt.ylabel(r'Стоимость $\Pi$')
plt.grid(True)
plt.legend()
plt.show()

# === ЗАДАНИЕ 4: Реализованная волатильность по ВАШИМ данным ===
print("--- ЗАДАНИЕ 4 ---")
df_weekly = pd.DataFrame({'S_j': weekly_prices})
df_weekly['Returns'] = np.log(df_weekly['S_j'] / df_weekly['S_j'].shift(1))
# Несмещенная оценка дисперсии [cite: 4]
var_weekly = df_weekly['Returns'].var(ddof=1)
sigma_realized = np.sqrt(var_weekly * 52)  # Годовая волатильность
print(f"Реализованная волатильность sigma: {sigma_realized:.4f}\n")

# === ЗАДАНИЕ 5: Греки ===
print("--- ЗАДАНИЕ 5 ---")
E_5 = 50.0
n_weeks = len(weekly_prices)
df_weekly['tau'] = np.linspace(1, 0, n_weeks)  # Время от 1 до 0

deltas, gammas, thetas = [], [], []

for i, row in df_weekly.iterrows():
    sj, tauj = row['S_j'], row['tau']
    if tauj <= 0.0001:
        deltas.append(1.0 if sj > E_5 else 0.0)
        gammas.append(0.0)
        thetas.append(0.0)
    else:
        d1 = (np.log(sj / E_5) + (r + 0.5 * sigma_realized ** 2) * tauj) / (sigma_realized * np.sqrt(tauj))
        d2 = d1 - sigma_realized * np.sqrt(tauj)
        deltas.append(norm.cdf(d1))
        gammas.append(norm.pdf(d1) / (sj * sigma_realized * np.sqrt(tauj)))
        thetas.append(
            -(sj * norm.pdf(d1) * sigma_realized) / (2 * np.sqrt(tauj)) - r * E_5 * np.exp(-r * tauj) * norm.cdf(d2))

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1);
plt.plot(df_weekly['tau'], deltas);
plt.title(r'Delta ($\Delta$)');
plt.gca().invert_xaxis()
plt.subplot(1, 3, 2);
plt.plot(df_weekly['tau'], gammas);
plt.title(r'Gamma ($\Gamma$)');
plt.gca().invert_xaxis()
plt.subplot(1, 3, 3);
plt.plot(df_weekly['tau'], thetas);
plt.title(r'Theta ($\Theta$)');
plt.gca().invert_xaxis()
plt.tight_layout();
plt.show()